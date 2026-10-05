#!/usr/bin/env python3
"""训练 FreeChunker 的跨粒度编码器（jina-embeddings-v2-small-en 版）。

相对原脚本的修改：

  1. **路径统一从 src.paths 取**。原脚本读 `./vector/jina-embeddings-v2-small-en/train`，
     而造数据脚本的输出路径与它对不上（已在 1b_merge_parts.py 里补齐）。
  2. **断点续跑**：每 --ckpt-interval 步存一次（模型+优化器+调度器+步数+loss 历史+RNG），
     启动时自动从最新断点续。原脚本只在 epoch 结束时存，中途挂了整轮白跑
     —— 这是 5 小时以上的无人值守任务，必须能续。
  3. **去掉 DataLoader，改用显式 per-epoch 置换**。batch=1 时 DataLoader 只是白搭一层；
     显式置换让「续跑从第 N 步开始」是 O(1) 的，不用空转 N 次数据加载。
     分布上与原脚本的 shuffle=True 等价（都是均匀随机排列）。
  4. **matplotlib 用 Agg 后端**，无人值守没有 DISPLAY。
  5. 打印步速与 ETA。
  6. 注意：`inputs_embeds` 必须带 batch 维（`[1, n, 512]`）。原脚本靠 DataLoader 的
     collate 自动加这一维，去掉 DataLoader 后要自己 unsqueeze。
  7. **数据集加 `with_format("numpy")`**。这是本脚本最大的一处性能改动：
     HF datasets 默认把 Arrow 行转成 Python 嵌套列表，实测 `ds[i]` 单次要 **61.9 ms**
     （要构造 15 万个 Python float），占整步 180 ms 的三分之一；
     换成 numpy 格式后同样的行只要 **2.5 ms**（直接返回 [n,512] 的 float32 ndarray）。
     20 万步下来省约 **3.3 小时**，且数值完全一致（启动时有一致性断言）。

**超参与原脚本完全一致**：AdamW lr=1e-4、batch=1、2 epoch、
cosine schedule + 前 1/3 步 warmup、每 1000 步用 200 条样本验证。
官方参考曲线（models/FreeChunk-jina/training_losses.json）：
    total_steps=200000, 前1000步均值=0.4497, 末1000步均值=0.0109, final_val=0.0103
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")          # 无人值守没有 DISPLAY
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup

from src.freechunker import FreeChunkerModel
from src.paths import (
    BGE_M3_LOCAL, OFFICIAL_FINAL_VAL, OFFICIAL_FIRST1000_MEAN,
    OFFICIAL_LAST1000_MEAN, OFFICIAL_TOTAL_STEPS, SAVE_DIR, load_vector_split,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--ckpt-interval", type=int, default=10000)
    ap.add_argument("--val-samples", type=int, default=200)
    ap.add_argument("--save-dir", default=SAVE_DIR)
    ap.add_argument("--base-model", default=BGE_M3_LOCAL,
                    help="跨粒度编码器的初始化权重（原脚本写死 BAAI/bge-m3，三个 backbone 共用）")
    ap.add_argument("--resume", action="store_true", help="从最新断点续跑")
    ap.add_argument("--max-steps", type=int, default=None, help="只跑 N 步（冒烟用）")
    ap.add_argument("--no-save", action="store_true", help="不落盘（冒烟用）")
    return ap.parse_args()


def row_tensor(it, key, dev):
    """把一行 [n, 512] 的向量转成模型要的 [1, n, 512] tensor。

    数据来自 `with_format("numpy")` 的 Dataset，`it[key]` 已经是 float32 ndarray；
    这里再走一次 np.asarray 只是兜底（万一某天格式变回 list）。
    """
    arr = np.asarray(it[key], dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float, device=dev).unsqueeze(0)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    log_f = open(os.path.join(args.save_dir, "train_log.txt"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("=" * 70)
    log(f"save_dir={args.save_dir}")
    log(f"device={dev} {torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''}")

    # with_format("numpy")：见文件头第 7 条，把 ds[i] 从 61.9ms 降到 2.5ms
    train_ds = load_vector_split("train").with_format("numpy")
    val_ds_full = load_vector_split("val").with_format("numpy")
    val_ds = val_ds_full.select(range(min(args.val_samples, len(val_ds_full))))
    log(f"train={len(train_ds)} val={len(val_ds)} | features={train_ds.features}")

    # 一次性自检：numpy 格式与原（python 列表）格式必须逐位一致
    try:
        ref = load_vector_split("train")
        a = np.asarray(ref[0]["input"], dtype=np.float32)
        b = np.asarray(train_ds[0]["input"], dtype=np.float32)
        la = np.asarray(ref[0]["label"], dtype=np.float32)
        lb = np.asarray(train_ds[0]["label"], dtype=np.float32)
        assert a.shape == b.shape and la.shape == lb.shape
        assert np.array_equal(a, b) and np.array_equal(la, lb)
        log(f"数据格式自检 OK：numpy 与默认格式逐位一致，input{a.shape} label{la.shape}")
        del ref, a, b, la, lb
    except Exception as e:  # noqa: BLE001
        log(f"!! 数据格式自检失败：{type(e).__name__}: {e}")
        raise

    steps_per_epoch = len(train_ds)
    num_training_steps = args.epochs * steps_per_epoch
    log(f"epochs={args.epochs} steps/epoch={steps_per_epoch} total_steps={num_training_steps}")
    if args.max_steps:
        log(f"!! --max-steps={args.max_steps}，这是冒烟模式，不是正式训练")

    log(f"加载初始化权重 {args.base_model}")
    model = FreeChunkerModel.from_pretrained(args.base_model, ignore_mismatched_sizes=True)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"参数量 {n_par/1e6:.1f}M | attn_impl={model.config._attn_implementation}")
    model = model.to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_warmup_steps = num_training_steps // 3
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    log(f"AdamW lr={args.lr} | warmup={num_warmup_steps} | cosine -> {num_training_steps}")

    step_losses, train_losses, val_losses, val_step_losses = [], [], [], []
    global_step = 0
    start_epoch = 0
    skip_in_epoch = 0

    ckpt_path = os.path.join(args.save_dir, "ckpt_latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        log(f"从断点续跑 {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        global_step = ck["global_step"]
        start_epoch = ck["epoch"]
        skip_in_epoch = ck["step_in_epoch"]
        step_losses = ck["step_losses"]
        train_losses = ck["train_losses"]
        val_losses = ck["val_losses"]
        val_step_losses = ck["val_step_losses"]
        try:
            torch.set_rng_state(ck["torch_rng"])
            np.random.set_state(ck["numpy_rng"])
            random.setstate(ck["python_rng"])
            if dev.type == "cuda" and ck.get("cuda_rng"):
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
        except Exception as e:  # noqa: BLE001
            log(f"!! RNG 状态恢复失败（不影响继续训练，只是随机序会变）: {e}")
        log(f"续跑点: epoch={start_epoch} 已完成 {skip_in_epoch} 步/该 epoch，global_step={global_step}")
    elif args.resume:
        log("--resume 但没找到断点，从头开始")

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, cnt = 0.0, 0
        for j in range(len(val_ds)):
            it = val_ds[j]
            out = model(inputs_embeds=row_tensor(it, "input", dev),
                        labels=row_tensor(it, "label", dev))
            tot += float(out["loss"].item())
            cnt += 1
        model.train()
        return tot / max(cnt, 1)

    def plot(cur_step):
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(step_losses) + 1), step_losses,
                 label="Train Loss (per step)", color="blue", alpha=0.7, linewidth=0.8)
        eval_steps = list(range(args.eval_interval, len(step_losses) + 1, args.eval_interval))
        if len(eval_steps) == len(val_losses):
            plt.plot(eval_steps, val_losses, label="Validation Loss (eval interval)",
                     color="red", linewidth=2)
        plt.xlabel("Training Step"); plt.ylabel("Loss"); plt.legend()
        plt.title("Training Loss (per step) vs Validation Loss (eval interval)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.save_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()

    def save_ckpt(epoch, step_in_epoch):
        if args.no_save:
            return
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step, "epoch": epoch, "step_in_epoch": step_in_epoch,
            "step_losses": step_losses, "train_losses": train_losses,
            "val_losses": val_losses, "val_step_losses": val_step_losses,
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
        }, ckpt_path)

    model.train()
    t_start = time.time()
    t_win = time.time()
    win = 0

    for epoch in range(start_epoch, args.epochs):
        log(f"===== Epoch {epoch+1}/{args.epochs} =====")
        # 显式 per-epoch 置换：可续跑、可复现
        g = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        perm = torch.randperm(steps_per_epoch, generator=g).tolist()
        skip = skip_in_epoch if epoch == start_epoch else 0
        if skip:
            log(f"  跳过前 {skip} 步（已训练过）")

        for step_in_epoch, idx in enumerate(perm[skip:], start=skip + 1):
            it = train_ds[idx]
            x = row_tensor(it, "input", dev)
            y = row_tensor(it, "label", dev)

            optimizer.zero_grad(set_to_none=True)
            out = model(inputs_embeds=x, labels=y)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()

            lv = float(loss.item())
            step_losses.append(lv)
            global_step += 1
            win += 1

            if global_step % 100 == 0:
                el = time.time() - t_win
                ms = el / win * 1000
                remain = (num_training_steps - global_step) * ms / 1000
                log(f"  step {global_step}/{num_training_steps} (epoch {epoch+1} "
                    f"{step_in_epoch}/{steps_per_epoch}) loss={lv:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f} "
                    f"{ms:.0f}ms/step ETA={remain/3600:.1f}h")
                t_win, win = time.time(), 0

            if global_step % args.eval_interval == 0:
                vl = evaluate()
                train_losses.append(sum(step_losses[-args.eval_interval:]) / args.eval_interval)
                val_losses.append(vl)
                val_step_losses.append(vl)
                log(f"  [eval] step {global_step} train_loss(avg last {args.eval_interval})="
                    f"{train_losses[-1]:.4f} val_loss={vl:.4f}")
                plot(global_step)

            if global_step % args.ckpt_interval == 0:
                save_ckpt(epoch, step_in_epoch)
                log(f"  [ckpt] 已存断点 @ step {global_step}")

            if args.max_steps and global_step >= args.max_steps:
                log(f"达到 --max-steps={args.max_steps}，提前退出")
                break

        avg = sum(step_losses[-steps_per_epoch:]) / max(len(step_losses[-steps_per_epoch:]), 1)
        vl = evaluate()
        train_losses.append(avg)
        val_losses.append(vl)
        val_step_losses.append(vl)
        log(f"Epoch {epoch+1} 结束: avg_train_loss={avg:.4f} val_loss={vl:.4f}")

        if not args.no_save:
            ep_dir = os.path.join(args.save_dir, f"epoch_{epoch}")
            model.save_pretrained(ep_dir)
            log(f"  已保存 {ep_dir}")
            save_ckpt(epoch + 1, 0)

        if args.max_steps and global_step >= args.max_steps:
            break

    dt = time.time() - t_start
    log(f"训练结束，共 {global_step} 步，本次耗时 {dt/3600:.2f} h")

    if not args.no_save and step_losses:
        final_dir = os.path.join(args.save_dir, "final")
        model.save_pretrained(final_dir)
        log(f"已保存最终权重 {final_dir}")

        plot(global_step)
        data = {
            "training_losses": {"step_losses": step_losses,
                                "steps": list(range(1, len(step_losses) + 1))},
            "validation_losses": {
                "val_losses": val_losses,
                "eval_steps": list(range(args.eval_interval, len(step_losses) + 1, args.eval_interval)),
            },
            "training_config": {"eval_interval": args.eval_interval, "num_epochs": args.epochs,
                                "total_steps": len(step_losses), "lr": args.lr, "seed": args.seed},
            "summary": {
                "final_train_loss": step_losses[-1] if step_losses else None,
                "final_val_loss": val_losses[-1] if val_losses else None,
                "min_train_loss": min(step_losses) if step_losses else None,
                "min_val_loss": min(val_losses) if val_losses else None,
                "first1000_mean": (sum(step_losses[:1000]) / 1000) if len(step_losses) >= 1000 else None,
                "last1000_mean": (sum(step_losses[-1000:]) / 1000) if len(step_losses) >= 1000 else None,
            },
        }
        p = os.path.join(args.save_dir, "training_losses.json")
        json.dump(data, open(p, "w", encoding="utf-8"), indent=2)
        log(f"已保存 {p}")

        s = data["summary"]
        log("=" * 70)
        log("G2 验收对照（官方值）")
        log(f"  total_steps       : {len(step_losses):>10}   (官方 {OFFICIAL_TOTAL_STEPS})")
        if s["first1000_mean"] is not None:
            log(f"  前1000步均值      : {s['first1000_mean']:>10.4f}   (官方 {OFFICIAL_FIRST1000_MEAN})")
        if s["last1000_mean"] is not None:
            log(f"  末1000步均值      : {s['last1000_mean']:>10.4f}   (官方 {OFFICIAL_LAST1000_MEAN})")
        log(f"  final_train_loss  : {s['final_train_loss']:>10.4f}")
        log(f"  final_val_loss    : {s['final_val_loss']:>10.4f}   (官方 {OFFICIAL_FINAL_VAL})")
        log("=" * 70)

    log("TRAIN_DONE")
    log_f.close()


if __name__ == "__main__":
    main()
