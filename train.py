from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data import PairedINbreastDataset, build_metadata
from evaluate import best_threshold, choose_heatmap_threshold, classification_metrics, dice_iou, localization_summary
from gradcam import paired_gradcam, save_overlay
from model import create_model


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def worker_seed(worker_id: int) -> None:
    value = torch.initial_seed() % 2**32
    np.random.seed(value); random.seed(value)


def make_loader(frame: pd.DataFrame, args, training: bool, batch_size: int | None = None) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(PairedINbreastDataset(frame, args.size, training), batch_size=batch_size or args.batch_size,
        shuffle=training, num_workers=args.workers, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_seed,
        generator=generator, persistent_workers=args.workers > 0)


def run_epoch(model, loader, loss_fn, optimizer, scaler, device, training: bool):
    model.train(training); total, labels, probabilities = 0., [], []
    for batch in loader:
        cc, mlo, targets = batch["cc_image"].to(device), batch["mlo_image"].to(device), batch["label"].to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(cc, mlo); loss = loss_fn(logits, targets)
        if training:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer); scaler.update()
        total += loss.item() * len(targets); labels.extend(targets.cpu().int().tolist()); probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return total / max(1, len(loader.dataset)), labels, probabilities


def save_sanity(pairs: pd.DataFrame, args) -> None:
    sample = pairs.sample(min(4, len(pairs)), random_state=args.seed)
    dataset = PairedINbreastDataset(sample, args.size, False)
    fig, axes = plt.subplots(4, 2, figsize=(10, 18), squeeze=False)
    for row_index in range(4):
        for column, view in enumerate(("cc", "mlo")):
            axis = axes[row_index, column]
            if row_index >= len(dataset): axis.axis("off"); continue
            item = dataset[row_index]; image = denormalize(item[f"{view}_image"]); mask = item[f"{view}_mask"][0].numpy()
            axis.imshow(image, cmap="gray"); axis.imshow(np.ma.masked_where(mask == 0, mask), cmap="Reds", alpha=.45)
            axis.set_title(f"{item['pair_id']} {view.upper()} label={int(item['label'])}"); axis.axis("off")
    path = args.output_dir / "sanity" / "paired_overlays.png"; path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig); print(f"Sanity check: {path}")


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    return np.clip(tensor[0].detach().cpu().numpy() * .229 + .485, 0, 1)


def save_history(history: list[dict], output: Path) -> None:
    frame = pd.DataFrame(history); frame.to_csv(output / "history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)); axes[0].plot(frame.epoch, frame.train_loss, label="train"); axes[0].plot(frame.epoch, frame.val_loss, label="validation")
    axes[0].legend(); axes[0].set_title("Loss"); axes[1].plot(frame.epoch, frame.f1_at_05); axes[1].set_title("Validation F1 @ 0.5")
    fig.tight_layout(); fig.savefig(output / "training_curves.png", dpi=150); plt.close(fig)


def train_model(pairs: pd.DataFrame, args, device: torch.device) -> None:
    train_frame, val_frame = pairs[pairs.split == "train"], pairs[pairs.split == "val"]
    train_loader, val_loader = make_loader(train_frame, args, True), make_loader(val_frame, args, False)
    model = create_model(args.size, pretrained=not args.no_pretrained, dropout=args.dropout).to(device)
    model.freeze_backbone(args.freeze_epochs > 0)
    positives, negatives = int(train_frame.label.sum()), int((train_frame.label == 0).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(max(1, negatives) / max(1, positives), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, stale, history, first_epoch = -1., 0, [], 1
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        best = float(saved.get("metrics_at_05", {}).get("f1", -1.)); first_epoch = int(saved["epoch"]) + 1
        if first_epoch > args.freeze_epochs:
            model.freeze_backbone(False)
        print(f"Nastavak treninga od epohe {first_epoch}: {args.resume}")
    for epoch in range(first_epoch, args.epochs + 1):
        if epoch == args.freeze_epochs + 1: model.freeze_backbone(False)
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, True)
        val_loss, labels, probabilities = run_epoch(model, val_loader, loss_fn, optimizer, scaler, device, False)
        fixed = classification_metrics(labels, probabilities, .5); threshold, tuned_f1 = best_threshold(labels, probabilities); scheduler.step(fixed["f1"])
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "f1_at_05": fixed["f1"], "tuned_f1": tuned_f1, "tuned_threshold": threshold, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row); print(f"Epoch {epoch:03d}: train={train_loss:.4f} val={val_loss:.4f} F1@.5={fixed['f1']:.4f}")
        if fixed["f1"] > best:
            best, stale = fixed["f1"], 0
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch,
                "size": args.size, "dropout": args.dropout, "threshold": threshold, "metrics_at_05": fixed,
                "metrics_tuned": classification_metrics(labels, probabilities, threshold), "config": vars(args)}, args.output_dir / "best.pt")
        else: stale += 1
        if stale >= args.patience: print("Early stopping."); break
    save_history(history, args.output_dir)


def load_checkpoint(args, device):
    checkpoint = torch.load(args.checkpoint or args.output_dir / "best.pt", map_location=device, weights_only=False)
    model = create_model(checkpoint["size"], pretrained=False, dropout=checkpoint.get("dropout", .3)).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    return model, checkpoint


def collect_localization(model, frame: pd.DataFrame, args, device, save: bool = False):
    loader = make_loader(frame, args, False, batch_size=1); heatmaps, masks, records = [], [], []
    for batch in loader:
        cc, mlo = batch["cc_image"].to(device), batch["mlo_image"].to(device)
        cc_cam, mlo_cam = paired_gradcam(model, cc, mlo)
        for view, cam in (("CC", cc_cam), ("MLO", mlo_cam)):
            prefix = view.lower(); has_roi = bool(batch[f"{prefix}_has_roi"].item()); mask = batch[f"{prefix}_mask"][0, 0].numpy()
            if has_roi and mask.any(): heatmaps.append(cam); masks.append(mask); records.append({"pair_id": batch["pair_id"][0], "view": view, "cam": cam, "mask": mask})
            if save:
                save_overlay(denormalize(batch[f"{prefix}_image"][0]), cam, mask, args.output_dir / "gradcam" / f"{batch['pair_id'][0]}_{view}.png", f"{batch['pair_id'][0]} {view}")
    return heatmaps, masks, records


def evaluate_model(pairs: pd.DataFrame, args, device, predict_one: bool = False) -> None:
    model, checkpoint = load_checkpoint(args, device)
    validation = pairs[pairs.split == "val"]
    if predict_one: validation = validation.iloc[:1]
    loader = make_loader(validation, args, False, batch_size=1 if predict_one else args.batch_size)
    loss = nn.BCEWithLogitsLoss(); _, labels, probabilities = run_epoch(model, loader, loss, None, torch.amp.GradScaler("cuda", enabled=False), device, False)
    result = {"classification_at_05": classification_metrics(labels, probabilities, .5),
              "classification_tuned": classification_metrics(labels, probabilities, checkpoint.get("threshold", .5))}
    train_heatmaps, train_masks, _ = collect_localization(model, pairs[pairs.split == "train"], args, device)
    heatmap_threshold = choose_heatmap_threshold(train_heatmaps, train_masks)
    _, _, records = collect_localization(model, validation, args, device, save=True)
    rows = []
    for record in records:
        dice, iou = dice_iou(record["cam"] >= heatmap_threshold, record["mask"])
        rows.append({"pair_id": record["pair_id"], "view": record["view"], "dice": dice, "iou": iou})
    result["localization_threshold"] = heatmap_threshold; result["localization"] = localization_summary(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output_dir / "localization_metrics.csv", index=False); print(json.dumps(result, indent=2))


def parse_args():
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument("--config", type=Path); known, _ = pre.parse_known_args()
    defaults = yaml.safe_load(known.config.read_text(encoding="utf-8")) if known.config else {}
    parser = argparse.ArgumentParser(parents=[pre]); parser.set_defaults(**(defaults or {}))
    parser.add_argument("--data-root", type=Path, required="data_root" not in (defaults or {})); parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--mode", choices=("prepare", "sanity", "train", "evaluate", "predict"), default="sanity")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts")); parser.add_argument("--checkpoint", type=Path); parser.add_argument("--resume", type=Path)
    parser.add_argument("--size", type=int, default=384); parser.add_argument("--batch-size", type=int, default=2); parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--workers", type=int, default=0); parser.add_argument("--lr", type=float, default=2e-5); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=.3); parser.add_argument("--freeze-epochs", type=int, default=2); parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    for name in ("data_root", "metadata_path", "output_dir", "checkpoint", "resume"):
        value = getattr(args, name, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, name, Path(value))
    return args


def main():
    args = parse_args(); seed_everything(args.seed); args.output_dir.mkdir(parents=True, exist_ok=True)
    _, pairs = build_metadata(args.data_root, args.output_dir, args.metadata_path, args.seed)
    print(f"Parovi: {len(pairs)}, pacijenti: {pairs.patient_id.nunique()}, train: {(pairs.split == 'train').sum()}, val: {(pairs.split == 'val').sum()}")
    if args.mode == "prepare": return
    if args.mode == "sanity": save_sanity(pairs, args); return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Device: {device}")
    if args.mode == "train": train_model(pairs, args, device)
    else: evaluate_model(pairs, args, device, predict_one=args.mode == "predict")


if __name__ == "__main__": main()
