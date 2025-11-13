import os, argparse, yaml, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from cyclone_locator.datasets.med_fullbasin import MedFullBasinDataset
from cyclone_locator.models.simplebaseline import SimpleBaseline
from cyclone_locator.losses.heatmap_loss import HeatmapMSE

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yml")
    ap.add_argument("--train_csv")
    ap.add_argument("--val_csv")
    ap.add_argument("--image_size", type=int)
    ap.add_argument("--heatmap_stride", type=int)
    ap.add_argument("--heatmap_sigma_px", type=float)
    ap.add_argument("--backbone")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--bs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--log_dir")
    return ap.parse_args()

def set_seed(sd):
    import random, numpy as np, torch
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)

def bce_logits(pred, target):
    return nn.functional.binary_cross_entropy_with_logits(pred, target)

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    # Optional CLI overrides keep backwards-compat with config-driven flow.
    if args.train_csv:
        cfg["data"]["manifest_train"] = args.train_csv
    if args.val_csv:
        cfg["data"]["manifest_val"] = args.val_csv
    if args.image_size:
        cfg["train"]["image_size"] = args.image_size
    if args.heatmap_stride:
        cfg["train"]["heatmap_stride"] = args.heatmap_stride
    if args.heatmap_sigma_px:
        cfg["loss"]["heatmap_sigma_px"] = args.heatmap_sigma_px
    if args.backbone:
        cfg["train"]["backbone"] = args.backbone
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.bs:
        cfg["train"]["batch_size"] = args.bs
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.log_dir:
        cfg["train"]["save_dir"] = args.log_dir

    set_seed(cfg["train"]["seed"])

    # Datasets
    ds_tr = MedFullBasinDataset(
        cfg["data"]["manifest_train"],
        image_size=cfg["train"]["image_size"],
        heatmap_stride=cfg["train"]["heatmap_stride"],
        heatmap_sigma_px=cfg["loss"]["heatmap_sigma_px"],
        use_aug=cfg["train"]["use_aug"],
        use_pre_letterboxed=cfg["data"]["use_pre_letterboxed"],
        letterbox_meta_csv=cfg["data"]["letterbox_meta_csv"],
        letterbox_size_assert=cfg["data"]["letterbox_size_assert"]
    )
    ds_va = MedFullBasinDataset(
        cfg["data"]["manifest_val"],
        image_size=cfg["train"]["image_size"],
        heatmap_stride=cfg["train"]["heatmap_stride"],
        heatmap_sigma_px=cfg["loss"]["heatmap_sigma_px"],
        use_aug=False,
        use_pre_letterboxed=cfg["data"]["use_pre_letterboxed"],
        letterbox_meta_csv=cfg["data"]["letterbox_meta_csv"],
        letterbox_size_assert=cfg["data"]["letterbox_size_assert"]
    )
    tr_loader = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True,
                           num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
    va_loader = DataLoader(ds_va, batch_size=cfg["train"]["batch_size"], shuffle=False,
                           num_workers=cfg["train"]["num_workers"], pin_memory=True)

    # Model
    model = SimpleBaseline(backbone=cfg["train"]["backbone"], out_heatmap_ch=1)
    model = model.cuda()

    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=cfg["train"]["amp"])
    hm_loss = HeatmapMSE()

    save_dir = cfg["train"]["save_dir"]; os.makedirs(save_dir, exist_ok=True)
    best_val = 1e9; best_path = None

    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        losses = []; hm_losses = []; pres_losses = []
        for batch in tr_loader:
            img = batch["image"].cuda(non_blocking=True)
            hm_t = batch["heatmap"].cuda(non_blocking=True)
            pres = batch["presence"].cuda(non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["train"]["amp"]):
                hm_p, pres_logit = model(img)
                L_hm = hm_loss(hm_p, hm_t, pres.view(-1))                # solo positivi
                L_pr = bce_logits(pres_logit, pres)                      # tutti
                L = cfg["loss"]["w_heatmap"]*L_hm + cfg["loss"]["w_presence"]*L_pr
            scaler.scale(L).backward()
            scaler.step(opt); scaler.update()

            losses.append(L.item()); hm_losses.append(L_hm.item()); pres_losses.append(L_pr.item())

        print(f"[Epoch {epoch}] train: L={np.mean(losses):.4f} (hm={np.mean(hm_losses):.4f}, pr={np.mean(pres_losses):.4f})")

        if epoch % cfg["train"]["val_every"] == 0:
            model.eval()
            with torch.no_grad():
                vL, vHm, vPr = [], [], []
                for batch in va_loader:
                    img = batch["image"].cuda()
                    hm_t = batch["heatmap"].cuda()
                    pres = batch["presence"].cuda()
                    hm_p, pres_logit = model(img)
                    L_hm = hm_loss(hm_p, hm_t, pres.view(-1))
                    L_pr = bce_logits(pres_logit, pres)
                    L = cfg["loss"]["w_heatmap"]*L_hm + cfg["loss"]["w_presence"]*L_pr
                    vL.append(L.item()); vHm.append(L_hm.item()); vPr.append(L_pr.item())
                val_score = np.mean(vL)
                print(f"          val:  L={val_score:.4f} (hm={np.mean(vHm):.4f}, pr={np.mean(vPr):.4f})")

            ckpt_path = os.path.join(save_dir, f"epoch{epoch:03d}_val{val_score:.4f}.ckpt")
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
            if val_score < best_val:
                best_val = val_score; best_path = ckpt_path
                torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(save_dir, "best.ckpt"))
                print("Saved best:", os.path.join(save_dir, "best.ckpt"))

    print("Done. Best:", best_path)

if __name__ == "__main__":
    main()
