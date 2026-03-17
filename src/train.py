import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils import get_logger, save_checkpoint, load_checkpoint


def train(config: dict):
    logger = get_logger("train", log_dir=config["paths"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    use_features = config.get("use_features", True)

    # --- Dataset ---
    if use_features:
        from src.dataset import THUMOSFeatureDataset
        dataset = THUMOSFeatureDataset(
            feature_dir=config["paths"]["feature_dir"],
            ann_path=os.path.join(
                config["paths"]["data_root"], "annotations", "thumos_14_anno.json"
            ),
            window_size=config["data"].get("window_size", 128),
            stride=config["data"].get("stride", 64),
        )
        # Infer feature dim from the first sample
        sample_feat, _ = dataset[0]
        feature_dim = sample_feat.shape[-1]
        logger.info(f"Feature dataset size: {len(dataset)}, feature_dim: {feature_dim}")

        from src.model import TemporalDetectionHead
        model = TemporalDetectionHead(
            feature_dim=feature_dim,
            num_classes=config["model"]["num_classes"],
        ).to(device)
    else:
        from src.dataset import THUMOSVideoDataset
        from src.model import ActionRecognitionModel
        dataset = THUMOSVideoDataset(
            video_dir=os.path.join(config["paths"]["data_root"], "videos"),
            ann_path=os.path.join(
                config["paths"]["data_root"], "annotations", "thumos_14_anno.json"
            ),
            clip_len_sec=config["data"].get("clip_len_sec", 2.0),
            stride_sec=config["data"].get("stride_sec", 1.0),
            num_frames=config["data"].get("num_frames", 16),
        )
        model = ActionRecognitionModel(
            backbone_name=config["model"]["backbone"],
            num_classes=config["model"]["num_classes"],
            pretrained=config["model"]["pretrained"],
        ).to(device)

    # --- Split ---
    val_size   = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    # --- Optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    criterion = nn.BCEWithLogitsLoss()   # Multi-label for temporal detection

    best_map = 0.0

    for epoch in range(config["training"]["epochs"]):
        # --- Train ---
        model.train()
        running_loss = 0.0
        for step, (feats, labels) in enumerate(train_loader):
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)          # (B, W, num_classes)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            if step % 20 == 0:
                logger.info(f"Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits   = model(feats)
                val_loss += criterion(logits, labels).item()
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save checkpoint every epoch; keep best separately
        ckpt_path = os.path.join(config["paths"]["checkpoint_dir"], f"epoch_{epoch}.pt")
        save_checkpoint(ckpt_path, model, optimizer, epoch, {"val_loss": val_loss})

        if val_loss < best_map:   # Swap for mAP once evaluate_model is wired in
            best_map = val_loss
            save_checkpoint(
                os.path.join(config["paths"]["checkpoint_dir"], "best.pt"),
                model, optimizer, epoch, {"val_loss": val_loss},
            )