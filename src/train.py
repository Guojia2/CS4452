import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import os


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- Model ---
    from src.model import ActionRecognitionModel
    model = ActionRecognitionModel(
        backbone_name=config["model"]["backbone"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    ).to(device)

    # --- Data ---
    from src.dataset import THUMOSDataset
    train_dataset = THUMOSDataset(
        root=config["paths"]["data_root"],
        split="train",
        clip_len=config["data"]["clip_len"],
        frame_interval=config["data"]["frame_interval"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    # --- Optimizer & Scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    criterion = nn.CrossEntropyLoss()

    # --- Loop ---
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        model.train()
        running_loss = 0.0

        for step, (clips, labels) in enumerate(train_loader):
            clips, labels = clips.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 20 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        scheduler.step()

        # Save checkpoint to the Modal Volume
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": running_loss / len(train_loader),
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")