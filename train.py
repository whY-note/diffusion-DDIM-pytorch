from dataset import create_dataset
from model.UNet import UNet
from utils.engine import GaussianDiffusionTrainer
from utils.tools import train_one_epoch, load_yaml
import torch
from utils.callbacks import ModelCheckpoint

# to record loss
import csv
import os

def train(config):
    consume = config["consume"]
    training_epochs = config["epochs"]

    dataset_name = config["Dataset"]["dataset"]
    loss_filename = dataset_name + "_loss.csv"

    log_dir = config.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    loss_csv_path = os.path.join(log_dir, loss_filename)

    # 如果是从头训练，写表头
    if not consume:
        with open(loss_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])

    if consume:
        cp = torch.load(config["consume_path"])
        config = cp["config"]  # 用旧的config
    print(config)

    device = torch.device(config["device"])
    loader = create_dataset(**config["Dataset"])
    start_epoch = 1
    end_epoch = training_epochs + 1

    model = UNet(**config["Model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)

    model_checkpoint = ModelCheckpoint(**config["Callback"])

    if consume:
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        model_checkpoint.load_state_dict(cp["model_checkpoint"])
        start_epoch = cp["start_epoch"] + 1
        end_epoch = training_epochs + start_epoch

    print(f"start_epoch: {start_epoch}")
    print(f"end_epoch:{end_epoch}")

    for epoch in range(start_epoch, end_epoch):
        loss = train_one_epoch(trainer, loader, optimizer, device, epoch)
        
        # record loss
        with open(loss_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss])

        model_checkpoint.step(loss, model=model.state_dict(), config=config,
                              optimizer=optimizer.state_dict(), start_epoch=epoch,
                              model_checkpoint=model_checkpoint.state_dict())


if __name__ == "__main__":
    config = load_yaml("config.yml", encoding="utf-8")
    train(config)
