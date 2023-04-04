from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def create_writer(experiment_name,
                  model_name,
                  extra):

    # get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d")

    log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

def save_checkpoint(model,
               optimizer,
               epoch,
               loss,
               target_dir,
               model_name):

    # create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # create model.pth save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # save the checkpoint
    print(f"[INFO] Saving checkpoint to: {model_save_path}")

    torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss,
    }, model_save_path)


def save_model(model,
               experiment_name,
               model_name,
               extra):
    # create path
    timestamp = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join("models", timestamp, experiment_name, model_name, extra)

    # create target directory
    target_dir_path = Path(path)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # create model.pth save path
    model_save_path = target_dir_path / "model.pth"

    # save the model.pth state_dict()
    print(f"[INFO] Saving model.pth to: {path}")
    torch.save(model.state_dict(), model_save_path)
