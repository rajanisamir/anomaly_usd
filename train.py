import torch
from pathlib import Path
import argparse
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from cnn_autoencoder import CNNAutoencoder

from custom_audio_dataset import UrbanSoundDataset

from torch.utils.tensorboard import SummaryWriter

# BATCH_SIZE = 256
# EPOCHS = 20
# LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

writer = SummaryWriter("logs/1")

def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Optim
    parser.add_argument("--epochs", type=int, default=10,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=256,
                        help='Effective batch size')
    parser.add_argument("--lr", type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument("--wd", type=float, default=1e-5,
                        help='Weight decay')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')

    return parser


def train_one_epoch(model, data_loader, loss_fn, optimizer, device, epoch):
    for i, (inputs, targets) in enumerate(data_loader, start=epoch * len(data_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        recons = model(inputs)
        loss = loss_fn(recons, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss", loss.item(), i)
    state = dict(
        epoch=epoch + 1,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    torch.save(state, args.exp_dir / "model.pth")


def train(model, data_loader, loss_fn, optimizer, device, epochs, start_epoch):
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device, epoch)
        print("---------------")
    print("Training is done.")
    torch.save(model.module.backbone.state_dict(), args.exp_dir / "conv_autoencoder.pth")

def main(args):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    args.exp_dir.mkdir(parents=True, exist_ok=True)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )

    usd = UrbanSoundDataset(
        ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device
    )

    train_data_loader = DataLoader(usd, batch_size=args.batch_size)

    model = CNNAutoencoder()
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model = model.to(device=device)
    # cnn_autoencoder.load_state_dict(torch.load('cnn_autoencoder.pth'))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )

    if (args.exp_dir / "model.pth").is_file():
        print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    train(model, train_data_loader, loss_fn, optimizer, device, args.epochs, start_epoch)

    # torch.save(model.state_dict(), "cnn_autoencoder.pth")
    # print("Model trained and stored at cnn_autoencoder.pth")


if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    main(args)

    