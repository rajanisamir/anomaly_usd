import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from cnn_autoencoder import CNNAutoencoder
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

from custom_audio_dataset import UrbanSoundDataset

PRETRAINED_PATH = "cnn_autoencoder.pth"

ANNOTATIONS_FILE = "UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def eval(model, dataloader, device):
    plt.figure(figsize=(10, 2))
    plt.gray()
    for i, (img, target) in enumerate(dataloader):
        if i >= 10:
            break
        print(f"Saving Sample {i}/{len(dataloader)}")
        img, target = img.to(device), target.to(device)
        recon = model(img)
        plt.subplot(2, 10, i + 1)
        plt.imshow(img[0][0], cmap="plasma")
        plt.subplot(2, 10, i + 11)
        plt.imshow(recon[0][0], cmap="plasma")
        time_series_img = librosa.feature.inverse.mel_to_audio(
            img[0][0].detach().numpy(), sr=SAMPLE_RATE, n_fft=1024, hop_length=512
        )
        time_series_recon = librosa.feature.inverse.mel_to_audio(
            recon[0][0].detach().numpy(), sr=SAMPLE_RATE, n_fft=1024, hop_length=512
        )
        if i == 7:
            sf.write("audio_img.wav", time_series_img, SAMPLE_RATE)
            sf.write("audio_recon.wav", time_series_recon, SAMPLE_RATE)
    plt.savefig("recon")


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )

    usd = UrbanSoundDataset(
        ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device
    )

    cnn_autoencoder = CNNAutoencoder().to(device=device)
    cnn_autoencoder.load_state_dict(torch.load(PRETRAINED_PATH))

    test_data_loader = DataLoader(usd, batch_size=1)

    with torch.no_grad():
        eval(cnn_autoencoder, test_data_loader, device)
