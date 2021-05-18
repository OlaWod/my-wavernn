import os
import argparse
import numpy as np
import torch
from scipy.io import wavfile

from hparams import hp
from model import get_model
from utils.spec import melspectrogram
from utils.tools import load_wav, label_2_float


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_testset(model, test_loader, hp, device, step):

    for i, (mel, x) in enumerate(test_loader):
        
        x, mel = x.to(device), mel.to(device)
        wav = model.inference(mel)[0].cpu().numpy()
        wav_truth = label_2_float(x[0], hp.bits).cpu().numpy()
        
        wavfile.write(
            os.path.join(hp.log_dir,f"{step}-{i}.wav"),
            hp.sample_rate,
            wav)

        wavfile.write(
            os.path.join(hp.log_dir,f"{step}-{i}-truth.wav"),
            hp.sample_rate,
            wav_truth)


def main(mel, args):
    mel = mel.to(device)
    model = get_model(hp, device, train=False, args=args)
    wav = model.inference(mel)[0].cpu().numpy()
        
    wavfile.write(
        os.path.join(hp.result_dir,"{}.wav".format(args.name)),
        hp.sample_rate,
        wav)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--restore_step", type=int,
                        required=True)
    parser.add_argument('-n', "--name", type=str,
                        required=True, help="wav name for saving")
    parser.add_argument('-m', "--mel_path", type=str,
                        help="path to mel.npy",)
    parser.add_argument('-w', "--wav_path", type=str,
                        help="path to wav")
    args = parser.parse_args()

    if args.mel_path is not None:
        mel = torch.from_numpy(np.load(args.mel_path)).unsqueeze(0)
    elif args.wav_path is not None:
        y = load_wav(args.wav_path)
        mel = melspectrogram(y)
        mel = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)

    if mel is not None:
        os.makedirs(hp.result_dir, exist_ok=True)
        main(mel, args)
