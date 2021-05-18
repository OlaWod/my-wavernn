import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import label_2_float
from hparams import hp


class VocoderDataset(Dataset):
    def __init__(self, basename):
        self.data_dir = hp.data_dir
        self.mel_dir = os.path.join(hp.data_dir, 'mel')
        self.quant_dir = os.path.join(hp.data_dir, 'quant')
        
        self.basename = basename

    def __getitem__(self, idx):
        basename = self.basename[idx]
        
        # mel
        mel_path = os.path.join(self.mel_dir, f'{basename}.npy')
        mel = torch.from_numpy(np.load(mel_path))
        # quant
        quant_path = os.path.join(self.quant_dir, f'{basename}.npy')
        quant = torch.from_numpy(np.load(quant_path))

        return mel, quant

    def __len__(self):
        return len(self.basename)


def collate_vocoder(batch):
    mel_win = hp.seq_len // hp.hop_length + 2 * hp.pad_len
    max_offsets = [x[0].size(-1) -2 - (mel_win + 2 * hp.pad_len) for x in batch]

    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.pad_len) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] \
            for i, x in enumerate(batch)]
    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp.seq_len + 1] \
              for i, x in enumerate(batch)]

    mels = torch.stack(mels).float()
    labels = torch.stack(labels).long()

    x = labels[:, :-1]
    y = labels[:, 1:]

    x = label_2_float(x.float(), hp.bits)

    return x, y, mels


def get_loader():

    with open(os.path.join(hp.data_dir, 'basenames.pkl'), 'rb') as f:
        basenames = pickle.load(f)

    random.shuffle(basenames)

    train_basenames = basenames[:-hp.val_size]
    test_basenames = basenames[-hp.val_size:]

    train_dataset = VocoderDataset(train_basenames)
    test_dataset = VocoderDataset(test_basenames)

    train_loader = DataLoader(train_dataset,
                           collate_fn=collate_vocoder,
                           batch_size=hp.batch_size,
                           num_workers=2,
                           shuffle=True)

    test_loader = DataLoader(test_dataset,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = get_loader()
    
    for batch in train_loader:
        print(batch)
