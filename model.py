import os
import time
import torch
from torch import Tensor
from torch import nn
from torch import optim
import torch.nn.functional as F
from typing import List, Tuple

from utils.display import *


class ResBlock(nn.Module):
    r"""ResNet block based on "Deep Residual Learning for Image Recognition"

    The paper link is https://arxiv.org/pdf/1512.03385.pdf.

    Args:
        n_freq: the number of bins in a spectrogram. (Default: ``128``)

    Examples
        >>> resblock = ResBlock()
        >>> input = torch.rand(10, 128, 512)  # a random spectrogram
        >>> output = resblock(input)  # shape: (10, 128, 512)
    """

    def __init__(self, n_freq: int = 128) -> None:
        super().__init__()

        self.resblock_model = nn.Sequential(
            nn.Conv1d(in_channels=n_freq, out_channels=n_freq, kernel_size=1, bias=False),
            nn.BatchNorm1d(n_freq),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=n_freq, out_channels=n_freq, kernel_size=1, bias=False),
            nn.BatchNorm1d(n_freq)
        )

    def forward(self, specgram: Tensor) -> Tensor:
        r"""Pass the input through the ResBlock layer.
        Args:
            specgram (Tensor): the input sequence to the ResBlock layer (n_batch, n_freq, n_time).

        Return:
            Tensor shape: (n_batch, n_freq, n_time)
        """

        return self.resblock_model(specgram) + specgram


class MelResNet(nn.Module):
    r"""MelResNet layer uses a stack of ResBlocks on spectrogram.

    Args:
        n_res_block: the number of ResBlock in stack. (Default: ``10``)
        n_freq: the number of bins in a spectrogram. (Default: ``128``)
        n_hidden: the number of hidden dimensions of resblock. (Default: ``128``)
        n_output: the number of output dimensions of melresnet. (Default: ``128``)
        kernel_size: the number of kernel size in the first Conv1d layer. (Default: ``5``)

    Examples
        >>> melresnet = MelResNet()
        >>> input = torch.rand(10, 128, 512)  # a random spectrogram
        >>> output = melresnet(input)  # shape: (10, 128, 508)
    """

    def __init__(self,
                 n_res_block: int = 10,
                 n_freq: int = 128,
                 n_hidden: int = 128,
                 n_output: int = 128,
                 kernel_size: int = 5) -> None:
        super().__init__()

        ResBlocks = [ResBlock(n_hidden) for _ in range(n_res_block)]

        self.melresnet_model = nn.Sequential(
            nn.Conv1d(in_channels=n_freq, out_channels=n_hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            *ResBlocks,
            nn.Conv1d(in_channels=n_hidden, out_channels=n_output, kernel_size=1)
        )

    def forward(self, specgram: Tensor) -> Tensor:
        r"""Pass the input through the MelResNet layer.
        Args:
            specgram (Tensor): the input sequence to the MelResNet layer (n_batch, n_freq, n_time).

        Return:
            Tensor shape: (n_batch, n_output, n_time - kernel_size + 1)
        """

        return self.melresnet_model(specgram)


class Stretch2d(nn.Module):
    r"""Upscale the frequency and time dimensions of a spectrogram.

    Args:
        time_scale: the scale factor in time dimension
        freq_scale: the scale factor in frequency dimension

    Examples
        >>> stretch2d = Stretch2d(time_scale=10, freq_scale=5)

        >>> input = torch.rand(10, 100, 512)  # a random spectrogram
        >>> output = stretch2d(input)  # shape: (10, 500, 5120)
    """

    def __init__(self,
                 time_scale: int,
                 freq_scale: int) -> None:
        super().__init__()

        self.freq_scale = freq_scale
        self.time_scale = time_scale

    def forward(self, specgram: Tensor) -> Tensor:
        r"""Pass the input through the Stretch2d layer.

        Args:
            specgram (Tensor): the input sequence to the Stretch2d layer (..., n_freq, n_time).

        Return:
            Tensor shape: (..., n_freq * freq_scale, n_time * time_scale)
        """

        return specgram.repeat_interleave(self.freq_scale, -2).repeat_interleave(self.time_scale, -1)


class UpsampleNetwork(nn.Module):
    r"""Upscale the dimensions of a spectrogram.

    Args:
        upsample_scales: the list of upsample scales.
        n_res_block: the number of ResBlock in stack. (Default: ``10``)
        n_freq: the number of bins in a spectrogram. (Default: ``128``)
        n_hidden: the number of hidden dimensions of resblock. (Default: ``128``)
        n_output: the number of output dimensions of melresnet. (Default: ``128``)
        kernel_size: the number of kernel size in the first Conv1d layer. (Default: ``5``)

    Examples
        >>> upsamplenetwork = UpsampleNetwork(upsample_scales=[4, 4, 16])
        >>> input = torch.rand(10, 128, 10)  # a random spectrogram
        >>> output = upsamplenetwork(input)  # shape: (10, 1536, 128), (10, 1536, 128)
    """

    def __init__(self,
                 upsample_scales: List[int],
                 n_res_block: int = 10,
                 n_freq: int = 128,
                 n_hidden: int = 128,
                 n_output: int = 128,
                 kernel_size: int = 5) -> None:
        super().__init__()

        total_scale = 1
        for upsample_scale in upsample_scales:
            total_scale *= upsample_scale

        self.indent = (kernel_size - 1) // 2 * total_scale
        self.resnet = MelResNet(n_res_block, n_freq, n_hidden, n_output, kernel_size)
        self.resnet_stretch = Stretch2d(total_scale, 1)

        up_layers = []
        for scale in upsample_scales:
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(in_channels=1,
                             out_channels=1,
                             kernel_size=(1, scale * 2 + 1),
                             padding=(0, scale),
                             bias=False)
            conv.weight.data.fill_(1. / (scale * 2 + 1))
            up_layers.append(stretch)
            up_layers.append(conv)
        self.upsample_layers = nn.Sequential(*up_layers)

    def forward(self, specgram: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the UpsampleNetwork layer.

        Args:
            specgram (Tensor): the input sequence to the UpsampleNetwork layer (n_batch, n_freq, n_time)

        Return:
            Tensor shape: (n_batch, n_freq, (n_time - kernel_size + 1) * total_scale),
                          (n_batch, n_output, (n_time - kernel_size + 1) * total_scale)
        where total_scale is the product of all elements in upsample_scales.
        """

        resnet_output = self.resnet(specgram).unsqueeze(1)
        resnet_output = self.resnet_stretch(resnet_output)
        resnet_output = resnet_output.squeeze(1)

        specgram = specgram.unsqueeze(1)
        upsampling_output = self.upsample_layers(specgram)
        upsampling_output = upsampling_output.squeeze(1)[:, :, self.indent:-self.indent]

        return upsampling_output, resnet_output


class WaveRNN(nn.Module):
    r"""WaveRNN model based on the implementation from `fatchord <https://github.com/fatchord/WaveRNN>`_.

    The original implementation was introduced in
    `"Efficient Neural Audio Synthesis" <https://arxiv.org/pdf/1802.08435.pdf>`_.
    The input channels of waveform and spectrogram have to be 1. The product of
    `upsample_scales` must equal `hop_length`.

    Args:
        upsample_scales: the list of upsample scales.
        n_classes: the number of output classes.
        hop_length: the number of samples between the starts of consecutive frames.
        n_res_block: the number of ResBlock in stack. (Default: ``10``)
        n_rnn: the dimension of RNN layer. (Default: ``512``)
        n_fc: the dimension of fully connected layer. (Default: ``512``)
        kernel_size: the number of kernel size in the first Conv1d layer. (Default: ``5``)
        n_freq: the number of bins in a spectrogram. (Default: ``128``)
        n_hidden: the number of hidden dimensions of resblock. (Default: ``128``)
        n_output: the number of output dimensions of melresnet. (Default: ``128``)

    Example
        >>> wavernn = WaveRNN(upsample_scales=[5,5,8], n_classes=512, hop_length=200)
        >>> waveform, sample_rate = torchaudio.load(file)
        >>> # waveform shape: (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length)
        >>> specgram = MelSpectrogram(sample_rate)(waveform)  # shape: (n_batch, n_channel, n_freq, n_time)
        >>> output = wavernn(waveform, specgram)
        >>> # output shape: (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length, n_classes)
    """

    def __init__(self,
                 upsample_scales: List[int],
                 n_classes: int,
                 hop_length: int,
                 n_res_block: int = 10,
                 n_rnn: int = 512,
                 n_fc: int = 512,
                 kernel_size: int = 5,
                 n_freq: int = 128,
                 n_hidden: int = 128,
                 n_output: int = 128,
                 pad: int = 2) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.n_rnn = n_rnn
        self.n_aux = n_output // 4
        self.hop_length = hop_length
        self.n_classes = n_classes
        self.pad = pad

        total_scale = 1
        for upsample_scale in upsample_scales:
            total_scale *= upsample_scale
        if total_scale != self.hop_length:
            raise ValueError(f"Expected: total_scale == hop_length, but found {total_scale} != {hop_length}")

        self.upsample = UpsampleNetwork(upsample_scales,
                                        n_res_block,
                                        n_freq,
                                        n_hidden,
                                        n_output,
                                        kernel_size)
        self.fc = nn.Linear(n_freq + self.n_aux + 1, n_rnn)

        self.rnn1 = nn.GRU(n_rnn, n_rnn, batch_first=True)
        self.rnn2 = nn.GRU(n_rnn + self.n_aux, n_rnn, batch_first=True)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(n_rnn + self.n_aux, n_fc)
        self.fc2 = nn.Linear(n_fc + self.n_aux, n_fc)
        self.fc3 = nn.Linear(n_fc, self.n_classes)

    def forward(self, waveform: Tensor, specgram: Tensor) -> Tensor:
        r"""Pass the input through the WaveRNN model.

        Args:
            waveform: the input waveform to the WaveRNN layer (n_batch, (n_time - kernel_size + 1) * hop_length)
            specgram: the input spectrogram to the WaveRNN layer (n_batch, n_freq, n_time)

        Return:
            Tensor shape: (n_batch, (n_time - kernel_size + 1) * hop_length, n_classes)
        """

        batch_size = waveform.size(0)
        h1 = torch.zeros(1, batch_size, self.n_rnn, dtype=waveform.dtype, device=waveform.device)
        h2 = torch.zeros(1, batch_size, self.n_rnn, dtype=waveform.dtype, device=waveform.device)
        # output of upsample:
        # specgram: (n_batch, n_freq, (n_time - kernel_size + 1) * total_scale)
        # aux: (n_batch, n_output, (n_time - kernel_size + 1) * total_scale)
        specgram, aux = self.upsample(specgram)
        specgram = specgram.transpose(1, 2)
        aux = aux.transpose(1, 2)

        aux_idx = [self.n_aux * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

        x = torch.cat([waveform.unsqueeze(-1), specgram, a1], dim=-1)
        x = self.fc(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=-1)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=-1)
        x = self.fc1(x)
        x = self.relu1(x)

        x = torch.cat([x, a4], dim=-1)
        x = self.fc2(x)
        x = self.relu2(x)
        logits = self.fc3(x)

        return logits

    def inference(self, specgram: Tensor) -> Tensor:
        r"""Pass the input through the WaveRNN model.

        Args:
            specgram: the input spectrogram to the WaveRNN layer (n_batch, n_freq, n_time)

        Return:
            Tensor shape: (n_batch, (n_time - kernel_size + 1) * hop_length)
        """
        
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)
        specgram = self.pad_tensor(specgram, pad=self.pad)

        batch_size = specgram.size(0)
        h1 = torch.zeros(batch_size, self.n_rnn, dtype=torch.float32, device=specgram.device)
        h2 = torch.zeros(batch_size, self.n_rnn, dtype=torch.float32, device=specgram.device)
        # output of upsample:
        # specgram: (n_batch, n_freq, (n_time - kernel_size + 1) * total_scale)
        # aux: (n_batch, n_output, (n_time - kernel_size + 1) * total_scale)
        specgram, aux = self.upsample(specgram)
        specgram = specgram.transpose(1, 2)
        aux = aux.transpose(1, 2)

        aux_idx = [self.n_aux * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

        seq_len = specgram.size(1)
        x = torch.zeros(batch_size, 1, device=specgram.device)
        output = []
        start = time.time()

        for i in range(seq_len):
            
            m_t = specgram[:,i,:]
            a1_t, a2_t, a3_t, a4_t = a1[:,i,:], a2[:,i,:], a3[:,i,:], a4[:,i,:]

            x = torch.cat([x, m_t, a1_t], dim=-1)
            x = self.fc(x)
            res = x
            h1 = rnn1(x, h1)
            x = h1

            x = x + res
            res = x
            x = torch.cat([x, a2_t], dim=-1)
            h2 = rnn2(x, h2)
            x = h2

            x = x + res
            x = torch.cat([x, a3_t], dim=-1)
            x = self.fc1(x)
            x = self.relu1(x)

            x = torch.cat([x, a4_t], dim=-1)
            x = self.fc2(x)
            x = self.relu2(x)
            logits = self.fc3(x)

            probs = F.softmax(logits, dim=-1)
            distrib = torch.distributions.Categorical(probs)
            sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
            output.append(sample)

            x = sample.unsqueeze(-1)

            if i % 100 == 0:
                self.gen_display(i, seq_len, start)

        output = torch.stack(output).transpose(0, 1)
        print()

        return output

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def pad_tensor(self, x, pad):
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/dims
        # x: (n_batch, n_freq, n_time)
        b, c, t = x.size()
        total = t + 2 * pad 
        padded = torch.zeros(b, c, total, device=x.device)
        padded[:, :, pad:pad + t] = x

        return padded

    def gen_display(self, i, seq_len, start):
        gen_rate = (i + 1) / (time.time() - start) / 1000
        pbar = progbar(i, seq_len)
        msg = f'| {pbar} {i}/{seq_len} | Gen Rate: {gen_rate:.1f}kHz | '
        stream(msg)


def get_model(hp, device, train=False, args=None):
    model = WaveRNN(upsample_scales=hp.upsample_scales,
                 n_classes=2**hp.bits,
                 hop_length=hp.hop_length,
                 n_res_block=hp.n_res_block,
                 n_rnn=hp.rnn_dim,
                 n_fc=hp.fc_dim,
                 kernel_size=2*hp.pad_len+1,
                 n_freq=hp.num_mels,
                 n_hidden=hp.hidden_dim,
                 n_output=hp.res_out_dim,
                 pad = hp.pad_len).to(device)

    restore_step = hp.restore_step
    if args is not None:
        restore_step = args.restore_step
    if restore_step:
        ckpt_path = os.path.join(
            hp.ckpt_dir,
            "{}.pth.tar".format(restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        optimizer = optim.Adam(model.parameters())
        for g in optimizer.param_groups: g['lr'] = hp.lr
        if restore_step:
            optimizer.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, optimizer

    model.eval()
    model.requires_grad_ = False
    return model
