"""LR-ASD inference architecture.

Adapted from Junhua-Liao/LR-ASD at revision
1b6dcd2d8fc2895683de6508ec6294ec47d388ca under the MIT License.
Copyright (c) 2025 Liao Junhua.
"""

from __future__ import annotations

import torch
from torch import nn


class AudioBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_1, kernel_2):
        super().__init__()
        self.relu = nn.ReLU()
        padding_1 = (kernel_1 - 1) // 2
        padding_2 = (kernel_2 - 1) // 2
        self.m_1 = nn.Conv2d(
            in_channels, out_channels // 2, kernel_size=(kernel_1, 1), padding=(padding_1, 0), bias=False
        )
        self.m_norm_1 = nn.BatchNorm2d(out_channels // 2, momentum=0.01, eps=0.001)
        self.m_2 = nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=(kernel_2, 1), padding=(padding_2, 0), bias=False
        )
        self.m_norm_2 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_1 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, kernel_1), padding=(0, padding_1), bias=False
        )
        self.t_norm_1 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, kernel_2), padding=(0, padding_2), bias=False
        )
        self.t_norm_2 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, value):
        value = self.relu(self.m_norm_1(self.m_1(value)))
        value = self.relu(self.m_norm_2(self.m_2(value)))
        value = self.relu(self.t_norm_1(self.t_1(value)))
        return self.relu(self.t_norm_2(self.t_2(value)))


class VisualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_1, kernel_2, is_down=False):
        super().__init__()
        self.relu = nn.ReLU()
        padding_1 = (kernel_1 - 1) // 2
        padding_2 = (kernel_2 - 1) // 2
        stride = (1, 2, 2) if is_down else (1, 1, 1)
        self.s_1 = nn.Conv3d(
            in_channels,
            out_channels // 2,
            kernel_size=(1, kernel_1, kernel_1),
            stride=stride,
            padding=(0, padding_1, padding_1),
            bias=False,
        )
        self.s_norm_1 = nn.BatchNorm3d(out_channels // 2, momentum=0.01, eps=0.001)
        self.s_2 = nn.Conv3d(
            out_channels // 2,
            out_channels,
            kernel_size=(1, kernel_2, kernel_2),
            padding=(0, padding_2, padding_2),
            bias=False,
        )
        self.s_norm_2 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        self.t_1 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(kernel_1, 1, 1),
            padding=(padding_1, 0, 0),
            bias=False,
        )
        self.t_norm_1 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        self.t_2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(kernel_2, 1, 1),
            padding=(padding_2, 0, 0),
            bias=False,
        )
        self.t_norm_2 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, value):
        value = self.relu(self.s_norm_1(self.s_1(value)))
        value = self.relu(self.s_norm_2(self.s_2(value)))
        value = self.relu(self.t_norm_1(self.t_1(value)))
        return self.relu(self.t_norm_2(self.t_2(value)))


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = VisualBlock(1, 32, 5, 3, is_down=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block2 = VisualBlock(32, 64, 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block3 = VisualBlock(64, 128, 5, 3)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, value):
        value = self.pool1(self.block1(value))
        value = self.pool2(self.block2(value))
        value = self.block3(value).transpose(1, 2)
        batch, frames, channels, width, height = value.shape
        value = self.maxpool(value.reshape(batch * frames, channels, width, height))
        return value.view(batch, frames, channels)


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = AudioBlock(1, 32, 5, 3)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.block2 = AudioBlock(32, 64, 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.block3 = AudioBlock(64, 128, 5, 3)

    def forward(self, value):
        value = self.pool1(self.block1(value))
        value = self.pool2(self.block2(value))
        value = self.block3(value)
        return torch.mean(value, dim=2, keepdim=True).squeeze(2).transpose(1, 2)


class Fusion(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.attention = nn.Conv1d(channel, channel, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(channel, momentum=0.01, eps=0.001)

    def forward(self, first, second):
        identity = torch.cat((first, second), 2).transpose(1, 2)
        weights = self.sigmoid(self.bn(self.attention(identity)))
        return (identity * weights).transpose(1, 2)


class Detector(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.gru_forward = nn.GRU(
            input_size=channel, hidden_size=channel // 4, num_layers=1, bidirectional=False, batch_first=True
        )
        self.gru_backward = nn.GRU(
            input_size=channel, hidden_size=channel // 4, num_layers=1, bidirectional=False, batch_first=True
        )
        self.drop = nn.Dropout(0.5)
        self.attention = Fusion(channel // 2)

    def forward(self, value):
        forward, _ = self.gru_forward(self.drop(value))
        backward, _ = self.gru_backward(self.drop(torch.flip(value, dims=[1])))
        return self.attention(forward, torch.flip(backward, dims=[1]))


class ASDModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Attribute names intentionally match the official checkpoint.
        self.visualEncoder = VisualEncoder()
        self.audioEncoder = AudioEncoder()
        self.fusion = Fusion(256)
        self.detector = Detector(256)

    def forward_visual_frontend(self, value):
        batch, frames, width, height = value.shape
        value = value.view(batch, 1, frames, width, height)
        return self.visualEncoder((value / 255 - 0.4161) / 0.1688)

    def forward_audio_frontend(self, value):
        return self.audioEncoder(value.unsqueeze(1).transpose(2, 3))

    def forward_audio_visual_backend(self, audio, visual):
        value = self.detector(self.fusion(audio, visual))
        return torch.reshape(value, (-1, 128))


class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.FC = nn.Linear(128, 2)


class LRASDInference(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ASDModel()
        self.lossAV = ClassificationHead()
        # Included so the official state dictionary loads strictly.
        self.lossV = ClassificationHead()

    def logits(self, audio_features, visual_features):
        audio = self.model.forward_audio_frontend(audio_features)
        visual = self.model.forward_visual_frontend(visual_features)
        fused = self.model.forward_audio_visual_backend(audio, visual)
        return self.lossAV.FC(fused)
