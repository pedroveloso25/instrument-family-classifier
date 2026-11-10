"""A small CNN over log-mel spectrograms (Phase 3)."""
import torch
from torch import nn


class LogMelCNN(nn.Module):
    """Conv blocks (Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2), global average pooling, dropout, Linear.

    Input: (batch, 1, n_mels, frames), already normalised per mel band. Output: (batch, n_classes) logits.
    With the default 4 blocks of 16/32/64/128 channels it has ~100k parameters, small on purpose:
    there are only 36 training instruments, so capacity is not what limits this model.
    """

    def __init__(self, n_classes=10, channels=(16, 32, 64, 128), dropout=0.3):
        super().__init__()
        blocks, c_in = [], 1
        for c_out in channels:
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),  # bias is redundant before BatchNorm
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c_in, n_classes)

    def embed(self, x):
        """One vector per note: the last conv block's channels averaged over frequency and time."""
        return self.features(x).mean(dim=(2, 3))

    def forward(self, x):
        return self.head(self.dropout(self.embed(x)))


def n_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = LogMelCNN()
    x = torch.zeros(2, 1, 128, 251)
    print(model)
    print(f"output shape {tuple(model(x).shape)}, {n_parameters(model):,} parameters")
