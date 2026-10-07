import torch.nn as nn

class ConvBNReLU(nn.Module):
    """

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """

    """

    def __init__(self, num_classes=2, dropout=0.35):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),      # 224 -> 112
            ConvBNReLU(32, 64, stride=2),     # 112 -> 56
            ConvBNReLU(64, 128, stride=2),    # 56 -> 28
            ConvBNReLU(128, 256, stride=2),   # 28 -> 14
            ConvBNReLU(256, 256, stride=1)    # Additional feature extraction
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """

        """

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x