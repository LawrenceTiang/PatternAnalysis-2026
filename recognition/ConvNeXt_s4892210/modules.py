import torch
import torch.nn as nn
import torchvision

class LayerNorm2d(nn.Module):
    """

    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)   # (N, C, H, W) -> (N, H, W, C)
        x = self.layer_norm(x)      # Apply LayerNorm over C
        x = x.permute(0, 3, 1, 2)   # (N, H, W, C) -> (N, C, H, W)

        return x


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


class ConvNeXtBlock(nn.Module):
    """

    """
    def __init__(self, dim, drop_path, layer_scale_initial_value):
        super().__init__()

        self.dwconv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            padding=3,
            groups=dim
        )
        self.norm = LayerNorm2d(num_channels=dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(
            in_channels=dim,
            out_channels=4 * dim,
            kernel_size=1
        )
        self.activation = nn.GELU()
        self.pwconv2 = nn.Conv2d(
            in_channels=4 * dim,
            out_channels=dim,
            kernel_size=1
        )

        if layer_scale_initial_value > 0:
            self.gamma = nn.Parameter(
                data=layer_scale_initial_value * torch.ones(
                    (dim),
                    requires_grad=True
                )
            )
        else:
            self.gamma = None

        if drop_path > 0:
            self.stochastic_depth = torchvision.ops.StochasticDepth(
                p=drop_path,
                mode="batch"
            )
        else:
            self.stochastic_depth = nn.Identity()

    def forward(self, x):
        """

        """
        residual = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.activation(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma[:, None, None] * x

        x = residual + self.stochastic_depth(x)
        return x

class ConvNextStage(nn.Module):
    """

    """
    def __init__(self, depth, dim, drop_path_rates, layer_scale_initial_value):
        super().__init__()
        blocks = []

        for i in range(depth):
            block = ConvNeXtBlock(
                dim=dim,
                drop_path=drop_path_rates[i],
                layer_scale_initial_value=layer_scale_initial_value
            )
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        """

        """
        return self.blocks(x)


class ConvNeXtClassifier(nn.Module):
    """

    """
    def __init__(
        self,
        input_channels=3,
        num_classes=2,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.1,
        layer_scale_initial_value=1e-6
    ):
        super().__init__()

        self.num_stages = len(depths)
        self.downsampling_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=dims[0],
                kernel_size=4,
                stride=4,
            ),
            LayerNorm2d(num_channels=dims[0], eps=1e-6)
        )

        self.downsampling_layers.append(stem)

        for i in range(len(dims) - 1):
            downsample = nn.Sequential(
                LayerNorm2d(num_channels=dims[i], eps=1e-6),
                nn.Conv2d(
                    in_channels=dims[i],
                    out_channels=dims[i + 1],
                    kernel_size=2,
                    stride=2
                )
            )

            self.downsampling_layers.append(downsample)
        
        drop_path_rates = [
            tensor.item() for tensor in torch.linspace(
                start=0.0,
                end=drop_path_rate,
                steps=sum(depths)
            )
        ]

        self.stages = nn.ModuleList()
        rate_index = 0

        for i in range(self.num_stages):
            stage_rates = drop_path_rates[rate_index: rate_index + depths[i]]

            self.stages.append(
                ConvNextStage(
                    depth=depths[i],
                    dim=dims[i],
                    drop_path_rates=stage_rates,
                    layer_scale_initial_value=layer_scale_initial_value
                )
            )

            rate_index += depths[i]
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=dims[-1], out_features=num_classes)
        )

    def forward(self, x):
        """

        """
        for i in range(self.num_stages):
            x = self.downsampling_layers[i](x)
            x = self.stages[i](x)

        x = self.avgpool(x)
        x = self.classifier(x)
        return x