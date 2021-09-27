import torch
import torch.nn as nn
from torchinfo import summary
from typing import Tuple, List


# MODULAR AUTO ENCODER
class Reshaper(nn.Module):
    def __init__(self, out_shape: int):
        super().__init__()
        self.out_shape = out_shape

    def forward(self, x: torch.Tensor):
        return torch.reshape(x, self.out_shape)


class ConvLeakyRelu(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, kernel_size: int, **kwargs):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_features, out_features, bias=True,
                              padding=padding, kernel_size=kernel_size, **kwargs)
        self.act = nn.LeakyReLU()


class ConvSigmoid(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, kernel_size: int, **kwargs):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_features, out_features, bias=True,
                              padding=padding, kernel_size=kernel_size, **kwargs)
        self.act = nn.Sigmoid()


class ConvLinear(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, kernel_size: int, **kwargs):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_features, out_features, bias=True,
                              padding=padding, kernel_size=kernel_size, **kwargs)


class FlatFullFullLeakyReluReshape(nn.Sequential):
    def __init__(self, image_at_bottleneck: Tuple[int, int], last_layer_f: int, btln_size: int = 128):
        super().__init__()
        self.flat1 = nn.Flatten()
        self.fc1 = nn.Linear(image_at_bottleneck[0] * image_at_bottleneck[1] * last_layer_f, btln_size)
        self.fc2 = nn.Linear(btln_size, image_at_bottleneck[0] * image_at_bottleneck[1] * last_layer_f)
        self.act = nn.LeakyReLU()
        self.resh = Reshaper((-1, last_layer_f, image_at_bottleneck[0], image_at_bottleneck[1]))


class Up(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvLeakyRelu(*args, **kwargs)
            # ConvBNLeakyRelu(*args, **kwargs)
        )


class UpSigmoid(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvSigmoid(*args, **kwargs)
            # ConvBNSigmoid(*args, **kwargs)
        )


class UpLinear(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvLinear(*args, **kwargs)
        )


class Down(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(
            ConvLeakyRelu(*args, kernel_size=3, stride=2, **kwargs)
            # ConvBNLeakyRelu(*args, kernel_size=3, stride=2, **kwargs)
        )


class Encoder(nn.Sequential):
    def __init__(self, widths: List[int], block: nn.Sequential = Down):
        in_out_widths = zip(widths, widths[1:])
        super().__init__(
            *[block(in_f, out_f) for in_f, out_f in in_out_widths]
        )


class Bottleneck(nn.Sequential):
    def __init__(self, *args, block: nn.Sequential = FlatFullFullLeakyReluReshape, **kwargs):
        super().__init__(
            block(*args, **kwargs)
        )


class Decoder(nn.Sequential):
    def __init__(self, widths: List[int], block: nn.Sequential = Up, final_block: nn.Sequential = UpSigmoid):
        in_out_widths = [el for el in zip(widths, widths[1:])]
        in_ff, out_ff = in_out_widths[-1]
        super().__init__(
            *[block(in_f, out_f, kernel_size=3) for in_f, out_f in in_out_widths[:-1]],
            final_block(in_ff, out_ff, kernel_size=3)
        )


class AE(nn.Sequential):
    def __init__(self, widths: List[int], image_shape: Tuple[int, int], bottleneck_size: int = 128,
                 down_block: nn.Sequential = Down, up_block: nn.Sequential = Up,
                 bottleneck_block: nn.Sequential = FlatFullFullLeakyReluReshape,
                 final_block: nn.Sequential = UpLinear):
        super().__init__()
        image_at_bottleneck = (image_shape[0] // (2 ** (len(widths) - 1)), image_shape[1] // (2 ** (len(widths) - 1)))
        self.encoder = Encoder(widths, block=down_block)
        self.bottleneck = Bottleneck(btln_size=bottleneck_size, image_at_bottleneck=image_at_bottleneck,
                                     block=bottleneck_block, last_layer_f=widths[-1])
        self.decoder = Decoder(widths[::-1], block=up_block, final_block=final_block)


if __name__ == "__main__":
    input_channel = 3
    bottleneck = 16
    layer_1_ft = 128
    layer_2_ft = layer_1_ft * 2
    layer_3_ft = layer_1_ft * 2
    layer_4_ft = layer_1_ft * 2 * 2

    widths_ = [
        input_channel,
        layer_1_ft,
        layer_2_ft,
        layer_3_ft,
        layer_4_ft,
    ]

    model = AE(widths_, image_shape=(64, 64), bottleneck_size=bottleneck)

    device_string = "cpu"

    device = torch.device(device_string)
    model.to(device)

    summary(model, input_size=(4, 3, 64, 64), device=device_string, depth=5,
            col_names=["input_size", "output_size", "num_params"])
