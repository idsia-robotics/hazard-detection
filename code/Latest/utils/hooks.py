from typing import Callable, Dict

import torch
from torch import nn, Tensor


class BottleneckEmbeddingExtractor(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.embedding = torch.empty(0)
        self.model.bottleneck[0].fc1.register_forward_hook(self.save_outputs_hook())

    def save_outputs_hook(self) -> Callable:
        def fn(_, __, output):
            self.embedding = output.detach().cpu()
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self.embedding
