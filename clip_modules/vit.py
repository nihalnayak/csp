import math
from functools import reduce
from operator import mul

import torch
import torch.nn as nn


class CustomVisualEncoder(nn.Module):
    def __init__(self, clip_visual, num_prompt_tokens) -> None:
        super().__init__()

        self.input_resolution = clip_visual.input_resolution
        self.conv1 = clip_visual.conv1

        self.class_embedding = clip_visual.class_embedding
        self.positional_embedding = clip_visual.positional_embedding
        self.ln_pre = clip_visual.ln_pre

        self.transformer = clip_visual.transformer

        self.ln_post = clip_visual.ln_post
        self.proj = clip_visual.proj

        self.num_prompt_tokens = num_prompt_tokens
        width = self.positional_embedding.size(1)
        self.visual_prompt = nn.Parameter(torch.zeros(1, self.num_prompt_tokens, width))

        patch_size = self.conv1.weight.shape[-1]
        val = math.sqrt(6.0 / float(3 * reduce(mul, [patch_size], 1) + width))  # noqa
        # xavier_uniform initialization
        print(-val, val)
        nn.init.uniform_(self.visual_prompt.data, -val, val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # shallow prompt
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = torch.cat(
            [
                x[:1, :, :],
                self.visual_prompt.expand(batch_size, -1, -1)
                .type(x.dtype)
                .permute(1, 0, 2)
                .to(x.device),
                x[1:, :, :],
            ]
        )

        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
