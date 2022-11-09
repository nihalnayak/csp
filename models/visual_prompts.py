import torch

from clip_modules.model_loader import load
from clip_modules.visual_interface import CLIPVisualInterface


def get_visual_prompts(config, device):
    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    interface = CLIPVisualInterface(clip_model, config, device=device)

    optimizer = torch.optim.Adam(
        [interface.custom_visual.visual_prompt],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    return interface, optimizer
