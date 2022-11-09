import argparse

import torch
from clip.model import CLIP

from .vit import CustomVisualEncoder


class CLIPVisualInterface(torch.nn.Module):
    def __init__(
        self,
        clip_model: CLIP,
        config: argparse.ArgumentParser,
        dtype: torch.dtype = None,
        device: torch.device = "cuda:0",
    ):
        """CLIP interface for our custom modules.

        Args:
            clip_model (CLIP): the clip model
            config (argparse.ArgumentParser): arguments used for
                training
            token_ids (torch.tensor): the input token ids to the text
                encoder
            soft_embeddings (torch.nn.Parameter, optional): the only
                parameter that we finetune in the experiment.
                Defaults to None.
            dtype (torch.dtype, optional): torch dtype for the
                transformer. This allows the half precision option.
                Defaults to None.
            device (torch.device, optional): the device where the model
                should be loaded. Defaults to "cuda:0".
            enable_pos_emb (bool, optional): if true, adds the learned
                positional embeddings. Defaults to False.
        """
        super().__init__()

        self.config = config

        self.clip_model = clip_model

        if dtype is None and device == "cpu":
            self.dtype = torch.float32
        elif dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype

        self.device = device

        self.custom_visual = CustomVisualEncoder(
            clip_model.visual, config.num_prompt_tokens
        )

    def encode_image(self, image):
        return self.custom_visual(image.type(self.dtype))

    def encode_text(self, text):
        return self.clip_model.encode_text(text)

    def tokenize(self, text):
        return self.clip_model.encode_text(text)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = self.clip_model.logit_scale.exp() * image_features @ text_features.t()

        return logits
