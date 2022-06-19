## Remove this later
import argparse
import os
from turtle import forward

import clip
from numpy import dtype
import torch
import torch.nn as nn
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class Adapter(nn.Module):
    def __init__(self, embed_dim, alpha, dtype=torch.float16):
        super().__init__()
        self.visual_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, dtype=dtype),
        )
        self.text_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.alpha = alpha

    def forward(self, representation, text=False):
        if text:
            return self.text_adapter(representation)
        else:
            return self.visual_adapter(representation)

def get_clip_adapters(
        train_dataset,
        config,
        device,
        prompt_template="a photo of x x"
    ):

    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs

    # cleaning the classes and the attributes
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]

    tokenized = torch.cat(
        [
            clip.tokenize(tok, context_length=config.context_length)
            for tok in attributes + classes
        ]
    )
    orig_token_embedding = clip_model.token_embedding(tokenized.to(device))

    with torch.no_grad():
        frozen_embedding = torch.zeros(
            (len(attributes) + len(classes), clip_model.token_embedding.weight.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            frozen_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

    # TODO: clip adapters with csp

    attr_obj_embs = frozen_embedding

    embed_dim = orig_token_embedding.size(-1)
    offset = len(attributes)

    class_token_ids = clip.tokenize(
        [prompt_template],
        context_length=config.context_length,
    )

    # 0.6 is based on their observation that
    # it works well for fine-grained datasets.
    adapter = Adapter(embed_dim, alpha=0.6)
    adapter.to(device)

    optimizer = torch.optim.Adam(
        adapter.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=1e-4
    )

    # only using the visual adapter
    model = CLIPAdapters(
        clip_model,
        config,
        offset,
        adapter,
        attr_obj_embs,
        class_token_ids,
        device=device,
        enable_pos_emb=True
    )


    return model, optimizer

class CLIPAdapters(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config: argparse.ArgumentParser,
        offset: int,
        adapter,
        soft_embeddings,
        token_ids: torch.tensor,
        device: torch.device = "cuda:0",
        enable_pos_emb: bool = True,
    ):
        # Note: soft embeddings aren't really used here.
        super().__init__(
            clip_model,
            config,
            token_ids,
            soft_embeddings=soft_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )
        self.token_ids = token_ids
        self.adapter = adapter
        self.offset = offset


    def construct_token_tensors(self, pair_idx):
        """Function creates the token tensor for further inference.

        Args:
            pair_idx (torch.Tensor): Shape [N x 2], where N is the number
                of pairs of attr and obj

        Returns:
            torch.Tensor: token tensor passed to the text encoder;
                shape [N x context_length x 512]
        """
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.token_ids[0].argmax())
        # soft_embeddings = self.attr_dropout(self.soft_embeddings)
        soft_embeddings = self.soft_embeddings
        token_tensor[:, eos_idx - 2, :] = soft_embeddings[
            attr_idx
        ].type(self.clip_model.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_embeddings[
            obj_idx + self.offset
        ].type(self.clip_model.dtype)

        return token_tensor


    # overriding the class method
    def encode_image(self, imgs):
        # return self.clip_model.encode_image(imgs)
        features = self.clip_model.encode_image(imgs)
        return self.adapter.alpha * self.adapter(features) - (1-self.adapter.alpha) * features