import os

from models.clip_adapters import get_clip_adapters
from models.coop import coop
from models.csp import get_csp, get_mix_csp
from models.visual_prompts import get_visual_prompts

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_model(train_dataset, config, device):
    if config.experiment_name == "coop":
        return coop(train_dataset, config, device)

    elif (
        config.experiment_name == "csp"
        or config.experiment_name == "csp_obj"
        or config.experiment_name == "csp_att"
    ):
        return get_csp(train_dataset, config, device)

    elif (
        config.experiment_name == "clip_adapter"
        or config.experiment_name == "clip_adapter_csp"
    ):
        return get_clip_adapters(train_dataset, config, device)

    # special experimental setup
    elif config.experiment_name == "mix_csp":
        return get_mix_csp(train_dataset, config, device)
    elif config.experiment_name == "visual_prompt":
        return get_visual_prompts(config, device)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Experiment Name {:s}.".format(config.experiment_name)
        )
