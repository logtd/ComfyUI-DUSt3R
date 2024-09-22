import os

from folder_paths import models_dir

from ..dust3r.dust3r.inference import load_model


DUST3R_PATH = os.path.join(models_dir, 'dust3r')
os.makedirs(DUST3R_PATH, exist_ok=True)


class LoadDust3rNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "checkpoint": (os.listdir(DUST3R_PATH), )}}

    RETURN_TYPES = ("DUST3R",)
    FUNCTION = "load"

    CATEGORY = "dust3r"

    def load(self, checkpoint):
        checkpoint_path = os.path.join(DUST3R_PATH, checkpoint)
        dust3r = load_model(checkpoint_path, 'cpu')
        return (dust3r,)
