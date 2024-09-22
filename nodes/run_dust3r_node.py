import torch

from ..dust3r.dust3r.inference import inference
from ..dust3r.dust3r.image_pairs import make_pairs
from ..dust3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from ..utils.pvd_utils import get_input_dict


class RunDust3rNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                { 
                    "dust3r": ("DUST3R", ),
                    "images": ("IMAGE",),
                    "clean_pc": ("BOOLEAN", { "default": True }),
                    "batch_size": ("INT", { "default": 8, "min": 1, "max": 99999, "step": 1}),
                    "n_iters": ("INT", {"default": 300, "min": 0, "max": 99999, "step": 1}),
                    "lr": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 99999, "step": 0.01}),
                    "schedule": (["linear"],),
                 }}

    RETURN_TYPES = ("DUST3R_SCENE",)
    FUNCTION = "run"

    CATEGORY = "dust3r"

    def run(self, dust3r, images, clean_pc, batch_size, n_iters, lr, schedule):
        device = 'cuda'
        _, h, w, _ = images.shape
        images = [get_input_dict(image.to(torch.float32)*2-1, idx) for idx, image in enumerate(images)]
        

        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        
        if mode == GlobalAlignerMode.PointCloudOptimizer and n_iters > 0 and lr > 0:
            with torch.inference_mode(False):
                with torch.enable_grad():
                    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
                    output = inference(pairs, dust3r, device, batch_size=batch_size)
                    scene = global_aligner(output, device=device, height=h, width=w, mode=mode)
                    scene.compute_global_alignment(init='mst', niter=n_iters, schedule=schedule, lr=lr)
            scene.requires_grad_(False)
        else:
            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
            output = inference(pairs, dust3r, device, batch_size=batch_size)
            scene = global_aligner(output, device=device, mode=mode)

        if clean_pc:
            scene = scene.clean_pointcloud()
        return (scene,)

