import os

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.structures import Pointclouds

import comfy.model_management

from ..dust3r.dust3r.utils.device import to_numpy
from ..utils.pvd_utils import generate_candidate_poses, generate_traj_specified, generate_traj_txt, setup_renderer, world_point_to_obj


class RenderDust3rSimpleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                { 
                    "scene": ("DUST3R_SCENE", ),
                    "dpt_trd": ("FLOAT", {"default": 1.0, "min": -99999, "max": 99999, "step": 0.01}),
                    "center_scale": ("FLOAT", {"default": 1.0, "min": -99999, "max": 99999, "step": 0.01}),
                    "elevation": ("FLOAT", {"default": 5.0, "min": -99999, "max": 99999, "step": 0.01}),
                    "d_theta": ("FLOAT", {"default": 10.0, "min": -99999, "max": 99999, "step": 0.01}),
                    "d_phi": ("FLOAT", {"default": 30.0, "min": -99999, "max": 99999, "step": 0.01}),
                    "d_r": ("FLOAT", {"default": 0.2, "min": -99999, "max": 99999, "step": 0.01}),
                    "video_length": ("INT", {"default": 16, "min": 1, "max": 99999, "step": 1}),
                    "output_width": ("INT", {"default": 1024, "min": 1, "max": 99999, "step": 1}),
                    "output_height": ("INT", {"default": 576, "min": 1, "max": 99999, "step": 1}),
                 }}

    RETURN_TYPES = ("IMAGE","MASK")
    FUNCTION = "render"
    OUTPUT_NODE = True


    CATEGORY = "dust3r"

    def render_pcd(self,pts3d,imgs,masks,views,renderer):
        device = comfy.model_management.get_torch_device()
        
        imgs = to_numpy(imgs)
        pts3d = to_numpy(pts3d)

        if masks == None:
            pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
            col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
        else:
            # masks = to_numpy(masks)
            pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)
        
        color_mask = torch.ones(col.shape).to(device)

        point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
        images = renderer(point_cloud)
        view_masks = renderer(point_cloud_mask)
        return images, view_masks

    def run_render(self, pcd, imgs,masks, H, W, camera_traj,num_views):
        render_setup = setup_renderer(camera_traj, image_size=(H,W))
        renderer = render_setup['renderer']
        render_results, viewmask = self.render_pcd(pcd, imgs, masks, num_views,renderer)
        return render_results, viewmask

    def render(self, scene, dpt_trd, center_scale, elevation, d_theta, d_phi, d_r, video_length, output_width, output_height):
        list_render = isinstance(d_phi, list) or isinstance(d_r, list) or isinstance(d_theta, list)

        if not isinstance(d_theta, list):
            d_theta = [d_theta]
        if not isinstance(d_phi, list):
            d_phi = [d_phi]
        if not isinstance(d_r, list):
            d_r = [d_r]
        device = comfy.model_management.get_torch_device()

        c2ws = scene.get_im_poses().detach()[1:] 
        principal_points = scene.get_principal_points().detach()[1:] #cx cy
        focals = scene.get_focals().detach()[1:] 
        shape = np.array([[scene.height, scene.width]], dtype=np.int32) # images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in scene.get_pts3d(clip_thred=dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in scene.get_depthmaps()]
        depth_avg = depth[-1][H//2,W//2] #以图像中心处的depth(z)为球心旋转
        radius = depth_avg*center_scale #缩放调整

        ## change coordinate
        c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=elevation, device=device)

        imgs = np.array(scene.imgs)
        
        masks = None

        mode = 'single_view_nbv' if not list_render else 'list_render'

        if mode == 'single_view_nbv':
            num_candidates = 2
            candidate_poses,thetas,phis = generate_candidate_poses(c2ws, H, W, focals, principal_points, d_theta[0], d_phi[0],num_candidates, device)
            _, viewmask = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, candidate_poses,num_candidates)
            nbv_id = torch.argmin(viewmask.sum(dim=[1,2,3])).item()
            theta_nbv = thetas[nbv_id]
            phi_nbv = phis[nbv_id]
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, theta_nbv, phi_nbv, d_r[0],video_length, device)
            elevation -= theta_nbv
        elif mode == 'single_view_target':
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, d_theta[0], d_phi[0], d_r[0],video_length, device)
        elif mode == 'list_render':
            camera_traj,num_views = generate_traj_txt(c2ws, H, W, focals, principal_points, d_phi, d_theta, d_r, video_length, device,viz_traj=False, save_dir =None)
        

        render_results, viewmask = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, camera_traj,num_views)
        
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=(output_height, output_width), mode='bilinear', align_corners=False).permute(0,2,3,1).cpu()
        viewmask = F.interpolate(viewmask.permute(0,3,1,2), size=(output_height, output_width), mode='bilinear', align_corners=False).permute(0,2,3,1)
        viewmask = viewmask[:,:,:,0].cpu()
        return (render_results,viewmask)
    
