# ComfyUI-DUSt3R
ComfyUI nodes to use [DUSt3R](https://dust3r.europe.naverlabs.com/)

DUSt3R allows you to take reference images and create a 3D point cloud scene.

## Installation

### Pytorch3d
This repo uses Pytorch3d to render DUSt3R scenes. You can find installation details here: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
For any issues regarding installation create issues on their github.

### Python Packages
If you do not install from ComfyUI Manager you can install with pip through `python -m pip -r install requirements.txt`.

### Models

Select a model from DUSt3R here: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

Place the checkpoint in the `ComfyUI/models/dust3r` directory.


## Examples
You can find example workflows in the `example_workflows` directory of this repo. The spline editor example requires KJNodes.

For more info on how to control the camera, the render is based on ViewCrafter's implementation with details here: https://github.com/Drexubery/ViewCrafter/blob/main/docs/render_help.md

https://github.com/user-attachments/assets/d996b7b9-5b03-47d4-8243-93f032727498

https://github.com/user-attachments/assets/34974f90-bbd5-4aa5-b77d-75f6320a746f


