import logging
import os
import sys
import copy
import ipywidgets
import json
import kaolin
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import os
import json
from thirdparty.gausplat.utils.graphics_utils import focal2fov
from thirdparty.gausplat.utils.system_utils import searchForMaxIteration
from thirdparty.gausplat.gaussian_renderer import render, GaussianModel
from thirdparty.gausplat.scene.cameras import Camera as GSCamera
try:
    from ipywidgets import Layout
except Exception as e:
    print('WARNING: Could not import ipywidgets, please reinstall it with pip or renderer will not appear correctly.')


class PipelineParamsNoparse:
    """ Same as PipelineParams but without argument parser. """

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(checkpt_dir, f"iteration_{iteration}", "point_cloud.ply")

    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


def try_load_camera(model_path):
    """ Load one of the default cameras for the scene. """
    cam_path = os.path.join(model_path, 'cameras.json')
    if not os.path.exists(cam_path):
        print(f'Could not find saved cameras for the scene at {camp_path}; using default for ficus.')
        return GSCamera(colmap_id=0,
                        R=np.array([[-9.9037e-01, 2.3305e-02, -1.3640e-01], [1.3838e-01, 1.6679e-01, -9.7623e-01],
                                    [-1.6444e-09, -9.8571e-01, -1.6841e-01]]),
                        T=np.array([6.8159e-09, 2.0721e-10, 4.03112e+00]),
                        FoVx=0.69111120, FoVy=0.69111120,
                        image=torch.zeros((3, 800, 800)),  # fake
                        gt_alpha_mask=None, image_name='fake', uid=0)

    with open(cam_path) as f:
        data = json.load(f)
        raw_camera = data[0]

    tmp = np.zeros((4, 4))
    tmp[:3, :3] = raw_camera['rotation']
    tmp[:3, 3] = raw_camera['position']
    tmp[3, 3] = 1
    C2W = np.linalg.inv(tmp)
    R = C2W[:3, :3].transpose()
    T = C2W[:3, 3]
    width = raw_camera['width']
    height = raw_camera['height']
    fovx = focal2fov(raw_camera['fx'], width)
    fovy = focal2fov(raw_camera['fy'], height)
    return GSCamera(colmap_id=0,
                    R=R, T=T, FoVx=fovx, FoVy=fovy,
                    image=torch.zeros((3, height, width)),  # fake
                    gt_alpha_mask=None, image_name='fake', uid=0)


def try_load_kaolin_camera(model_path):
    gs_camera = try_load_camera(model_path)
    return convert_gs_camera(gs_camera)


def compute_cam_fov(intrinsics, axis='x'):
    # compute FOV from focal
    aspectScale = intrinsics.width / 2.0
    tanHalfAngle = aspectScale / (intrinsics.focal_x if axis == 'x' else intrinsics.focal_y).item()
    fov = np.arctan(tanHalfAngle) * 2
    return fov

def convert_kaolin_camera(kal_camera):
    """ Convert kaolin camera to GS camera. """
    R = kal_camera.extrinsics.R[0]
    R[1:3] = -R[1:3]
    T = kal_camera.extrinsics.t.squeeze()
    T[1:3] = -T[1:3]
    return GSCamera(colmap_id=0,
                    R=R.transpose(1, 0).cpu().numpy(),
                    T=T.cpu().numpy(),
                    FoVx=compute_cam_fov(kal_camera.intrinsics, 'x'),
                    FoVy=compute_cam_fov(kal_camera.intrinsics, 'y'),
                    image=torch.zeros((3, kal_camera.height, kal_camera.width)),  # fake
                    gt_alpha_mask=None,
                    image_name='fake',
                    uid=0)

def convert_gs_camera(gs_camera):
    """ Convert GS camera to Kaolin camera. """
    view_mat = gs_camera.world_view_transform.transpose(1, 0)
    view_mat[1:3] = -view_mat[1:3]
    res = kaolin.render.camera.Camera.from_args(
        view_matrix=view_mat,
        width=gs_camera.image_width, height=gs_camera.image_height,
        fov=gs_camera.FoVx, device='cpu')
    return res


class GaussianSplatsRendererSetup:
    def __init__(self, gaussians, kal_cam,
                 log_scale=True, log_opacity=False, timeline=None):
        self.gaussians = gaussians
        self.pipeline = PipelineParamsNoparse()
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.timeline = [] if timeline is None else timeline

        # Uncomment to switch between log scale and scale
        if log_scale:
            scaling_property = gaussians._scaling
            scale_description = "Log Scale"
        else:
            scaling_property = gaussians.get_scaling
            scale_description = "Scale Range"
        self.scaling_property = scaling_property

        if log_opacity:
            opacity_property = gaussians._opacity
            opacity_description = "Log Opacity Range"
        else:
            opacity_property = gaussians.get_opacity
            opacity_description = "Opacity"
        self.opacity_property = opacity_property

        max_sh_deg1_response = gaussians.get_features[:,1:9].max(dim=2)[0].max(dim=1)[0].unsqueeze(-1)
        max_sh_deg2_response = gaussians.get_features[:,4:9].max(dim=2)[0].max(dim=1)[0].unsqueeze(-1)
        max_sh_deg3_response = gaussians.get_features[:,9:].max(dim=2)[0].max(dim=1)[0].unsqueeze(-1)
        max_sh1_description = "Max SH1"
        max_sh2_description = "Max SH2"
        max_sh3_description = "Max SH3"

        # Instantiate slider to control scaling
        scaling = scaling_property.max(dim=1)[0]
        scaling_slider = ipywidgets.FloatRangeSlider(
            value=[scaling.min().item(), scaling.max().item()],
            min=scaling.min().item() -0.01, max=scaling.max().item() + 0.01,
            step=0.001,
            description=f'{scale_description}:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout=Layout(width='1000px')
        )
        self.scaling_slider = scaling_slider

        # Instantiate slider to control opacity
        opacity = opacity_property.max(dim=1)[0]
        opacity_slider = ipywidgets.FloatRangeSlider(
            value=[opacity.min().item(), opacity.max().item()],
            min=opacity.min().item() -0.01, max=opacity.max().item() + 0.01,
            step=0.001,
            description=f'{opacity_description}:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout=Layout(width='1000px')
        )
        self.opacity_slider = opacity_slider

        # Instantiate slider to control SHs
        sh1 = max_sh_deg1_response.max(dim=1)[0]
        sh1_slider = ipywidgets.FloatRangeSlider(
            value=[sh1.min().item(), sh1.max().item()],
            min=sh1.min().item() -0.01, max=sh1.max().item() + 0.01,
            step=0.001,
            description=f'{max_sh1_description}:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout=Layout(width='1000px')
        )
        self.sh1_slider = sh1_slider

        sh2 = max_sh_deg2_response.max(dim=1)[0]
        sh2_slider = ipywidgets.FloatRangeSlider(
            value=[sh2.min().item(), sh2.max().item()],
            min=sh2.min().item() -0.01, max=sh2.max().item() + 0.01,
            step=0.001,
            description=f'{max_sh2_description}:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout=Layout(width='1000px')
        )
        self.sh2_slider = sh2_slider

        # Instantiate slider to control SHs
        sh3 = max_sh_deg3_response.max(dim=1)[0]
        sh3_slider = ipywidgets.FloatRangeSlider(
            value=[sh3.min().item(), sh3.max().item()],
            min=sh3.min().item() -0.01, max=sh3.max().item() + 0.01,
            step=0.001,
            description=f'{max_sh3_description}:',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout=Layout(width='1000px')
        )
        self.sh3_slider = sh3_slider

        attenuate_slider = ipywidgets.FloatSlider(
            value=1.0,
            min=0.0, max=2.0,
            step=0.01,
            description=f'SH+-',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout=Layout(width='1000px')
        )
        self.attenuate_slider = attenuate_slider

        rescale_slider = ipywidgets.FloatSlider(
            value=1.0,
            min=0.001, max=10.0,
            step=0.01,
            description=f'Rescale',
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout=Layout(width='1000px')
        )
        self.rescale_slider = rescale_slider

        def selective_render_kaolin(kaolin_cam):
            """Same rendering as above, but we subsample gaussians based on their scale."""
            # Select only the gaussians with radius below value
            scale_min_mask = scaling_property.min(dim=1)[0] > scaling_slider.value[0]
            scale_max_mask = scaling_property.max(dim=1)[0] < scaling_slider.value[1]
            scale_mask = scale_min_mask & scale_max_mask
            opacity_min_mask = opacity_property.min(dim=1)[0] > opacity_slider.value[0]
            opacity_max_mask = opacity_property.max(dim=1)[0] < opacity_slider.value[1]
            opacity_mask = opacity_min_mask & opacity_max_mask
            sh1_min_mask = max_sh_deg1_response.min(dim=1)[0] > sh1_slider.value[0]
            sh1_max_mask = max_sh_deg1_response.max(dim=1)[0] < sh1_slider.value[1]
            sh1_mask = sh1_min_mask & sh1_max_mask
            sh2_min_mask = max_sh_deg2_response.min(dim=1)[0] > sh2_slider.value[0]
            sh2_max_mask = max_sh_deg2_response.max(dim=1)[0] < sh2_slider.value[1]
            sh2_mask = sh2_min_mask & sh2_max_mask
            sh3_min_mask = max_sh_deg3_response.min(dim=1)[0] > sh3_slider.value[0]
            sh3_max_mask = max_sh_deg3_response.max(dim=1)[0] < sh3_slider.value[1]
            sh3_mask = sh3_min_mask & sh3_max_mask
            mask = scale_mask & opacity_mask & sh1_mask & sh2_mask & sh3_mask
            tmp_gaussians = GaussianModel(gaussians.max_sh_degree)
            tmp_gaussians._xyz = gaussians._xyz[mask, :]
            tmp_gaussians._features_dc = gaussians._features_dc[mask, ...]
            tmp_gaussians._features_rest = gaussians._features_rest[mask, ...] * attenuate_slider.value
            tmp_gaussians._opacity = gaussians._opacity[mask, ...]
            tmp_gaussians._scaling = gaussians._scaling[mask, ...] * rescale_slider.value
            tmp_gaussians._rotation = gaussians._rotation[mask, ...]
            tmp_gaussians.active_sh_degree = gaussians.max_sh_degree

            cam = convert_kaolin_camera(kaolin_cam)
            render_res = render(cam, tmp_gaussians, self.pipeline, self.background)
            rendering = render_res["render"]
            return (rendering.permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu()

        def handle_slider(e):
            self.visualizer.out.clear_output()
            with self.visualizer.out:
                self.visualizer.render_update()

        # Instantiate visualizer with this custom render function
        focus_at = (kal_cam.cam_pos() - 4. * kal_cam.extrinsics.cam_forward()).squeeze()
        self.visualizer = kaolin.visualize.IpyTurntableVisualizer(
            512, 512, copy.deepcopy(kal_cam), selective_render_kaolin,
            focus_at=focus_at, world_up_axis=2, max_fps=12)
        self.visualizer.render_update()

        scaling_slider.observe(handle_slider, names='value')
        opacity_slider.observe(handle_slider, names='value')
        sh1_slider.observe(handle_slider, names='value')
        sh2_slider.observe(handle_slider, names='value')
        sh3_slider.observe(handle_slider, names='value')
        attenuate_slider.observe(handle_slider, names='value')
        rescale_slider.observe(handle_slider, names='value')

    def display(self):
        display(
            self.visualizer.canvas,
            self.visualizer.out,
            self.scaling_slider,
            self.opacity_slider,
            self.sh1_slider,
            self.sh2_slider,
            self.sh3_slider,
            self.attenuate_slider,
            self.rescale_slider
        )