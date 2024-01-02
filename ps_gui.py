from __future__ import annotations

import copy
import math
import os.path

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from typing import Tuple, Dict, Union
from dataclasses import dataclass
from thirdparty.gausplat.gaussian_renderer import render, GaussianModel
from thirdparty.gausplat.scene.cameras import Camera as GSCamera
from gaussian_splat_utils import TransformableGaussianModel

DEFAULT_DEVICE = torch.device('cuda')
MAX_DEPTH = 10.0  # arbitrary, used for colormap range of depth channel

@dataclass
class GUIConfig:
    program_name: str = 'Simplicits'
    object_name: str = 'Scene'
    model_path: str = 'n/a'
    up_axis: str = 'neg_y_up'
    front_axis: str = 'neg_z_front'
    navigation_style: str = 'free'
    enable_vsync: bool = False
    max_fps: int = -1
    background_color: Tuple = (0., 0., 0.)
    window_size: Tuple = (1920, 1080)

# Cartesian coordinate-system and turntable camera controller fit most post-trained scenes by default
DEFAULT_CONFIG = GUIConfig(up_axis='y_up', front_axis='z_front', navigation_style='turntable')
# NeRF synthetic uses the blender coordinate-system
BLENDER_CONFIG = GUIConfig(up_axis='z_up', front_axis='neg_y_front', navigation_style='turntable')
# Colmap scenes, mip360: cartesian coordinate-system
CARTESIAN_CONFIG = GUIConfig(up_axis='y_up', front_axis='z_front', navigation_style='free')

class GSplatPipelineParams:
    """ Same as PipelineParams but without argument parser. """

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = True
        self.debug = False

def to_np(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().cpu().numpy()


def polyscope_to_gsplat_camera(camera: ps.CameraParameters = None, downsample_factor: int = 1):
    window_w, window_h = ps.get_window_size()
    image_w = window_w // downsample_factor
    image_h = window_h // downsample_factor

    # By default use the current polyscope viewing camera, unless an explicit one was given
    ps_camera = ps.get_view_camera_parameters() if camera is None else camera

    fov_y = ps_camera.get_fov_vertical_deg() * np.pi / 180.0
    aspect = ps_camera.get_aspect()
    fov_x = 2.0 * np.arctan(np.tan(0.5 * fov_y) * aspect)
    gs_cam = GSCamera(
        colmap_id=0,
        R=ps_camera.get_R(),
        T=ps_camera.get_T(),
        FoVx=fov_x,
        FoVy=fov_y,
        image=torch.zeros((3, image_h, image_w)),  # fake
        gt_alpha_mask=None,
        image_name='fake',
        uid=0
    )

    return gs_cam


def project_gaussian_means_to_2d(model: GaussianModel, camera: ps.CameraParameters):
    gaussians_pos = model.get_xyz
    # N: Number of gaussian splats
    N = gaussians_pos.shape[0]
    # Shape (N, 1)
    ones_padding = gaussians_pos.new_ones(N, 1)
    # Shape (N, 4)
    xyz_homogeneous = torch.cat([gaussians_pos, ones_padding], dim=1)
    # Shape (N, 4, 1)
    xyz_homogeneous = xyz_homogeneous.unsqueeze(-1)
    # Shape (4, 4)
    gsplat_cam = polyscope_to_gsplat_camera(camera)
    # Shape (N, 4, 4)
    cam_view_projection_matrix = gsplat_cam.full_proj_transform.T[None].expand(N, 4, 4)
    # Shape (N, 4, 1)
    transformed_xyz = cam_view_projection_matrix @ xyz_homogeneous
    # Shape (N, 4)
    transformed_xyz = transformed_xyz.squeeze(-1)
    # Perform perspective division to obtain (N, 4) of [x_ndc, y_ndc, depth, 1.0]
    transformed_xyz /= transformed_xyz[:, -1:]
    return transformed_xyz


def toggle_off_gspalts(model, mask):
    model._opacity[mask] = -10000.0


def toggle_on_gspalts(model, mask, restored_buffer):
    model._opacity[mask] = restored_buffer[mask]


@dataclass
class CallbackPayload:
    model: GaussianModel = None
    camera: ps.CameraParameters = None
    last_selection: GSplatSelection = None
    selection_preview: GSplatSelection = None
    drag_bounds: Tuple[float, float, float, float] = None   # x0, y0, x1, y1
    segments: dict[str, Segment] = None

@dataclass
class GSplatSelection:
    mask: torch.BoolTensor = None
    """ Holds for each gaussian whether it's selected or not """

    @torch.no_grad()
    def select(self, payload: CallbackPayload):
        transformed_xyz = project_gaussian_means_to_2d(payload.model, payload.camera)
        x0, y0, x1, y1 = payload.drag_bounds
        self.mask = (transformed_xyz[:, 0] >= x0) & (transformed_xyz[:, 0] <= x1) & \
                    (transformed_xyz[:, 1] >= y0) & (transformed_xyz[:, 1] <= y1)

        # Only select gaussians that are currently in any visible segment
        enabled_gaussians = self.mask.new_zeros(self.mask.shape[0], dtype=torch.bool)
        for segment in payload.segments.values():
            if segment.is_enabled:
                enabled_gaussians |= segment.mask
        self.mask &= enabled_gaussians

        return self

    @torch.no_grad()
    def add(self, payload: CallbackPayload):
        other = GSplatSelection().select(payload)
        if self.mask is not None:
            self.mask |= other.mask
        else:
            self.mask = other.mask
        return self

    @torch.no_grad()
    def remove(self, payload: CallbackPayload):
        other = GSplatSelection().select(payload)
        if self.mask is not None:
            self.mask &= ~other.mask
        else:
            self.mask = other.mask & False
        return self

    @torch.no_grad()
    def intersect(self, payload: CallbackPayload):
        other = GSplatSelection().select(payload)
        if self.mask is not None:
            self.mask &= other.mask
        else:
            self.mask = other.mask & False
        return self

    def reset(self):
        self.mask = None
        return self

    def select_all(self, payload: CallbackPayload):
        gaussians_pos = payload.model.get_xyz
        N = gaussians_pos.shape[0]
        self.mask = gaussians_pos.new_ones(N, dtype=torch.bool)
        return self

    def __len__(self):
        return 0 if self.mask is None else torch.sum(self.mask)


class Segment:
    running_counter = 1
    def __init__(self, mask, name=None):
        self.mask = mask
        self.properties = dict()
        if name is None or name == f"Segment {Segment.running_counter}":
            name = f"Segment {Segment.running_counter}"
            Segment.running_counter += 1
        self.name = name
        self.point_cloud = ps.register_point_cloud(self.name, np.zeros([0, 3]), transparency=0.0)
        self.attributes = self.define_attributes()

    def define_attributes(self):
        return {
            "Young's Modulus": 1e3,
            "Rho": 1e2,
            "Poisson Ratio": 0.45,
        }

    @property
    def is_enabled(self):
        return self.point_cloud.is_enabled()

    @is_enabled.setter
    def is_enabled(self, val):
        self.point_cloud.set_enabled(val)

    @property
    def color(self):
        return self.point_cloud.get_color()

    @color.setter
    def color(self, val):
        return self.point_cloud.set_color(val)

    def __getstate__(self):
        """For serialization"""
        state = self.__dict__.copy()
        state['point_cloud'] = None # Handle unpickable fields
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.point_cloud = ps.register_point_cloud(self.name, np.zeros([0, 3]), transparency=0.0)


class DragHandler:

    GIZMOS_GROUP = "GIZMO_ELEMENTS"

    def __init__(self, continuous_selection=False):
        selection_rect = ps.register_curve_network(
            "selection_rect",
            nodes=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            edges="loop",
            color=(1.0, 1.0, 1.0),
            enabled=False
        )
        selection_rect.set_radius(0.005, relative=False)
        selection_rect.set_is_using_screen_coords(True)

        # TODO (operel): Groups should be handled more delicately
        gizmos_group = ps.create_group(DragHandler.GIZMOS_GROUP)
        gizmos_group.set_is_hide_from_ui(True)
        gizmos_group.set_hide_descendants_from_structure_lists(True);
        gizmos_group.set_show_child_details(False)
        selection_rect.add_to_group(gizmos_group)
        # Is a drag event currently in progress
        self.is_dragging: bool = False
        # Last (x,y) where the user clicked -- this is where the drag starts
        self.lastMouseClick: Tuple[float, float] = None
        # Number of pixels (x, y) the user dragged relative to lastMouseClick
        self.lastDragDelta = None
        # Stores the navigation style of polyscope before drag started, as navigation is disabled during drag
        self.navigation_style: Tuple[float, float] = None
        # If true, elements will be selected online during the drag event, without releasing the mouse
        # If false, elements will be selected only when the mouse button is released and drag ends
        self.continuous_selection: bool = continuous_selection

        self.assigned_mouse_button = psim.ImGuiMouseButton_Middle

    @property
    def selection_rect(self):
        return ps.get_curve_network("selection_rect")

    def _drag_start(self):
        self.lastMouseClick = psim.GetMousePos()
        self.navigation_style = ps.get_navigation_style()
        ps.set_navigation_style('none')
        self.is_dragging = True

    def _drag_move(self):
        window_w, window_h = ps.get_window_size()
        self.selection_rect.set_enabled(True)
        currMousePos = psim.GetMousePos()
        self.lastDragDelta = (currMousePos[0] - self.lastMouseClick[0], currMousePos[1] - self.lastMouseClick[1])
        x0 = self.lastMouseClick[0] / window_w
        y0 = self.lastMouseClick[1] / window_h
        x1 = x0 + self.lastDragDelta[0] / window_w
        y1 = y0 + self.lastDragDelta[1] / window_h
        x0 = 2.0 * x0 - 1.0
        x1 = 2.0 * x1 - 1.0
        y0 = 1.0 - 2.0 * y0
        y1 = 1.0 - 2.0 * y1
        self.selection_rect.update_node_positions(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        if self.continuous_selection:
            return x0, y0, x1, y1
        else:
            return None

    def _drag_end(self):
        window_w, window_h = ps.get_window_size()
        currMousePos = psim.GetMousePos()
        self.lastDragDelta = (currMousePos[0] - self.lastMouseClick[0], currMousePos[1] - self.lastMouseClick[1])
        x0 = self.lastMouseClick[0] / window_w
        y0 = self.lastMouseClick[1] / window_h
        x1 = x0 + self.lastDragDelta[0] / window_w
        y1 = y0 + self.lastDragDelta[1] / window_h
        x0 = 2.0 * x0 - 1.0
        x1 = 2.0 * x1 - 1.0
        y0 = 2.0 * y0 - 1.0
        y1 = 2.0 * y1 - 1.0
        return x0, y0, x1, y1

    def handle_callback(self, payload: CallbackPayload):
        io = psim.GetIO()
        drag_bounds = None
        finalize_selection = False   # Should update the payload with the selection
        if psim.IsAnyMouseDown():
            if psim.IsMouseDown(self.assigned_mouse_button) and not self.is_dragging:
                self._drag_start()
            if self.is_dragging and self.lastMouseClick is not None:
                drag_bounds = self._drag_move() # Returns bound only for continuous selection mode, otherwise None
        if psim.IsMouseReleased(self.assigned_mouse_button):
            if self.is_dragging:
                drag_bounds = self._drag_end()
                finalize_selection = True
                ps.set_navigation_style(self.navigation_style)
                self.selection_rect.set_enabled(False)
                self.is_dragging = False
            else:
                # TODO (operel): Fix this case, it would never happen
                payload.last_selection.reset()

        if drag_bounds is not None:
            x0, y0, x1, y1 = drag_bounds
            # Swap to make sure x0, y0 < x1, y1
            x0, x1 = (x1, x0) if x0 > x1 else (x0, x1)
            y0, y1 = (y1, y0) if y0 > y1 else (y0, y1)
            payload.drag_bounds = x0, y0, x1, y1

            if io.KeyShift:
                payload.selection_preview = payload.last_selection.add(payload)
            elif io.KeyAlt:
                payload.selection_preview = payload.last_selection.remove(payload)
            elif io.KeyCtrl:
                payload.selection_preview = payload.last_selection.intersect(payload)
            else:
                payload.selection_preview = GSplatSelection().select(payload)

            if finalize_selection:
                payload.last_selection = payload.selection_preview


class KeyHandler:
    # Uses GLFW key mappings: https://www.glfw.org/docs/3.3/group__keys.html
    import string
    # Key mappings
    KEY_A = 65
    for char_idx, char in enumerate(string.ascii_uppercase[1:]):    # Register the rest of the alphabet-keys
        vars()[f'KEY_{char}'] = KEY_A + char_idx + 1
    KEY_0 = 48
    for digit_idx, digit in enumerate(string.digits[1:]):           # Register the rest of the digits-keys
        vars()[f'KEY_{digit}'] = KEY_0 + digit_idx + 1
    KEY_SPACE = 32
    KEY_ESCAPE = 256
    KEY_ENTER = 257
    KEY_TAB = 258
    KEY_BACKSPACE = 259
    KEY_DELETE = 261
    KEY_CTRL = 341

    def __init__(self):
        pass

    def handle_callback(self, payload: CallbackPayload):
        io = psim.GetIO()

        # Select all
        if io.KeyCtrl and psim.IsKeyDown(KeyHandler.KEY_A):
            payload.selection_preview = GSplatSelection().select_all(payload)
            payload.last_selection = payload.selection_preview

        # Reset selection
        if psim.IsKeyDown(KeyHandler.KEY_SPACE):
            payload.last_selection.reset()


class NewSegmentDialog:
    new_segment_modal_id = "New Segment"

    def __init__(self, segments):
        self.current_mask = None
        self.should_open = False
        self.pending_name = None
        self.segments = segments

    def open(self, mask):
        self.current_mask = mask
        self.should_open = True
        self.pending_name = f"Segment {Segment.running_counter}"
        psim.OpenPopup(self.new_segment_modal_id)

    def enforce_partiton_of_union(self, new_segment):
        """ Make sure to remove gaussians of new segment from existing segments, to enforce partition of union """
        for seg_name, existing_segment in self.segments.items():
            if new_segment == existing_segment:
                continue
            existing_segment.mask &= ~new_segment.mask

    def show_dialog(self):
        if self.should_open and \
            psim.BeginPopupModal(self.new_segment_modal_id, self.should_open, psim.ImGuiWindowFlags_AlwaysAutoResize):
            if not psim.IsAnyItemActive():
                psim.SetKeyboardFocusHere(0)
            _, self.pending_name = psim.InputText("Segment Name", self.pending_name, psim.ImGuiInputTextFlags_AutoSelectAll)
            if psim.Button("Ok") or psim.IsKeyDown(KeyHandler.KEY_ENTER):
                segment = Segment(mask=copy.deepcopy(self.current_mask.detach()), name=self.pending_name)
                self.segments[segment.name] = segment
                self.current_mask = None
                self.should_open = False
                self.enforce_partiton_of_union(segment)
                psim.CloseCurrentPopup()
            elif psim.IsKeyDown(KeyHandler.KEY_ESCAPE):
                self.current_mask = None
                self.should_open = False
                psim.CloseCurrentPopup()
            psim.EndPopup()
            return True
        else:
            return False

class StageForcesDialog:
    stage_forces_modal_id = "Stage Forces"

    def __init__(self):
        self.simplicits_simulator = None
        self.available_mask = None
        self.is_open = False
        self.temp_fixed_bc_mask = None
        self.temp_moving_bc_mask = None
        self.current_fixed_bc_mask = None
        self.current_moving_bc_mask = None
        self.object_name = None

    def open(self, payload: CallbackPayload, simplicits_simulator: SimplicitsSimulator):
        self.simplicits_simulator = simplicits_simulator
        object_name = simplicits_simulator.cfg.object_name
        staged_segments = simplicits_simulator.cfg.segments

        gaussians_pos = payload.model.get_xyz
        N = gaussians_pos.shape[0]
        available_mask = gaussians_pos.new_zeros(N, dtype=torch.bool)
        for segment in staged_segments.values():
            available_mask |= segment.mask
        self.available_mask = available_mask
        self.temp_fixed_bc_mask = gaussians_pos.new_zeros(N, dtype=torch.bool)
        self.temp_moving_bc_mask = gaussians_pos.new_zeros(N, dtype=torch.bool)
        self.object_name = object_name
        self.is_open = True

    def show_dialog(self, payload: CallbackPayload):
        if not self.is_open:
            self.simplicits_simulator = None
            self.available_mask = None
            self.is_open = False
            self.temp_fixed_bc_mask = None
            self.temp_moving_bc_mask = None
            self.object_name = None
        else:
            _, self.is_open = psim.Begin(self.window_name, True, psim.ImGuiWindowFlags_AlwaysAutoResize)
            if psim.Button("Set Fixed Boundary Conditions", (250, 30)):
                if payload.last_selection is not None and payload.last_selection.mask is not None:
                    self.temp_fixed_bc_mask = payload.last_selection.mask & self.available_mask
            if psim.Button("Set Moving Boundary Conditions", (250, 30)):
                if payload.last_selection is not None and payload.last_selection.mask is not None:
                    self.temp_moving_bc_mask = payload.last_selection.mask & self.available_mask
            if psim.Button("Ok") or psim.IsKeyDown(KeyHandler.KEY_ENTER):
                self.current_fixed_bc_mask = self.temp_fixed_bc_mask
                self.current_moving_bc_mask = self.temp_moving_bc_mask
                simulated_objects_mask = torch.from_numpy(self.simplicits_simulator.np_object['SimulatedMask'])
                simulated_objects_mask = simulated_objects_mask.to(self.current_fixed_bc_mask.device)
                gravity = -9.8
                scene = SimulatedScene(
                    Name=self.object_name + 'Scene',
                    Steps=50,
                    Gravity=(0, gravity * math.cos(math.radians(30)), gravity * math.sin(math.radians(30)))
                )
                scene.add_object(
                    name=self.object_name,
                    position_delta=(0, 0, 0),
                    fixed_bc=self.current_fixed_bc_mask[simulated_objects_mask],
                    moving_bc=self.current_moving_bc_mask[simulated_objects_mask]
                )
                self.simplicits_simulator.set_scene(scene)
                self.is_open = False
            psim.SameLine()
            if psim.Button("Cancel") or psim.IsKeyDown(KeyHandler.KEY_ESCAPE):
                self.is_open = False
            psim.End()
            return True
        return False

    @property
    def window_name(self):
        return f'{self.stage_forces_modal_id} ({self.object_name})'

class GUI:
    def __init__(self, conf: GUIConfig, model, initial_camera: GSCamera = None, scene_bbox=None, dataset=None):

        self.conf = conf
        ps.set_program_name(conf.program_name)
        ps.set_use_prefs_file(False)
        ps.set_up_dir(conf.up_axis)
        ps.set_front_dir(conf.front_axis)
        ps.set_navigation_style(conf.navigation_style)
        ps.set_enable_vsync(conf.enable_vsync)
        ps.set_max_fps(conf.max_fps)
        ps.set_background_color(conf.background_color)
        ps.set_ground_plane_mode("none")
        ps.set_window_resizable(True)
        ps.set_window_size(*conf.window_size)
        ps.set_give_focus_on_show(True)

        ps.set_automatically_compute_scene_extents(True)
        # ps.set_automatically_compute_scene_extents(False)
        # ps.set_bounding_box(to_np(scene_bbox[0]), to_np(scene_bbox[1]))

        # viz stateful parameters & options
        self.viz_render_color_buffer = None  # buffer for color outputs
        self.viz_render_scalar_buffer = None # buffer for color mapped scalar outputs
        self.live_update = True # if disabled , will skip rendering updates to accelerate background training loop
        self.viz_render_enabled = True # if disabled, rendering will not take place
        self.viz_render_downsample = 1  # downsample factor for rendering resolution, output will be stretched on window
        self.viz_render_modes = ['rgb', 'alpha', 'depth', 'semantic']  # available channels for rendering
        self.viz_render_mode_ind = 0    # index of default rendering mode

        self.viz_render_name = 'gsplat object' # object name in scene graph

        self.dataset = dataset  # optional, for visualizing assets like cameras
        self.model = model      # the scene graph of objects displayed
        self.preview_buffers = dict(
            _features_dc=copy.deepcopy(model._features_dc.detach()),
            _opacity=copy.deepcopy(model._opacity.detach())
        )
        self.pipeline = GSplatPipelineParams()
        self.background = torch.tensor(conf.background_color, dtype=torch.float32, device=DEFAULT_DEVICE)

        # self.viz_do_train = False
        # self.viz_bbox = False
        # self.viz_curr_render_size = None
        # self.viz_curr_render_style_ind = None
        # self.viz_render_train_view = False

        ps.init()

        # Initial camera parameters were explicitly given, initialize from them
        if initial_camera is not None:
            # Set view matrix from GS camera
            view_mat = np.zeros((4, 4))
            view_mat[:3, :3] = initial_camera.R
            view_mat[:3, 3] = initial_camera.T
            view_mat[3, 3] = 1.0

            # Use vertical fov from GS camera and compute horizontal fov with the window aspect ratio
            aspect = ps.get_view_camera_parameters().get_aspect()
            fov_y = initial_camera.FoVy
            fov_x = 2.0 * np.arctan(np.tan(0.5 * fov_y) * aspect)

            ps_cam_param = ps.CameraParameters(
                ps.CameraIntrinsics(
                    fov_vertical_deg=np.degrees(fov_y),
                    fov_horizontal_deg=np.degrees(fov_x)
                ),
                ps.CameraExtrinsics(mat=view_mat)
            )
            ps.set_view_camera_parameters(ps_cam_param)
        else:
            # If no camera is given, use some default lookat values that work for NeRF synthetic
            ps.look_at(camera_location=(0.54, 3.93, 0.67), target=(0.68, 4.91, 0.84))

        # self.ps_point_cloud = ps.register_point_cloud("centers", to_np(model.get_positions()),
        #                         radius=1e-3, point_render_mode='quad')
        # self.ps_point_cloud_buffer = self.ps_point_cloud.get_buffer("points")

        if dataset is not None:
            self.create_dataset_camera_visualization(dataset)

        if scene_bbox is not None:
            bbox_min, bbox_max = scene_bbox
            nodes = np.array([[bbox_min[0], bbox_min[1], bbox_min[2]], [bbox_max[0], bbox_min[1], bbox_min[2]], [bbox_min[0], bbox_max[1], bbox_min[2]],
                                [bbox_min[0], bbox_min[1], bbox_max[2]], [bbox_max[0], bbox_max[1], bbox_min[2]], [bbox_max[0], bbox_min[1], bbox_max[2]],
                                [bbox_min[0], bbox_max[1], bbox_max[2]], [bbox_max[0], bbox_max[1], bbox_max[2]]])
            edges = np.array(
                [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 6], [2, 4], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]])
            ps.register_curve_network("bbox", nodes, edges)

        self.segments: Dict[str, Segment] = self.init_segments()

        ps.set_user_callback(self.ps_ui_callback)
        ps.set_structure_callback(self.ps_structure_callback)
        self.new_segment_dialog = NewSegmentDialog(segments=self.segments)
        self.stage_forces_dialog = StageForcesDialog()
        self.drag_handler = DragHandler()
        self.key_handler = KeyHandler()
        # Holds the info of the currently existing selection in the gui
        self.current_selection: GSplatSelection = GSplatSelection()
        # Holds the info on current boundary conditions (type -> mask of splats affected by a certain bc)
        self.boundary_conditions: Dict[str, torch.BoolTensor] = dict()

        # Drives the dynamics training and simulation, only initialized upon request
        self.simplicits_simulator: SimplicitsSimulator = None

        # Update once to popualate lazily-created structures
        self.update_render_view_viz(force=True)

    def init_segments(self):
        segments: Dict[str, Segment] = dict()
        N = self.model.get_xyz.shape[0]
        segments['Background'] = Segment(mask=self.model.get_xyz.new_ones(N, dtype=torch.bool), name="Background")
        return segments

    def create_camera_visualization(self, cam_list):
        '''
        Given a list-of-dicts of camera & image info, register them in polyscope
        to create a visualization
        '''
        for i_cam, cam in enumerate(cam_list):

            ps_cam_param = ps.CameraParameters(
                ps.CameraIntrinsics(
                    fov_vertical_deg=np.degrees(cam['fov_h']),
                    fov_horizontal_deg=np.degrees(cam['fov_w'])
                ),
                ps.CameraExtrinsics(mat=cam['ext_mat'])
            )

            cam_color = (1., 1., 1.)
            if cam['split'] == 'train':
                cam_color = (1., .7, .7)
            elif cam['split'] == 'val':
                cam_color = (.7, .1, .7)

            ps_cam = ps.register_camera_view(f"{cam['split']}_view_{i_cam:03d}", ps_cam_param, widget_color=cam_color)

            ps_cam.add_color_image_quantity("target image", cam['rgb_img'][:, :, :3], enabled=True)

    def create_dataset_camera_visualization(self, dataset):

        # just one global intrinsic mat for now
        intrinsics = to_np(dataset.K)

        cam_list = []

        for i_cam, pose in enumerate(dataset.poses):
            trans_mat = np.eye(4)
            trans_mat[:3, :4] = pose
            trans_mat_world_to_camera = np.linalg.inv(trans_mat)

            # these cameras follow the opposite convention from polyscope
            camera_convention_rot = np.array([[1., 0., 0., 0., ],
                                              [0., -1., 0., 0., ],
                                              [0., 0., -1., 0., ],
                                              [0., 0., 0., 1., ]])
            trans_mat_world_to_camera = camera_convention_rot @ trans_mat_world_to_camera

            w = dataset.image_w
            h = dataset.image_h
            f_w = intrinsics[0, 0]
            f_h = intrinsics[1, 1]

            fov_w = 2. * np.arctan(0.5 * w / f_w)
            fov_h = 2. * np.arctan(0.5 * h / f_h)

            rgb = to_np(dataset.rgbs[i_cam, :]).reshape(h, w, 3)

            cam_list.append({
                'ext_mat': trans_mat_world_to_camera,
                'w': w,
                'h': h,
                'fov_w': fov_w,
                'fov_h': fov_h,
                'rgb_img': rgb,
                'split': dataset.split,
            })

        self.create_camera_visualization(cam_list)

    def update_cloud_viz(self):

        # re-initialize the viz
        if self.ps_point_cloud is None or self.ps_point_cloud.n_points() != self.model.get_positions().shape[0]:
            self.ps_point_cloud = ps.register_point_cloud("centers", to_np(self.model.get_positions()))
            self.ps_point_cloud_buffer = self.ps_point_cloud.get_buffer("points")

        # direct on-GPU update, must not have changed size
        self.ps_point_cloud_buffer.update_data_from_device(self.model.get_positions().detach())

    @torch.no_grad()
    def render_gaussians(self, render_mode='rgb', camera: ps.CameraParameters = None):
        model_features = self.model._features_dc
        model_opacities = self.model._opacity
        try:
            self.model._features_dc = self.preview_buffers['_features_dc']
            self.model._opacity = self.preview_buffers['_opacity']
            # Convert polyscope camera to gsplats camera
            cam = polyscope_to_gsplat_camera(camera, downsample_factor=self.viz_render_downsample)
            # Render depending on required channel
            if render_mode == 'rgb':
                outputs = render(cam, self.model, self.pipeline, self.background)
                renderbuffer = torch.clamp(outputs["render"], 0.0, 1.0)
                renderbuffer = renderbuffer.permute(1, 2, 0)
            elif render_mode == 'alpha':
                outputs = render(cam, self.model, self.pipeline, self.background, override_color=torch.ones_like(self.model.get_xyz))
                renderbuffer = torch.clamp(outputs["render"][0], 0.0, 1.0)
            elif render_mode == 'depth':
                # TODO (operel): gsplats may have a bug with the proj-matrix (from github),
                #   so as a temp hack we go for camera space Z distance rather than screen space depth (which is normalized
                #   with near / far planes from camera)
                depth_by_3d = True
                xyz = self.model.get_xyz
                if depth_by_3d:
                    background = torch.ones_like(self.background) * MAX_DEPTH
                    R = torch.from_numpy(cam.R).to(device=xyz.device, dtype=xyz.dtype)
                    t = torch.from_numpy(cam.T).to(device=xyz.device, dtype=xyz.dtype)
                    override_color = xyz @ R + t
                else:
                    gaussian_means_ndc = project_gaussian_means_to_2d(self.model, camera)
                    depth = gaussian_means_ndc[:, 2]
                    background = torch.ones_like(self.background)
                    override_color = torch.stack([depth, depth, depth], dim=1)
                outputs = render(cam, self.model, self.pipeline, background, override_color=override_color)
                renderbuffer = torch.clamp(outputs["render"][2], 0.0, MAX_DEPTH)
            elif render_mode == 'semantic':
                for segment in self.segments.values():
                    self.model._features_dc[segment.mask] = self.model._features_dc.new_tensor(segment.color)
                outputs = render(cam, self.model, self.pipeline, self.background)
                renderbuffer = torch.clamp(outputs["render"], 0.0, 1.0)
                renderbuffer = renderbuffer.permute(1, 2, 0)
            else:
                raise ValueError('Unsupported render mode for gsplats')
        finally:
            self.model._features_dc = model_features
            self.model._opacity = model_opacities
        return renderbuffer

    @torch.no_grad()
    def update_render_view_viz(self, force=False):

        window_w, window_h = ps.get_window_size()
        window_w = window_w // self.viz_render_downsample
        window_h = window_h // self.viz_render_downsample

        # re-initialize if needed
        style = self.viz_render_modes[self.viz_render_mode_ind]
        if force or self.viz_curr_render_style_ind !=  self.viz_render_mode_ind or  self.viz_curr_render_size != (window_w, window_h):
            self.viz_curr_render_style_ind = self.viz_render_mode_ind
            self.viz_curr_render_size = (window_w, window_h)

            if style in ("rgb", "semantic"):

                dummy_image = np.ones((window_h, window_w, 4), dtype=np.float32)

                ps.add_color_alpha_image_quantity(
                    self.viz_render_name,
                    dummy_image,
                    enabled=self.viz_render_enabled,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                )

                self.viz_render_color_buffer = ps.get_quantity_buffer(self.viz_render_name, "colors")
                self.viz_render_scalar_buffer = None
            
            elif style == "alpha":
            
                dummy_vals = np.zeros((window_h, window_w), dtype=np.float32)
                dummy_vals[0] = 1.0  # hack so the default polyscope scale gets set more nicely

                self.viz_main_image = ps.add_scalar_image_quantity(
                    self.viz_render_name,
                    dummy_vals,
                    enabled=self.viz_render_enabled,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                    cmap="spectral",
                    vminmax=(0, 1),
                )

                self.viz_render_color_buffer = None
                self.viz_render_scalar_buffer = ps.get_quantity_buffer(self.viz_render_name, "values")

            elif style == "depth":
            
                dummy_vals = np.zeros((window_h, window_w), dtype=np.float32)
                dummy_vals[0] = MAX_DEPTH  # hack so the default polyscope scale gets set more nicely

                ps.add_scalar_image_quantity(
                    self.viz_render_name,
                    dummy_vals,
                    enabled=True,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                    cmap="jet",
                    vminmax=(0, MAX_DEPTH),
                )

                self.viz_render_color_buffer = None
                self.viz_render_scalar_buffer = ps.get_quantity_buffer(self.viz_render_name, "values")

        N = len(self.model._opacity)
        mask = self.model._opacity.new_ones(N, dtype=torch.bool)
        # dummy_buffer = np.ones((window_h, window_w, 4), dtype=np.float32)
        for segment in self.segments.values():

            # segment.point_cloud.add_scalar_quantity("mask", to_np(segment.mask))

            # pcl_buffer = segment.point_cloud.get_buffer("points")
            # pcl_buffer.update_data_from_device(self.model.get_positions().detach())
            # ps.add_color_alpha_image_quantity(
            #     segment.name,
            #     dummy_buffer,
            #     enabled=segment.is_enabled,
            #     image_origin="upper_left",f
            #     show_fullscreen=True,
            #     show_in_imgui_window=False,
            # )
            if not segment.is_enabled:
                mask &= ~segment.mask
            self.preview_buffers['_opacity'] = copy.deepcopy(self.model._opacity.detach())
            self.preview_buffers['_opacity'][~mask] = -10000.0

        # do the actual rendering
        rb = self.render_gaussians(render_mode=style)

        # update the data
        if style in ("rgb", "semantic"):
            # append 1s for alpha
            rb = torch.cat((rb, torch.ones_like(rb[:, :, 0:1])), dim=-1)
            self.viz_render_color_buffer.update_data_from_device(rb.detach())
        elif style in ("alpha", "depth"):
            self.viz_render_scalar_buffer.update_data_from_device(rb.detach())

        with torch.enable_grad():
            ps.frame_tick()

    def populate_rolling_buffers(self):
        self.ps_point_cloud.add_scalar_quantity("rolling_error", to_np(self.model.rolling_error[:,0]))
        self.ps_point_cloud.add_scalar_quantity("rolling_weights", to_np(self.model.rolling_weight_contrib[:,0]))
        self.ps_point_cloud.add_scalar_quantity("rolling_error / rolling_weight_contrib", to_np(self.model.rolling_error[:,0]/self.model.rolling_weight_contrib[:,0]))
        self.ps_point_cloud.add_scalar_quantity("opacity", to_np(self.model.get_density()[:,0]))
        self.ps_point_cloud.add_scalar_quantity("scale_min", to_np(torch.min(self.model.get_scale(), dim=1).values))
        self.ps_point_cloud.add_scalar_quantity("scale_max", to_np(torch.max(self.model.get_scale(), dim=1).values))

    def get_model_id(self):
        try:
            return self.conf.model_path.split(os.path.sep)[-1]
        except:
            return "n/a"

    def ps_structure_callback(self):

        if psim.CollapsingHeader("Entities", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
            if psim.TreeNode(f"{self.conf.object_name} (gsplats)"):
                psim.Text(f"Model: {self.get_model_id()}")
                psim.Text(f"Total Count: {self.model.get_xyz.shape[0]}")
                psim.TreePop()
            psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
            if psim.TreeNode("Selection (gsplats)"):
                selection_count = 0
                if self.current_selection is not None and self.current_selection.mask is not None:
                    selection_count = self.current_selection.mask.sum()
                psim.Text(f"Total Count: {selection_count}")
                psim.TreePop()

        if psim.CollapsingHeader("Segments", psim.ImGuiTreeNodeFlags_DefaultOpen):
            to_delete = []
            for segment in self.segments.values():
                psim.PushId(segment.name)
                psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
                if psim.TreeNode(f'{segment.name} [Count: {segment.mask.sum()}]'):
                    _, segment.is_enabled = psim.Checkbox("Enabled", segment.is_enabled)
                    psim.SameLine()
                    _, segment.color = psim.ColorEdit3("Color", segment.color, psim.ImGuiColorEditFlags_NoInputs)
                    psim.SameLine()
                    psim.Button(f'Options')
                    if psim.Button(f'Select'):
                        self.current_selection.mask = copy.deepcopy(segment.mask.detach())
                    if segment.name != 'Background':
                        psim.SameLine()
                        if psim.Button(f'Delete'):
                            to_delete.append(segment)

                    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
                    if psim.TreeNode("Material Properties"):
                        to_write = dict()
                        for att_name, att_value in segment.attributes.items():
                            psim.PushId(att_name)
                            # psim.ImGuiSliderFlags_Logarithmic
                            psim.PushItemWidth(100)
                            is_updated, new_value = psim.SliderFloat(att_name, att_value, 1e-3, 1e4, "%.4f")
                            psim.PopItemWidth()
                            if is_updated:
                                to_write[att_name] = new_value
                            psim.PopID()
                        for att_name, new_val in to_write.items():
                            segment.attributes[att_name] = new_val
                        psim.TreePop()

                    # Boundary Conditions, if any
                    seg_fixed_bc = seg_moving_bc = None
                    fixed_bc = self.boundary_conditions.get('fixed_bc')
                    moving_bc = self.boundary_conditions.get('moving_bc')
                    if fixed_bc is not None:
                        seg_fixed_bc = segment.mask & fixed_bc
                        seg_fixed_bc = seg_fixed_bc if seg_fixed_bc.sum() > 0 else None
                    if moving_bc is not None:
                        seg_moving_bc = segment.mask & moving_bc
                        seg_moving_bc = seg_moving_bc if seg_moving_bc.sum() > 0 else None
                    if seg_fixed_bc is not None or seg_moving_bc is not None:
                        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
                        if psim.TreeNode("Boundary Conditions"):
                            if seg_fixed_bc is not None:
                                if psim.Button(f'Show Fixed Boundary Cond.'):
                                    self.current_selection.mask = seg_fixed_bc
                            if seg_moving_bc is not None:
                                if psim.Button(f'Show Moving Boundary Cond.'):
                                    self.current_selection.mask = seg_moving_bc
                            psim.TreePop()
                    psim.TreePop()
                psim.PopID()

        for seg in to_delete:
            self.segments['Background'].mask |= seg.mask
            del self.segments[seg.name]

    def draw_main_menu(self, payload: CallbackPayload):
        if psim.BeginMainMenuBar():
            if psim.BeginMenu("File"):
                if psim.MenuItem("New", "Ctrl+N"):
                    self.segments.clear()
                    segments = self.init_segments()
                    self.segments.update(segments)
                    self.current_selection = GSplatSelection()
                if psim.MenuItem("Save Segments", "Ctrl+S"):
                    filename = f'{self.get_model_id()}_segments.pt'
                    torch.save(self.segments, filename)
                    ps.warning(f"Saved segments state to: {os.getcwd()}{os.path.sep}{filename}")
                psim.MenuItem("Save As..", enabled=False)
                if psim.MenuItem("Load Segments", "Ctrl+L"):
                    filename = f'{self.get_model_id()}_segments.pt'
                    try:
                        segments = torch.load(filename)
                        ps.warning(f"Loaded segments state from: {os.getcwd()}{os.path.sep}{filename}")
                        self.segments.clear()
                        self.segments.update(segments)
                        self.current_selection = GSplatSelection()
                    except FileNotFoundError as e:
                        ps.error(f"Missing segment state file at: {os.getcwd()}{os.path.sep}{filename}")
                psim.EndMenu()
            if psim.BeginMenu("Edit"):
                psim.MenuItem("Undo", "Ctrl+Z", enabled=False)
                psim.MenuItem("Redo", "Ctrl+Shift+Z", enabled=False)
                psim.EndMenu()
            if psim.BeginMenu("Run"):
                if psim.MenuItem("Setup New Handles Training", "Ctrl+1"):
                    foreground_segments = {k:v for k,v in self.segments.items() if k != 'Background'}
                    self.setup_new_simulation(foreground_segments)
                if psim.MenuItem("Train Handles", "Ctrl+2", enabled=self.simplicits_simulator is not None):
                    with torch.enable_grad():
                        self.simplicits_simulator.run_training()
                if psim.MenuItem("Load Trained Handles", "Ctrl+3"):
                    foreground_segments = {k: v for k, v in self.segments.items() if k != 'Background'}
                    self.load_existing_simulation_setup(foreground_segments)
                if psim.MenuItem("Stage Scene..", "Ctrl+4",
                                 enabled=self.simplicits_simulator is not None and \
                                         self.stage_forces_dialog is not None and \
                                         not self.stage_forces_dialog.is_open):
                    self.stage_forces_dialog.open(payload=payload, simplicits_simulator=self.simplicits_simulator)
                if psim.MenuItem("Load Scene", "Ctrl+5",
                                 enabled=self.simplicits_simulator is not None):
                    self.simplicits_simulator.load_scene('poke_z')  # TODO (operel): Take as input
                if psim.MenuItem("Simulate Dynamics", "Ctrl+6",
                                 enabled=self.simplicits_simulator is not None and \
                                         self.simplicits_simulator.scene is not None):
                    with torch.enable_grad():
                        self.model = self.simplicits_simulator.simulate_scene()
                        self.prune_redundant_gaussians()    # TODO (operel): Hacky way around out of mem
                        self.simplicits_simulator.cfg.model = self.model
                if psim.MenuItem("Load Previous Simulation", "Ctrl+7",
                                 enabled=self.simplicits_simulator is not None and \
                                         self.simplicits_simulator.scene is not None):
                    self.model = self.simplicits_simulator.load_simulated_scene()
                    self.prune_redundant_gaussians()    # TODO (operel): Hacky way around out of mem
                    self.simplicits_simulator.cfg.model = self.model
                psim.EndMenu()
            psim.EndMainMenuBar()

    def prune_redundant_gaussians(self, threshold = 0.35):
        """ To save mem """
        opacity_mask = self.model.get_opacity[:, 0] >= threshold
        self.model._xyz = self.model._xyz[opacity_mask]
        self.model._features_dc = self.model._features_dc[opacity_mask]
        self.model._features_rest = self.model._features_rest[opacity_mask]
        self.model._scaling = self.model._scaling[opacity_mask]
        self.model._rotation = self.model._rotation[opacity_mask]
        self.model._opacity = self.model._opacity[opacity_mask]

        if isinstance(self.model, TransformableGaussianModel):
            model_mask = opacity_mask.to(self.model.x0.device)
            self.model.x0 = self.model.x0[model_mask]
            self.model.animated_mask = self.model.animated_mask[model_mask]
            self.model.init_weights()

        for segment in self.segments.values():
            segment.mask = segment.mask[opacity_mask]

        if self.current_selection is not None and self.current_selection.mask is not None:
            self.current_selection.mask = self.current_selection.mask[opacity_mask]

        self.preview_buffers = {buffer_name: buffer[opacity_mask]
                                for buffer_name, buffer in self.preview_buffers.items()}

        self.boundary_conditions = {buffer_name: buffer[opacity_mask]
                                    for buffer_name, buffer in self.boundary_conditions.items()}


    @torch.no_grad()
    def ps_ui_callback(self):
        payload = CallbackPayload(
            model=self.model,
            camera=ps.get_view_camera_parameters(),
            last_selection=self.current_selection,
            selection_preview=self.current_selection,
            segments=self.segments,
        )

        # Draw main menu and possibly open dialogs
        self.draw_main_menu(payload)

        # Handle dialog of boundary conditions - show it if needed, and update state with latest boundary conditions
        if self.stage_forces_dialog.show_dialog(payload): # Only shows if dialog is open
            self.boundary_conditions['fixed_bc'] = self.stage_forces_dialog.current_fixed_bc_mask
            self.boundary_conditions['moving_bc'] = self.stage_forces_dialog.current_moving_bc_mask

        if self.new_segment_dialog.show_dialog(): # If modal dialog is open avoid the rest of the logic
            return

        if self.model is not None and isinstance(self.model, TransformableGaussianModel):
            psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
            if psim.TreeNode("Animation"):
                _, keyframe = psim.SliderInt("Timestep", self.model.keyframe, 0, self.model.num_keyframes)
                self.model.set_timestep(keyframe)
                psim.TreePop()

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Training"):
            # _, self.viz_do_train = psim.Checkbox("Train", self.viz_do_train)
            # psim.SameLine()
            _, self.live_update = psim.Checkbox("Update View", self.live_update)

            psim.TreePop()

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Render"):
            psim.PushItemWidth(100)

            if(psim.Button("Show")):
                self.viz_render_enabled = True
                self.update_render_view_viz(force=True)
            psim.SameLine()
            if(psim.Button("Hide")):
                self.viz_render_enabled = False
                self.update_render_view_viz(force=True)


            _, self.viz_render_mode_ind = psim.Combo("Style", self.viz_render_mode_ind, self.viz_render_modes)

            changed, self.viz_render_downsample = psim.InputInt("Downsample Res. Factor", self.viz_render_downsample, 1)
            if changed:
                self.viz_render_downsample = max(self.viz_render_downsample, 1)
            
            # _, self.viz_render_train_view = psim.Checkbox("render w/ train=True", self.viz_render_train_view)

            # psim.SameLine()
            # if(psim.Button("Viz rolling buffers")):
            #     self.update_cloud_viz()
            #     self.populate_rolling_buffers()

            psim.PopItemWidth()
            psim.TreePop()

            with torch.no_grad():
                self.drag_handler.handle_callback(payload)
                self.key_handler.handle_callback(payload)
                self.current_selection = payload.last_selection

                # Update selection color
                self.preview_buffers['_features_dc'] = copy.deepcopy(self.model._features_dc.detach())
                if self.current_selection is not None and len(self.current_selection) > 0:
                    self.preview_buffers['_features_dc'][self.current_selection.mask] = 1.0
                    # self.preview_buffers['_features_dc'][self.current_selection.mask] = torch.tensor([0.0, 0.0, 1.0], device='cuda')

                io = psim.GetIO()

                # Erase command
                if (psim.IsKeyDown(KeyHandler.KEY_BACKSPACE) or psim.IsKeyDown(KeyHandler.KEY_DELETE)) and io.KeyShift:
                    toggle_off_gspalts(self.model, ~self.current_selection.mask)
                # Erase inverse command
                elif psim.IsKeyDown(KeyHandler.KEY_BACKSPACE) or psim.IsKeyDown(KeyHandler.KEY_DELETE):
                    toggle_off_gspalts(self.model, self.current_selection.mask)
                # New segment command
                elif psim.IsKeyDown(KeyHandler.KEY_S) and self.current_selection.mask is not None:
                    self.new_segment_dialog.open(mask=self.current_selection.mask)

##########

    def setup_new_simulation(self, segment: Segment=None):
        sim_config = SimulationConfig(
            object_name=self.conf.object_name,
            segments=segment if segment is not None else self.segments,
            model=self.model,
            new_training=True
        )
        self.simplicits_simulator = SimplicitsSimulator(sim_config)
        if isinstance(segment, dict):
            print(f"Creating new simulation setup for object {self.conf.object_name} on segments: " +
                  f", ".join({seg_name for seg_name in segment.keys()}))

    def load_existing_simulation_setup(self, segment: Segment=None):
        sim_config = SimulationConfig(
            object_name=self.conf.object_name,
            segments=segment if segment is not None else self.segments,
            model=self.model,
            new_training=False
        )
        self.simplicits_simulator = SimplicitsSimulator(sim_config)
        print(f"Loading existing simulation setup for object {self.conf.object_name}")
        if isinstance(segment, dict):
            print(f"Simulated segments: " +
                  f", ".join({seg_name for seg_name in segment.keys()}))


@dataclass
class SimulationConfig:

    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    object_name: str = ""                     # Object name
    segments: Union[Segment, Dict] = None
    model: GaussianModel = None
    new_training: bool = False                # If False, will try to load previous training, else start new training
    poisson_infilling: bool = True              # If True, will fill the object volume with additional sample points

    training_id: str = ""                     # Identifier for this dynamics training session
    training_num_skinning_handles: int = 40   # Number of skinning handles
    training_num_steps: int = 30000           # Training steps
    training_num_sample_pts: int  = 1000      # Sampled pts during training
    training_LRStart: float = 1e-3            # LR scheduling
    training_LREnd: float = 1e-4              # LR scheduling end (linear scheduling)
    training_TSamplingStdev: float = 1        # Magnitude of random deformations:
                                              #  magnitude of deformation applied per handle
    # batch_size = 10                           # Number of deformations batched at the same time


import json
import random, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from SetupObject_1 import getDefaultTrainingSettings
from SimplicitHelpers import *
from PhysicsSim import PhysicsSimulator
from SimulatedScene import SimulatedScene
from trainer import Trainer

class SimplicitsSimulator:

    def __init__(self, sim_config: SimulationConfig):
        self.cfg = sim_config
        object_name = self.cfg.object_name
        training_name = self.training_name

        if self.cfg.new_training:
            # Step 1
            np_object = self.prepare_object_record()
            training_settings = self.prepare_training_settings()
            self._save_object_record(np_object, object_name)
            self._save_training_settings(training_settings, object_name, training_name)

            # Step 2
            Handles_post, Handles_pre = self.setup_model(np_object, training_settings)
            self._save_model(Handles_post, object_name, training_name, "losses")
            self._save_model(Handles_post, object_name, training_name, "Handles_post")
            self._save_model(Handles_pre, object_name, training_name, "Handles_pre")
        else:
            np_object = self._load_object_record(object_name)
            training_settings = self._load_training_settings(object_name, training_name)
            Handles_post = self._load_model(object_name, training_name, "Handles_post")
            Handles_pre = self._load_model(object_name, training_name, "Handles_pre")

        self.np_object, self.training_settings = np_object, training_settings
        self.Handles_post, self.Handles_pre = Handles_post, Handles_pre

        self.scene = None
        self.physics_simulator = None
        self.simulation_states = None

        # TODO (@operel): (Should print architecture?)
        # self.print_named_params(Handles_post)
        # self.print_architecture(Handles_pre)

        # t_O = torch.tensor(np_object["ObjectSamplePts"][:, 0:3]).to(device)
        # np_W0, np_X0, np_G0 = test(Handles_post, t_O, int(t_O.shape[0] / 10))
        # TODO (@operel): (visualize initial handles)
        # plot_handle_regions(np_X0, np_W0, "Pre Training Handle Weights")
        # TODO (@operel): (visualize initial YM - uniform for non-biological objects)
        # plot_implicit(np_object["ObjectSamplePts"], np_object["ObjectYMs"] )

    # Step 1a
    def prepare_object_record(self) -> Dict:
        """
        segment: If specified, runs simulation on isolated segment, else uses all segments
        """
        model = self.cfg.model
        if isinstance(self.cfg.segments, Segment):
            segment = self.cfg.segments
            simulated_mask = segment.mask
            N = simulated_mask.sum()
            gs_pos = model.get_xyz[simulated_mask]
            gs_rgb = model._features_dc[simulated_mask].squeeze(1)
            gs_YMs = torch.full(N, segment.attributes["Young's Modulus"])
            gs_PRs = torch.full(N, segment.attributes["Poisson Ratio"])
            gs_Rho = torch.full(N, segment.attributes["Rho"])
        else: # Dict of segements, could be only some of the segments!
            gs_pos = model.get_xyz
            N = gs_pos.shape[0]
            simulated_mask = torch.zeros(N, dtype=bool, device=gs_pos.device)
            for seg in self.cfg.segments.values():
                simulated_mask |= seg.mask
            gs_pos = model.get_xyz[simulated_mask]
            gs_rgb = model._features_dc[simulated_mask].squeeze(1)
            gs_YMs, gs_PRs, gs_Rho = torch.ones(N), torch.ones(N), torch.ones(N)
            for seg in self.cfg.segments.values():
                gs_YMs[seg.mask] = seg.attributes["Young's Modulus"]
                gs_PRs[seg.mask] = seg.attributes["Poisson Ratio"]
                gs_Rho[seg.mask] = seg.attributes["Rho"]
            simulated_mask = simulated_mask.cpu()
            gs_YMs = gs_YMs[simulated_mask]
            gs_PRs = gs_PRs[simulated_mask]
            gs_Rho = gs_Rho[simulated_mask]
        # else:  # Full dictionary of all segments
            # gs_pos = model.get_xyz
            # gs_rgb = model._features_dc.squeeze(1)
            # N = gs_pos.shape[0]
            # gs_YMs, gs_PRs, gs_Rho = torch.ones(N), torch.ones(N), torch.ones(N)
            # for seg in self.cfg.segments.values():
            #     gs_YMs[seg.mask] = seg.attributes["Young's Modulus"]
            #     gs_PRs[seg.mask] = seg.attributes["Poisson Ratio"]
            #     gs_Rho[seg.mask] = seg.attributes["Rho"]

        np_object = {
            "Name": self.cfg.object_name,
            "Dim": 3,
            "BoundingBoxSamplePts": None,
            "BoundingBoxSignedDists": None,
            "ObjectSamplePts": gs_pos.cpu().detach().numpy(),
            "ObjectSampleColors": gs_rgb.cpu().detach().numpy(),
            "ObjectYMs": gs_YMs.cpu().detach().numpy(),
            "ObjectPRs": gs_PRs.cpu().detach().numpy(),
            "ObjectRho": gs_Rho.cpu().detach().numpy(),
            "ObjectColors": None,
            "ObjectVol": 1,
            "SurfV": None,
            "SurfF": None,
            "MarchingCubesRes": -1,
            "SimulatedMask": simulated_mask.cpu().detach().numpy()  # Which gaussians of the model were simulated
        }
        return np_object

    # Step 1b
    def prepare_training_settings(self):
        training_dict = getDefaultTrainingSettings()
        training_dict["NumHandles"] = self.cfg.training_num_skinning_handles
        training_dict["NumTrainingSteps"] = self.cfg.training_num_steps
        training_dict["NumSamplePts"] = self.cfg.training_num_sample_pts
        training_dict["LRStart"] = self.cfg.training_LRStart
        training_dict["LREnd"] = self.cfg.training_LREnd
        training_dict["TSamplingStdev"] = self.cfg.training_TSamplingStdev
        return training_dict

    def setup_model(self, np_object: Dict = None, training_settings: Dict = None):
        Handles_pre = HandleModels(
            num_handles=training_settings["NumHandles"],
            num_layers=training_settings["NumLayers"],
            layer_width=training_settings["LayerWidth"],
            activation_func=training_settings["ActivationFunc"],
            dim=np_object["Dim"],
            lr_start=training_settings["LRStart"]
        )
        Handles_post = HandleModels(
            num_handles=training_settings["NumHandles"],
            num_layers=training_settings["NumLayers"],
            layer_width=training_settings["LayerWidth"],
            activation_func=training_settings["ActivationFunc"],
            dim=np_object["Dim"],
            lr_start=training_settings["LRStart"]
        )
        Handles_post.to_device(self.cfg.device)
        Handles_pre.to_device(self.cfg.device)
        Handles_pre.eval()
        return Handles_post, Handles_pre

    def run_training(self):
        # Step 3
        print("Start Training")
        self._run_training(self.np_object, self.training_settings, self.Handles_post, self.Handles_pre)
        print("Training complete: Handles_post weights are now initialized.")

    def _run_training(self, np_object: Dict, training_settings: Dict, Handles_post, Handles_pre):
        device = self.cfg.device
        object_name = self.cfg.object_name
        training_name = self.training_name

        trainer = Trainer(object_name, training_name, np_object, training_settings, Handles_post, Handles_pre)
        name_and_training_dir = object_name + "/" + training_name + "-training"

        STARTCUDATIME = torch.cuda.Event(enable_timing=True)
        ENDCUDATIME = torch.cuda.Event(enable_timing=True)
        losses = []
        timings = []
        clock = []

        for e in range(1, trainer.TOTAL_TRAINING_STEPS):
            num_handles = Handles_post.num_handles
            batch_size = int(training_settings["TBatchSize"])
            batchTs = getBatchOfTs(num_handles, batch_size, e).to(device).float()
            t_batchTs = batchTs * float(training_settings["TSamplingStdev"])
            STARTCLOCKTIME = time.time()
            STARTCUDATIME.record()
            l1, l2 = trainer.train_step(
                Handles_post,
                trainer.t_O,
                trainer.t_YMs,
                trainer.t_PRs,
                trainer.loss_fcn,
                t_batchTs,
                e
            )
            ENDCUDATIME.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()    # TODO (operel): remove this
            ENDCLOCKTIME = time.time()

            timings.append(STARTCUDATIME.elapsed_time(ENDCUDATIME))  # milliseconds
            clock.append(ENDCLOCKTIME - STARTCLOCKTIME)

            if e % 100 == 0:
                print("Step: ", e, "Loss:", l1 + l2, " > l1: ", l1, " > l2: ", l2)
            losses.append(np.array([l1 + l2, l1, l2]))

            if e % int(training_settings["SaveHandleIts"]) == 0:
                # Compute the moving average

                # save loss and handle state at current its
                torch.save(clock, name_and_training_dir + "/clocktimes-its-" + str(e))
                torch.save(timings, name_and_training_dir + "/timings-its-" + str(e))
                torch.save(losses, name_and_training_dir + "/losses-its-" + str(e))
                torch.save(Handles_post, name_and_training_dir + "/Handles_post-its-" + str(e))

                torch.save(clock, name_and_training_dir + "/clocktimes")
                torch.save(timings, name_and_training_dir + "/timings")
                torch.save(losses, name_and_training_dir + "/losses")
                torch.save(Handles_post, name_and_training_dir + "/Handles_post")

            if e % int(training_settings["SaveSampleIts"]) == 0:
                O = torch.tensor(trainer.np_object["ObjectSamplePts"], dtype=torch.float32, device=device)
                for b in range(batchTs.shape[0]):
                    Ts = batchTs[b, :, :, :]
                    O_new = trainer.getX(Ts, O, Handles_post)
                    write_ply(name_and_training_dir + "/training-epoch-" + str(e) + "-batch-" + str(b) + ".ply", O_new)

        torch.save(clock, name_and_training_dir + "/clocktimes")
        torch.save(timings, name_and_training_dir + "/timings")
        torch.save(losses, name_and_training_dir + "/losses")
        torch.save(Handles_post, name_and_training_dir + "/Handles_post")
        torch.save(Handles_pre, name_and_training_dir + "/Handles_pre")

    def set_scene(self, scene: SimulatedScene):
        self.scene = scene

    def load_scene(self, scene_name: str):
        self.scene = SimulatedScene.from_json(scene_name)

    def simulate_scene(self):
        scene = self.scene
        np_object = self.np_object
        name_and_training_dir = self.cfg.object_name + "/" + self.training_name + "-training"

        self.physics_simulator = PhysicsSimulator(
            scene=scene, np_object=self.np_object, name_and_training_dir=name_and_training_dir
        )

        # timestep
        random_batch_indices = np.random.randint(0, np_object["ObjectSamplePts"].shape[0],
                                                 size=int(scene.NumCubaturePts))

        np_X = np_object["ObjectSamplePts"][:, 0:3][random_batch_indices, :]
        np_YMs = np_object["ObjectYMs"][random_batch_indices, np.newaxis]
        np_Rho = np_object["ObjectRho"][random_batch_indices, np.newaxis]
        np_PRs = np_object["ObjectPRs"][random_batch_indices, np.newaxis]
        fixed_bc = self.physics_simulator.set_fixed_bc_mask[random_batch_indices]
        moving_bc = self.physics_simulator.set_fixed_bc_mask[random_batch_indices]

        scene_name = self.scene.Name
        torch.save(np_X, name_and_training_dir + "/" + scene_name + "-sim_X0")
        torch.save(np_YMs, name_and_training_dir + "/" + scene_name + "-sim_W")

        states, t_X0, explicit_W = self.physics_simulator.simulate(np_X, np_YMs, np_PRs, np_Rho,
                                                                   self.physics_simulator.E_pot, self.Handles_post,
                                                                   fixed_bc, moving_bc)

        states_path = name_and_training_dir + "/" + scene_name + "-sim_states"
        torch.save(states, states_path)
        torch.save(t_X0, name_and_training_dir + "/" + scene_name + "-sim_X0")
        torch.save(explicit_W, name_and_training_dir + "/" + scene_name + "-sim_W")
        self.simulation_states = states
        print(f"Done simulating scene. Output saved to: {states_path}")
        return self.animate_model()

    def load_simulated_scene(self, scene_name=None):
        if scene_name is None:
            scene_name = self.scene.Name
        device = self.cfg.device
        object_name = self.cfg.object_name
        training_name = self.training_name
        filename = object_name + "/" + training_name + "-training" + "/" + f"{scene_name}-sim_states"
        states = torch.load(filename,map_location=torch.device(device))
        self.simulation_states = states
        print(f"Loaded simulated scene from {filename}")
        return self.animate_model()

    def animate_model(self) -> TransformableGaussianModel:

        model = self.cfg.model
        sh_deg = self.cfg.model.max_sh_degree
        transformable_model = TransformableGaussianModel(
            handles_model=self.Handles_post,
            sh_degree=sh_deg,
            animated_mask=self.np_object['SimulatedMask']
        )
        transformable_model._xyz = model._xyz
        transformable_model._features_dc = model._features_dc
        transformable_model._features_rest = model._features_rest
        transformable_model._opacity = model._opacity
        transformable_model._scaling = model._scaling
        transformable_model._rotation = model._rotation
        transformable_model.active_sh_degree = sh_deg
        transformable_model.init_weights()

        device = self.cfg.device
        # states: List at length (num_frames); each entry is a flattened tensor of (num_handles, 3, 4)
        states = self.simulation_states
        num_frames = len(states)
        transforms = torch.stack(states).reshape(num_frames, -1, 3, 4)  # (num_frames, num_handles+1, 3, 4)
        transformable_model.transforms = transforms.to(device)  # (num_frames, num_handles+1, 3, 4)

        transformable_model.x0 = transformable_model._xyz.to('cpu')

        # computed_W_X0: tensor of (num_gaussians, num_handles + 1), representing the weight each handle assigns to each point at rest frame
        # X0 = torch.tensor(self.np_object["ObjectSamplePts"], dtype=torch.float32)
        # BATCH_SIZE = 100_000
        # chunk_weights = []
        # for idx, chunk in enumerate(X0.split(BATCH_SIZE)):
        #     chunk_w = torch.cat((self.Handles_post.getAllWeightsSoftmax(chunk), torch.ones(chunk.shape[0], 1)), dim=1)
        #     chunk_weights.append(chunk_w.cpu())
        # weights = torch.cat(chunk_weights, dim=0)
        # transformable_model.weights = weights.to(device)
        return transformable_model

    @property
    def training_name(self):
        segment_name = f"-{self.cfg.segments.name}" if isinstance(self.cfg.segments, Segment) else ""
        training_id = f"-{self.cfg.training_id}" if len(self.cfg.training_id) > 0 else ""
        training_name = f"{self.cfg.object_name}-training{segment_name}{training_id}"
        return training_name

    @staticmethod
    def _save_object_record(np_object, object_name):
        if not os.path.exists(object_name):
            os.makedirs(object_name)
        torch.save(np_object, object_name + "/" + object_name + "-" + "object")
        print(f'Object saved to: {os.getcwd() + "/" + object_name + "/" + object_name + "-" + "object"}')

    @staticmethod
    def _load_object_record(object_name):
        np_object = torch.load(f"{object_name}/{object_name}-object")
        return np_object

    @staticmethod
    def _save_training_settings(training_settings, object_name, training_name):
        training_folder = object_name + "/" + training_name + "-training"
        if not os.path.exists(training_folder):
            os.makedirs(training_folder)
        with open(f"{training_folder}/training-settings.json", 'w', encoding='utf-8') as f:
            json.dump(training_settings, f, ensure_ascii=False, indent=4)
        print(f'Training settings saved to: {os.getcwd() + "/" + training_folder + "/training-settings.json"}')

    @staticmethod
    def _load_training_settings(object_name, training_name):
        training_folder = object_name + "/" + training_name + "-training"
        # Open JSON file with training settings
        with open(f"{training_folder}/training-settings.json", 'r') as openfile:
            training_settings = json.load(openfile)
        return training_settings

    @staticmethod
    def _save_model(model, object_name, training_name, filename):
        torch.save(model, object_name + "/" + training_name + "-training" + "/" + filename)
        print(f'{filename} saved to: '
              f'{os.getcwd() + "/" + object_name + "/" + training_name + "-training" + "/" + filename}')

    @staticmethod
    def _load_model(object_name, training_name, filename):
        model = torch.load(object_name + "/" + training_name + "-training" + "/" + filename)
        return model

    @staticmethod
    def print_named_params(handles_post):
        print('--List of learned params:--')
        for nnnn, pppp in handles_post.model.named_parameters():
            print(nnnn, pppp.size())

    @staticmethod
    def print_architecture(handles_pre):
        print('--Architecture:--')
        print(handles_pre.__dict__)

    # @staticmethod
    # def _save_training_settings(training_dict, sim_obj_name):
    #     if not os.path.exists(sim_obj_name):
    #         os.makedirs(sim_obj_name)
    #     json_object = json.dumps(training_dict, indent=4)
    #     with open(sim_obj_name + "/" + sim_obj_name + "-training-settings.json", "w") as outfile:
    #         outfile.write(json_object)
    #     print(f'Training settings saved to: {os.getcwd() + "/" + sim_obj_name + "/" + sim_obj_name + "-training-settings.json"}')

    # @staticmethod
    # def _load_training_settings(object_name):
    #     # Open JSON file with training settings
    #     with open(f"{object_name}/{object_name}-training-settings.json", 'r') as openfile:
    #         training_settings = json.load(openfile)
    #     return training_settings