import os
import sys
simplicits_working_dir = os.getcwd()
GAUSSIAN_SPLATS_REPO = f'{simplicits_working_dir}/thirdparty/gausplat'
sys.path.append(GAUSSIAN_SPLATS_REPO) # need to make gaussplat works like a module
from gaussian_splat_utils import load_checkpoint, try_load_camera
from ps_gui import GUI, BLENDER_CONFIG, DEFAULT_CONFIG

# Required installations:
# python -m pip install git+ssh://git@github.com/nmwsharp/polyscope-py.git@v2
# pip install cuda-python cupy

FICUS = 'db198a06-1'
GARDEN = '132dbdde-8'

config = DEFAULT_CONFIG
config.model_path = f'{GAUSSIAN_SPLATS_REPO}/output/{GARDEN}'
config.object_name = "mip360-garden"

gaussians = load_checkpoint(config.model_path)
initial_camera = try_load_camera(config.model_path)

gui = GUI(conf=config, model=gaussians, initial_camera=initial_camera, scene_bbox=None, dataset=None)

while True:
    gui.update_render_view_viz()


# camera = try_load_kaolin_camera(OUTPUT_FOLDER)
# renderer = GaussianSplatsRendererSetup(gaussians, camera)
# renderer.display()