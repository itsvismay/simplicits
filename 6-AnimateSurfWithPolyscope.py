import igl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from functorch import vmap
import random, math, sys, json
import matplotlib.tri as tri
import polyscope as ps
import polyscope.imgui as psim
#import skimage 
from SimplicitHelpers import *
import skimage 
import matplotlib.colors as mcolors
import potpourri3d as pp3d
import os


device = "cpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

#Read in the object (hardcoded for now)
args = sys.argv[1:]
object_name = str(args[0])
training_name = str(args[1])
scene_name = str(args[2])
name_and_training_dir = object_name+"/"+training_name+"-training"
fname = str(args[0])+"/"+str(args[0])

# Opening JSON file with training settings
with open(fname+"-training-settings.json", 'r') as openfile:
    training_settings = json.load(openfile)
np_object = torch.load(fname+"-object")
scene = json.loads(open(name_and_training_dir + "/../"+str(args[2])+".json", "r").read())

use_handle_its = scene["HandleIts"] if "HandleIts" in scene else ""
loaded_Handles = torch.load(object_name+"/"+training_name+"-training" + "/Handles_post"+use_handle_its,  map_location=torch.device(device))

for nnnn, pppp in loaded_Handles.model.named_parameters():
    print(nnnn, pppp.size())

loaded_X0 = torch.tensor(np_object["ObjectSamplePts"],  dtype=torch.float32)
loaded_ym = torch.tensor(np_object["ObjectYMs"]).detach().numpy()
loaded_states = torch.load(object_name+"/"+training_name+"-training" +"/"+str(args[2])+"-sim_states", map_location=torch.device(device))



showPointcloud = True
showUniformSample = True 
showReconstruction = False

def getX(Ts, l_X0, l_W):
    def x(x0, tW):
        x0_i = x0.unsqueeze(0)
        x03 = torch.cat((x0_i, torch.tensor([[1]], device=device)), dim=1)
        Ws = tW.T
        
        def inner_over_handles(T, w):
            return w*T@x03.T

        wTx03s = torch.vmap(inner_over_handles)(Ts.to(device), Ws)
        x_i =  torch.sum(wTx03s, dim=0)
        return x_i.T + x0_i
    
    X = torch.vmap(x, randomness="same")(l_X0.to(device), l_W.to(device))
    return X[:,0,:].cpu().detach().numpy()


frame_num = 0
ps.init()

#set bounding box for floor plane and scene extents
ps.set_automatically_compute_scene_extents(False)
ps.set_length_scale(1.)
low = np.array((float(scene["BoundingX"][0]), 
                float(scene["BoundingY"][0]), 
                float(scene["BoundingZ"][0]))) 

high = np.array((float(scene["BoundingX"][1]), 
                 float(scene["BoundingY"][1]), 
                 float(scene["BoundingZ"][1])))
ps.set_bounding_box(low, high)
ps.set_ground_plane_height_factor(-float(scene["Floor"]) + float(scene["BoundingY"][0]), is_relative=False) # adjust the plane height



V0 = torch.tensor(np_object["SurfV"], dtype=torch.float32)
loaded_F = np_object["SurfF"]
computed_W_V = torch.cat((loaded_Handles.getAllWeightsSoftmax(V0), torch.ones(V0.shape[0], 1)), dim=1)

ps_surf = ps.register_surface_mesh("samples", V0.cpu().numpy(), loaded_F, enabled=True)

for p in range(len(scene["CollisionObjects"])):
    poky = np.array([scene["CollisionObjects"][p]["Position"]],   dtype=np.float64) 
    ps_poky = ps.register_point_cloud("poky", points = poky, radius=0.5)

def callback():
   global frame_num, loaded_states, computed_W, loaded_all, ps_cloud, ps_surf#, ps_vol#, ps_surf
   changed, frame_num = psim.SliderInt("Frame", frame_num, v_min=0, v_max=len(loaded_states)-1)
   if changed:
      Ts = loaded_states[frame_num].reshape(-1, 3,4)
      V = getX(Ts, V0, computed_W_V)
      ps_surf.update_vertex_positions(V)
      for p in range(len(scene["CollisionObjects"])):
        poky_new = np.copy(poky)
        poky_new[0,1] -= 0.1*frame_num # np.cos(0.05*float(frame_num))
        print(frame_num)
        ps_poky.update_point_positions(poky_new)

ps.set_user_callback(callback)
ps.show()
