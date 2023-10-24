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
from SimplicitHelpers import *
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


loaded_Handles = torch.load(object_name+"/"+training_name+"-training" + "/Handles_post",  map_location=torch.device(device))

for nnnn, pppp in loaded_Handles.model.named_parameters():
    print(nnnn, pppp.size())

loaded_X0 = torch.tensor(np_object["ObjectSamplePts"],  dtype=torch.float32)
# loaded_X0 = torch.load(object_name+"/"+training_name+"-training" +"/"+str(args[2])+"-sim_X0", map_location=torch.device(device))
loaded_ym = torch.tensor(np_object["ObjectYMs"]).detach().numpy()
# loaded_ym = torch.ones(loaded_X0.shape[0])
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


ps_cloud = ps.register_point_cloud("samples", loaded_X0.cpu().numpy(), enabled=True)
ps_cloud.add_scalar_quantity("yms", loaded_ym, enabled=True)


if (len(scene["CollisionObjects"])>0):
    pokes = np.zeros((len(scene["CollisionObjects"]), 3))
    rad = scene["CollisionObjects"][0]["Radius"]
    for p in range(len(scene["CollisionObjects"])):
        collision_object = torch.tensor(scene["CollisionObjects"][p]["Position"], dtype=torch.float32, device=device)
        pokes[p,:] = collision_object
    ps_pokes = ps.register_point_cloud("Pokes", pokes, radius=rad, enabled=True)


computed_W_X0 =  torch.cat((loaded_Handles.getAllWeightsSoftmax(loaded_X0), torch.ones(loaded_X0.shape[0], 1)), dim=1) 

def callback():
   global frame_num, loaded_states, computed_W, loaded_all, ps_cloud, ps_surf#, ps_vol#, ps_surf
   changed, frame_num = psim.SliderInt("Frame", frame_num, v_min=0, v_max=len(loaded_states)-1)
   if changed:
      Ts = loaded_states[frame_num].reshape(-1, 3,4)
      X = getX(Ts, loaded_X0, computed_W_X0)
      ps_cloud.update_point_positions(X)

      if (len(scene["CollisionObjects"])>0):
        for p in range(len(scene["CollisionObjects"])):
                collision_object = np.array(scene["CollisionObjects"][p]["Position"])
                l_dict = {"collision_obj" : collision_object, "dt" : 0.1, "simulation_iteration" : frame_num}
                exec(scene["CollisionObjects"][p]["Update_code"], None, l_dict)
                collision_object = l_dict["pos"]
                pokes[p,:] = collision_object
        ps_pokes.update_point_positions(pokes)


ps.set_user_callback(callback)
ps.show()
