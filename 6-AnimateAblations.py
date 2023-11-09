from enum import auto
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

#Read in the object (hardcoded for now)
args = sys.argv[1:]
object_name = str(args[0])

# DO THE TRAINING
training_list = [name.split("-training")[0] for name in os.listdir(object_name) if name.startswith("ablation_") and len(name.split("_"))<3]
print(training_list)
def getObjectSimulation(object_name, training_name, scene_name):
    #Read in the object (hardcoded for now)
    name_and_training_dir = object_name+"/"+training_name+"-training"
    fname = object_name +"/"+object_name

    # Opening JSON file with training settings
    with open(name_and_training_dir+"/training-settings.json", 'r') as openfile:
        training_settings = json.load(openfile)
    np_object = torch.load(fname+"-object")
    scene = json.loads(open(name_and_training_dir + "/../"+scene_name+".json", "r").read())


    loaded_Handles = torch.load(object_name+"/"+training_name+"-training" + "/Handles_post",  map_location=torch.device(device))

    loaded_X0 = torch.tensor(np_object["ObjectSamplePts"],  dtype=torch.float32)
    # loaded_X0 = torch.load(object_name+"/"+training_name+"-training" +"/"+str(args[2])+"-sim_X0", map_location=torch.device(device))
    loaded_ym = torch.tensor(np_object["ObjectYMs"]).detach().numpy()

    V0 = torch.tensor(np_object["SurfV"], dtype=torch.float32)
    F = np_object["SurfF"]

    # loaded_ym = torch.ones(loaded_X0.shape[0])
    loaded_states = torch.load(object_name+"/"+training_name+"-training" +"/"+scene_name+"-sim_states", map_location=torch.device(device))
    computed_W_X0 =  torch.cat((loaded_Handles.getAllWeightsSoftmax(V0), torch.ones(V0.shape[0], 1)), dim=1) 
    

    return V0, F, loaded_ym, loaded_states, computed_W_X0, scene

training_id = 0
frame_num = 0

V0, F, YM, STATES, WX0, scene = getObjectSimulation(object_name, training_list[0], "twist")

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




ps_surf = ps.register_surface_mesh("samples", V0.cpu().numpy(), F, enabled=True)


if (len(scene["CollisionObjects"])>0):
    pokes = np.zeros((len(scene["CollisionObjects"]), 3))
    rad = scene["CollisionObjects"][0]["Radius"]
    for p in range(len(scene["CollisionObjects"])):
        collision_object = torch.tensor(scene["CollisionObjects"][p]["Position"], dtype=torch.float32, device=device)
        pokes[p,:] = collision_object
    ps_pokes = ps.register_point_cloud("Pokes", pokes, radius=rad, enabled=True)

autoplay = False 

def callback():
    global frame_num, training_id, autoplay, V0, YM, WX0, STATES, scene, ps_surf

    if(psim.Button("Autoplay Toggle")):
        # This code is executed when the button is pressed
        autoplay = not autoplay
        print(autoplay)

    change_object1 = False
    change_framenum1 = False
    if autoplay:
        time.sleep(0.01)
        frame_num += 1
        if training_id == len(training_list):
            training_id = 0
            frame_num = 0
            change_object1 = True
            change_framenum1 = True
        
        if frame_num == len(STATES):
            frame_num = 0
            training_id += 1
            change_object1 = True
    
        change_framenum1 = True

    change_object2, training_id = psim.SliderInt("Training Index:", training_id, v_min=0, v_max=len(training_list)-1)
    change_framenum2, frame_num = psim.SliderInt("Frame", frame_num, v_min=0, v_max=len(STATES)-1)
    if change_object2 or change_object1:
        print(training_list[training_id])
    if change_object1 or change_object2:
        V0, F, YM, STATES, WX0, scene  = getObjectSimulation(object_name, training_list[training_id], "twist")
        change_framenum2 = True
        ps_surf = ps.register_surface_mesh("samples", V0.cpu().numpy(), F, enabled=True)

    if change_framenum1 or change_framenum2:
        Ts = STATES[frame_num].reshape(-1, 3,4)
        X = getX(Ts, V0, WX0)
        ps_surf.update_vertex_positions(X)

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
