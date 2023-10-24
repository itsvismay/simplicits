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


def scalar_to_rgb(S_normalized, mincolorstr, maxcolorstr):
    S_normalized = (S_normalized - np.min(S_normalized)) / (np.max(S_normalized) - np.min(S_normalized))
    cmap = mcolors.LinearSegmentedColormap.from_list("", [mincolorstr, maxcolorstr])
    colors_rgb = cmap(S_normalized)
    return colors_rgb

device = "cpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

#Read in the object (hardcoded for now)
args = sys.argv[1:]

name_object = str(args[0])
name_num_handles = str(args[1])
name_scene = str(args[2])
name = str(args[0])+"/"+str(args[0]) + "-" + str(args[1]) + "-"

scene = json.loads(open(name + "dir/../"+ str(args[2]) +".json", "r").read())

####################
np_W = igl.read_dmat(name+"dir/W.dmat")
np_X, np_T, np_F = igl.read_mesh(name + "dir/"+"MESH.mesh")
np_YM = 1e7*np.ones((np_X.shape[0],1))
np_PR = 0.45

# Compute the Euclidean distance from each point to the origin
# Find the indices of points within the specified radius
distances = np.linalg.norm(np_X, axis=1)
indices_within_radius = np.where(distances <= 0.2)[0]
np_YM[indices_within_radius,:] /= 1000
np_Vol = np.sum(igl.volume(np_X, np_T))
####################
computed_W_X0 =  torch.tensor(np_W) 
all_O = torch.tensor(np_X,  dtype=torch.float32)
midpt_x = (torch.min(all_O[:,0]) + torch.max(all_O[:,0]))/2.0
partial_O = all_O[all_O[:,0]>midpt_x,:]
partial_W = computed_W_X0[all_O[:,0]>midpt_x, :]


loaded_YM = torch.tensor(np_YM)
partial_YM = loaded_YM[all_O[:,0]>midpt_x]
S_normalized = partial_YM.cpu().detach().numpy()
ps_cloud_colors = scalar_to_rgb(S_normalized, "white", "black")

#mass matrix created from masses assuming uniform density over mesh
#total mass of object
density_m = float(scene["density"]) #kg/m^2
volume_m = np_Vol/all_O.shape[0]
m = density_m*volume_m

loaded_states = torch.load(name+"dir/"+str(args[2])+"-sim_states", map_location=torch.device(device))


showPointcloud = True
showUniformSample = True 
showReconstruction = False

def getX(Ts, X0, W):
    def x(x0, tW):
        x0_i = x0.unsqueeze(0)
        x03 = torch.cat((x0_i, torch.tensor([[1]], device=device)), dim=1)
        Ws = tW.T
        
        def inner_over_handles(T, w):
            return w*T@x03.T

        wTx03s = torch.vmap(inner_over_handles)(Ts, Ws)
        x_i =  torch.sum(wTx03s, dim=0)
        return x_i.T + x0_i
    
    X = torch.vmap(x, randomness="same")(X0, W)
    return X[:,0,:]


def getK(loaded_states, frame_num, m, X0, W):
    #KE = 0.5*mv^2
    #Impact = mv/2dt
    dt = float(scene["dt"])

    def force(xpp, xp, x):
        velp = ((xp - xpp)/dt)
        vel = ((x-xp)/dt)
        a = (vel-velp)/dt
        return torch.norm(m*a)
    
    if frame_num==0:
        Xp = getX(loaded_states[0].reshape(-1, 3, 4), X0, W)
        Xpp = getX(loaded_states[0].reshape(-1, 3, 4), X0, W) 
    elif frame_num ==1:
        Xp = getX(loaded_states[0].reshape(-1, 3, 4), X0, W)
        Xpp = getX(loaded_states[0].reshape(-1, 3, 4), X0, W)
    else:
        Xpp = getX(loaded_states[frame_num-2].reshape(-1, 3, 4), X0, W) 
        Xp = getX(loaded_states[frame_num-1].reshape(-1, 3, 4), X0, W)
    
    X = getX(loaded_states[frame_num].reshape(-1, 3, 4), X0, W)
    pt_wise_KE = torch.vmap(force, randomness="same")(Xpp, Xp, X)
    return pt_wise_KE

def getE(X0, W, YMs, Ts):
    poisson = float(0.45)
    mus = YMs/(2*(1+poisson)) #shead modulus
    lams = YMs*poisson/((1+poisson)*(1-2*poisson)) #

    def elastic_energy(F, mu, lam):
        E = neohookean_E2(mu, lam, F[0,:,:])
        # E = linear_elastic_E(mu, lam, F[0,:,:])
        return E
    
    def x(x0, wi):
        x0_i = x0.unsqueeze(0)
        x03 = torch.cat((x0_i, torch.tensor([[1]], device=device)), dim=1)
        t_W = wi.T
        
        def inner_over_handles(T, w):
            return w*T@x03.T

        wTx03s = torch.vmap(inner_over_handles)(Ts, t_W)
        x_i =  torch.sum(wTx03s, dim=0)
        return x_i.T +x0_i
    
    pt_wise_Fs = torch.vmap(torch.func.jacrev(x), randomness="same")(X0, W)
    pt_wise_E = torch.vmap(elastic_energy, randomness="same")(pt_wise_Fs, mus, lams)
    return pt_wise_E


frame_num = 0

ps.init()

#set bounding box for floor plane and scene extents
ps.set_automatically_compute_scene_extents(False)
ps.set_length_scale(1.)
low = np.array((float(scene["bounding_x"][0]), 
                float(scene["bounding_y"][0]), 
                float(scene["bounding_z"][0]))) 

high = np.array((float(scene["bounding_x"][1]), 
                 float(scene["bounding_y"][1]), 
                 float(scene["bounding_z"][1])))
ps.set_bounding_box(low, high)
ps.set_ground_plane_height_factor(-float(scene["floor"]) + float(scene["bounding_y"][0]), is_relative=False) # adjust the plane height


ps_cloud = ps.register_point_cloud("samples", partial_O.cpu().numpy(), enabled=True)
E0s = getE(partial_O, partial_W, partial_YM, loaded_states[0].reshape(-1, 3,4))
Es = getE(partial_O, partial_W, partial_YM, loaded_states[0].reshape(-1, 3,4)) - E0s
ps_cloud.add_scalar_quantity("E", Es.detach().numpy().flatten())
Ks = getK(loaded_states, frame_num, m, partial_O, partial_W)
ps_cloud.add_scalar_quantity("K", Ks.detach().numpy().flatten())



write_mesh_folder = "../Results/"+name_object+"/"+name_object+"-"+name_num_handles+"-"+name_scene+"/mesh/"
write_samples_folder = "../Results/"+name_object+"/"+name_object+"-"+name_num_handles+"-"+name_scene+"/samples/"
write_reconstruction_folder = "../Results/"+name_object+"/"+name_object+"-"+name_num_handles+"-"+name_scene+"/reconstruction/"
if not os.path.exists(write_mesh_folder):
    os.makedirs(write_mesh_folder)
if not os.path.exists(write_samples_folder):
    os.makedirs(write_samples_folder)
if not os.path.exists(write_reconstruction_folder):
    os.makedirs(write_reconstruction_folder)

# for i in range(len(loaded_states)):
#     print(i)
#     Ts = loaded_states[i].reshape(-1, 3,4)
#     # X = getX(Ts, partial_O, partial_W)
#     t_V = getX(Ts, t_V0, computed_surfW)
#     pp3d.write_mesh(t_V, loaded_F, write_mesh_folder + name_object+"-"+name_num_handles+"-"+name_scene+"-"+str(i)+".obj")
# #     # pp3d.write_point_cloud(X, write_samples_folder + name_object+"-"+name_num_handles+"-"+name_scene+"-"+str(i)+".obj") 

def callback():
   global frame_num, loaded_states, computed_W, loaded_all, ps_cloud, ps_surf#, ps_vol#, ps_surf
   changed, frame_num = psim.SliderInt("Frame", frame_num, v_min=0, v_max=len(loaded_states)-1)
   if changed:
      Ts = loaded_states[frame_num].reshape(-1, 3,4)
      X = getX(Ts.to(device), partial_O, partial_W)
      ps_cloud.update_point_positions(X.detach().numpy())
      Es = getE(partial_O, partial_W, partial_YM, Ts) - E0s
      ps_cloud.add_scalar_quantity("E", Es.detach().numpy().flatten(), vminmax=(0, 1e1))
    #   Ks = getK(loaded_states, frame_num, m, partial_O, partial_W)
    #   ps_cloud.add_scalar_quantity("K", Ks.detach().numpy())

      
ps.set_user_callback(callback)
ps.show()
