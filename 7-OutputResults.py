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
from SimplicitHelpers import *
import matplotlib.colors as mcolors
import potpourri3d as pp3d
import os
import skimage
from plyfile import PlyData, PlyElement

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
loaded_ym = torch.tensor(np_object["ObjectYMs"]).detach().numpy()
loaded_states = torch.load(object_name+"/"+training_name+"-training" +"/"+str(args[2])+"-sim_states", map_location=torch.device(device))

# all_O = torch.tensor(np_object["ObjectSamplePts"],  dtype=torch.float32)
# midpt_x = (torch.min(all_O[:,0]) + torch.max(all_O[:,0]))/2.0
# np_object["ObjectSamplePts"] = all_O[all_O[:,0]>midpt_x,:].cpu().detach().numpy()

# loaded_YM = torch.tensor(np_object["ObjectYMs"])
# partial_YM = loaded_YM[all_O[:,0]>midpt_x]
# np_object["ObjectYMs"] = partial_YM.cpu().detach().numpy()



def scalar_to_rgb(S_normalized, mincolorstr, maxcolorstr):
    cmap = mcolors.LinearSegmentedColormap.from_list("", [mincolorstr, maxcolorstr])

    if(np.sum(S_normalized)==0):
        return cmap(S_normalized)[:, 0:3]*255
    else:
        S_normalized = (S_normalized - np.min(S_normalized)) / (np.max(S_normalized) - np.min(S_normalized))
        colors_rgb = cmap(S_normalized)[:, 0:3]*255
        return colors_rgb

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

def getE(X0, YMs, PRs, Ts, Handles):
    poisson = PRs
    mus = YMs/(2*(1+poisson)) #shead modulus
    lams = YMs*poisson/((1+poisson)*(1-2*poisson)) #

    def elastic_energy(F, mu, lam):
        E = neohookean_E2(mu, lam, F[0,:,:])
        # E = linear_elastic_E(mu, lam, F[0,:,:])
        return E
    
    def x(x0):
        x0_i = x0.unsqueeze(0)
        x03 = torch.cat((x0_i, torch.tensor([[1]], device=device)), dim=1)
        t_W = torch.cat((Handles.getAllWeightsSoftmax(x0_i), torch.tensor([[1]]).to(device)), dim=1).T
        
        def inner_over_handles(T, w):
            return w*T@x03.T

        wTx03s = torch.vmap(inner_over_handles)(Ts, t_W)
        x_i =  torch.sum(wTx03s, dim=0)
        return x_i.T +x0_i
    
    pt_wise_Fs = torch.vmap(torch.func.jacrev(x), randomness="same")(X0)
    pt_wise_E = torch.vmap(elastic_energy, randomness="same")(pt_wise_Fs, mus, lams)
    return pt_wise_E

def getSurfacePts(Pts, SDs):
    sample_pts = Pts#[random_batch_indices, :]
    sample_sds = SDs#[random_batch_indices]
    voxel_grid, voxel_pts, voxel_sdf = interpolate_point_cloud(sample_pts, 0.05, sample_sds)
    reconstructed_V, reconstructed_F, normals, other = skimage.measure.marching_cubes(voxel_grid, level=0)
    interior_voxel_pts = voxel_pts[np.nonzero(voxel_sdf<=0)[0], :]
    loaded_V = rescale_and_recenter(reconstructed_V, np.min(interior_voxel_pts, axis=0), np.max(interior_voxel_pts, axis=0))
    return loaded_V, reconstructed_F

# 0 = output sample points in ply (colored by YM white to black) (default)
if "0" in args[3]: 
    write_samples_folder = "../Results/"+object_name+"-"+training_name+"-"+scene_name+"/samples/"
    if not os.path.exists(write_samples_folder):
        os.makedirs(write_samples_folder)

    loaded_O = torch.tensor(np_object["ObjectSamplePts"],  dtype=torch.float32)
    loaded_ym = torch.tensor(np_object["ObjectYMs"]).detach().numpy()

    num_samples = loaded_O.shape[0]  
    if num_samples > 10000:
        num_samples = 10000
    random_batch_indices = torch.randint(low=0, high= loaded_O.shape[0], size=(num_samples,))

    loaded_O = loaded_O[random_batch_indices]
    loaded_ym = loaded_ym[random_batch_indices]

    compute_W_O = torch.cat((loaded_Handles.getAllWeightsSoftmax(loaded_O), torch.ones(loaded_O.shape[0], 1)), dim=1) 

    rgb_values = scalar_to_rgb(loaded_ym, "white", "black")
    for i in range(len(loaded_states)):
        print(i)
        Ts = loaded_states[i].reshape(-1, 3,4).to(device)
        X = getX(Ts, loaded_O, compute_W_O)
        write_ply(write_samples_folder+object_name+"-"+training_name+"-"+scene_name+"-samplepts-"+str(i)+".ply", X.detach().numpy(), rgb_values, None)
        
# 1 = output push forward verts in ply
if "1" in args[3]: 
    write_mesh_folder = "../Results/"+object_name+"-"+training_name+"-"+scene_name+"/mesh/"
    if not os.path.exists(write_mesh_folder):
        os.makedirs(write_mesh_folder)

    V0 = torch.tensor(np_object["SurfV"], dtype=torch.float32)
    loaded_F = np_object["SurfF"]
    computed_W_V = torch.cat((loaded_Handles.getAllWeightsSoftmax(V0), torch.ones(V0.shape[0], 1)), dim=1)
    loaded_ym = np_object["SurfYM"]
    if loaded_ym == None:
        loaded_ym = np.ones(np_object["surfV"].shape[0])
    rgb_values = scalar_to_rgb(loaded_ym, "white", "black")
    for i in range(len(loaded_states)):
        print(i)
        Ts = loaded_states[i].reshape(-1, 3,4).to(device)
        V = getX(Ts, V0, computed_W_V)
        write_ply(write_mesh_folder+object_name+"-"+training_name+"-"+scene_name+"-surface-"+str(i)+".ply", V.detach().numpy(), rgb_values, loaded_F)

# 2 = output reconstruction in obj
if "2" in args[3]: 
    write_reconstruction_folder = "../Results/"+object_name+"-"+training_name+"-"+scene_name+"/reconstruction/"
    if not os.path.exists(write_reconstruction_folder):
        os.makedirs(write_reconstruction_folder)
    
    loaded_uniformbb = torch.tensor(np_object["BoundingBoxSamplePts"], dtype=torch.float32)
    loaded_uniformbbsdf = np_object["BoundingBoxSignedDists"]
    computed_W_bb = torch.cat((loaded_Handles.getAllWeightsSoftmax(loaded_uniformbb), torch.ones(loaded_uniformbb.shape[0], 1)), dim=1)

    for i in range(len(loaded_states)):
        Ts = loaded_states[i].reshape(-1, 3,4)
        updatedbb = getX(Ts, loaded_uniformbb, computed_W_bb)
        t_V, t_F = getSurfacePts(updatedbb.detach().numpy()[0::50,:], loaded_uniformbbsdf[0::50])
        write_ply(write_reconstruction_folder+object_name+"-"+training_name+"-"+scene_name+"-reconstruction-"+str(i)+".ply", t_V, None, t_F)

# 3 = output sdf in ply (colored by coolwarm)
if "3" in args[3]: 
    write_sdf_folder = "../Results/"+object_name+"-"+training_name+"-"+scene_name+"/sdf/"
    if not os.path.exists(write_sdf_folder):
        os.makedirs(write_sdf_folder)
    loaded_uniformbb = torch.tensor(np_object["BoundingBoxSamplePts"], dtype=torch.float32)
    loaded_uniformbbsdf = np_object["BoundingBoxSignedDists"]
    computed_W_bb = torch.cat((loaded_Handles.getAllWeightsSoftmax(loaded_uniformbb), torch.ones(loaded_uniformbb.shape[0], 1)), dim=1)

    sdfrgb = scalar_to_rgb(loaded_uniformbbsdf, "blue", "red")

    for i in range(len(loaded_states)):
        print(i)
        Ts = loaded_states[i].reshape(-1, 3,4)
        updatedbb = getX(Ts, loaded_uniformbb, computed_W_bb)
        write_ply(write_sdf_folder+object_name+"-"+training_name+"-"+scene_name+"-sdf-"+str(i)+".ply", updatedbb.detach().numpy()[0::5,:], sdfrgb[0::5], None)

# 4 = output sample pt energies (cool warm)
if "4" in args[3]: 
    write_energies_folder = "../Results/"+object_name+"-"+training_name+"-"+scene_name+"/energies/"
    if not os.path.exists(write_energies_folder):
        os.makedirs(write_energies_folder)
    loaded_O = torch.tensor(np_object["ObjectSamplePts"],  dtype=torch.float32)
    compute_W_O = torch.cat((loaded_Handles.getAllWeightsSoftmax(loaded_O), torch.ones(loaded_O.shape[0], 1)), dim=1) 
    loaded_ym = torch.tensor(np_object["ObjectYMs"], dtype=torch.float32)
    loaded_pr = torch.tensor(np_object["ObjectPRs"], dtype=torch.float32)
    E0s = getE(loaded_O, loaded_ym, loaded_states[0].reshape(-1, 3,4), loaded_Handles)
    Es = getE(loaded_O, loaded_ym, loaded_states[0].reshape(-1, 3,4), loaded_Handles) - E0s
    rgb_values = scalar_to_rgb(Es.detach().numpy(), "magenta", "yellow")
    for i in range(len(loaded_states)):
        print(i)
        Ts = loaded_states[i].reshape(-1, 3,4).to(device)
        X = getX(Ts, loaded_O, compute_W_O)
        Es = getE(loaded_O, loaded_ym, loaded_states[i].reshape(-1, 3,4), loaded_Handles) - E0s
        rgb_values = scalar_to_rgb(Es.detach().numpy(), "magenta", "yellow")
        write_ply(write_energies_folder+object_name+"-"+training_name+"-"+scene_name+"-energies-"+str(i)+".ply", X.detach().numpy(), rgb_values, None)

# 5 = output sample points in ply (colored by nerf colors) (default)
if "5" in args[3]:
    write_samples_folder = "../Results/"+object_name+"-"+training_name+"-"+scene_name+"/nerf/"
    if not os.path.exists(write_samples_folder):
        os.makedirs(write_samples_folder)
    loaded_O = torch.tensor(np_object["ObjectSamplePts"],  dtype=torch.float32)
    rgb_values = torch.tensor(np_object["ObjectSampleColors"]).detach().numpy()
    num_samples = loaded_O.shape[0]  
    if num_samples > 50000:
        num_samples = 50000
    random_batch_indices = torch.randint(low=0, high= loaded_O.shape[0], size=(num_samples,))

    loaded_O = loaded_O[random_batch_indices]
    rgb_values = rgb_values[random_batch_indices]


    compute_W_O = torch.cat((loaded_Handles.getAllWeightsSoftmax(loaded_O), torch.ones(loaded_O.shape[0], 1)), dim=1) 
    

    print(loaded_O.shape, rgb_values.shape)
    for i in range(len(loaded_states)):
        print(i)
        Ts = loaded_states[i].reshape(-1, 3,4).to(device)
        X = getX(Ts, loaded_O, compute_W_O)
        write_ply(write_samples_folder+object_name+"-"+training_name+"-"+scene_name+"-samplepts-"+str(i)+".ply", X.detach().numpy(), rgb_values, None)
        