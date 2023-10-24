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

fname = str(args[0])+"/"+str(args[0])
np_object = torch.load(fname+"-object")


print(training_name)

# Opening JSON file with training settings
with open(fname+"-training-settings.json", 'r') as openfile:
    training_settings = json.load(openfile)

name = object_name+"/"+training_name+"-training" 

# Get a list of all files in the folder
all_files = os.listdir(name+"/")
# Filter files with the "training" prefix
training_files = [filename for filename in all_files if filename.startswith('training')]

epoch_list = sorted(list(set([int(f.split("epoch-")[1].split("-")[0]) for f in training_files])))
batch_list = sorted(list(set([int(f.split("batch-")[1].split(".")[0]) for f in training_files])))

global epoch_ind, batch_ind
epoch_ind = 0
batch_ind = 0

# Combo box to choose from options
# There, the options are a list of strings in `ui_options`,
# and the currently selected element is stored in `ui_options_selected`.
def callback():
    global epoch_ind, batch_ind
    changed1, epoch_ind = psim.SliderInt("Epoch:", epoch_ind, v_min=0, v_max=len(epoch_list)-1)
    changed2, batch_ind = psim.SliderInt("Batch:", batch_ind, v_min=0, v_max=len(batch_list)-1)
    if changed1 or changed2:
        ei = epoch_list[epoch_ind]
        bi = batch_list[batch_ind]
        Onew = pp3d.read_point_cloud(name+"/training-epoch-"+str(ei)+"-batch-"+str(bi)+".ply")
        ps_cloud.update_point_positions(Onew)





# # Show the plot
ps.init()
ps_rest_cloud = ps.register_point_cloud("Rest", np_object["ObjectSamplePts"])
ps_cloud = ps.register_point_cloud("samples",  np_object["ObjectSamplePts"], enabled=True)
ps.set_user_callback(callback)
ps.show()
