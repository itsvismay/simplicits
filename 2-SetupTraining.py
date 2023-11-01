import torch
import torch.nn as nn
import torch.nn.functional as F 
import random, os, sys
from SimplicitHelpers import *
import json

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
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


Handles_pre = HandleModels(training_settings["NumHandles"], training_settings["NumLayers"], training_settings["LayerWidth"], training_settings["ActivationFunc"], np_object["Dim"], training_settings["LRStart"])
Handles_post = HandleModels(training_settings["NumHandles"], training_settings["NumLayers"], training_settings["LayerWidth"], training_settings["ActivationFunc"], np_object["Dim"], training_settings["LRStart"])

## Moving stuff to GPU
Handles_post.to_device(device)
Handles_pre.to_device(device)
Handles_pre.eval()

t_O = torch.tensor(np_object["ObjectSamplePts"][:,0:3]).to(device)

for nnnn, pppp in Handles_post.model.named_parameters():
    print(nnnn, pppp.size())

np_W0, np_X0, np_G0 = test(Handles_post, t_O, int(t_O.shape[0]/10))
plot_handle_regions(np_X0, np_W0, "Pre Training Handle Weights")
plot_implicit(np_object["ObjectSamplePts"], np_object["ObjectYMs"] )


print(object_name+"/"+training_name+"-training")
if not os.path.exists(object_name+"/"+training_name+"-training"):
    os.makedirs(object_name+"/"+training_name+"-training")

# rewrite over training settings, and losses and handle state (final)
with open(object_name+"/"+training_name+"-training/training-settings.json", 'w', encoding='utf-8') as f:
    json.dump(training_settings, f, ensure_ascii=False, indent=4)

torch.save(Handles_post, object_name+"/"+training_name+"-training"+"/Handles_post")
torch.save(Handles_pre, object_name+"/"+training_name+"-training"+"/Handles_pre")