import torch
import torch.nn as nn
import torch.nn.functional as F 
import random, os, sys
from SimplicitHelpers import *
import json
import subprocess


#Read in the object (hardcoded for now)
args = sys.argv[1:]
object_name = str(args[0])

fname = object_name+"/"+object_name
np_object = torch.load(fname+"-object")


#SETUP TRAINING
def setupTraining(object_name, training_name, training_settings):
    Handles_pre = HandleModels(training_settings["NumHandles"], training_settings["NumLayers"], training_settings["LayerWidth"], training_settings["ActivationFunc"], np_object["Dim"], training_settings["LRStart"])
    Handles_post = HandleModels(training_settings["NumHandles"], training_settings["NumLayers"], training_settings["LayerWidth"], training_settings["ActivationFunc"], np_object["Dim"], training_settings["LRStart"])

    ## Moving stuff to GPU
    Handles_post.to_device(device)
    Handles_pre.to_device(device)
    Handles_pre.eval()

    print(object_name+"/"+training_name+"-training")
    if not os.path.exists(object_name+"/"+training_name+"-training"):
        os.makedirs(object_name+"/"+training_name+"-training")

    # rewrite over training settings, and losses and handle state (final)
    with open(object_name+"/"+training_name+"-training/training-settings.json", 'w', encoding='utf-8') as f:
        json.dump(training_settings, f, ensure_ascii=False, indent=4)

    torch.save(Handles_post, object_name+"/"+training_name+"-training"+"/Handles_post")
    torch.save(Handles_pre, object_name+"/"+training_name+"-training"+"/Handles_pre")


# Opening JSON file with training settings
with open(fname+"-training-settings.json", 'r') as openfile:
    default_training_settings = json.load(openfile)

training_names =["AB_6H6L_FD1_T", "AB_6H6L_FD1_LE", "AB_6H6L_FD1_TLE", "AB_12H6L_FD1_T", "AB_12H6L_FD1_LE", "AB_12H6L_FD1_TLE" ]

# for tname in training_names:
#     numH = int(tname.split("_")[1].split("H")[0])
#     numL = int(tname.split("_")[1].split("L")[0].split("H")[1])
#     default_training_settings["NumHandles"] = numH
#     default_training_settings["NumLayers"] = numL   
#     setupTraining(object_name=object_name, training_name=tname, training_settings=default_training_settings)

# DO THE TRAINING
# for name in training_names:
#     # Define the Bash command you want to execute
#     bash_command = "python 3-Training.py "+object_name+" "+name  # Replace this with the desired command
#     print("Running-------:"+bash_command)
#     try:
#         # Execute the Bash command and capture the output
#         result = subprocess.check_output(bash_command, shell=True, text=True)
#         print(result)
#     except subprocess.CalledProcessError as e:
#         print(f"Command failed with error code {e.returncode}")

# Simulate
# for name in training_names:
#     # Define the Bash command you want to execute
#     bash_command = "python 5-PhysicsSimMultiObject.py "+object_name+" "+name +" droop"  # Replace this with the desired command
#     print("Running-------:"+bash_command)
#     try:
#         # Execute the Bash command and capture the output
#         result = subprocess.check_output(bash_command, shell=True, text=True)
#         print(result)
#     except subprocess.CalledProcessError as e:
#         print(f"Command failed with error code {e.returncode}")

# Output
for name in training_names:
    # Define the Bash command you want to execute
    bash_command = "python 7-OutputResults.py "+object_name+" "+name +" twist_FD 1"  # Replace this with the desired command
    print("Running-------:"+bash_command)
    try:
        # Execute the Bash command and capture the output
        result = subprocess.check_output(bash_command, shell=True, text=True)
        print(result)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error code {e.returncode}")
