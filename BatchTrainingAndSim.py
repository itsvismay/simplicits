import torch
import torch.nn as nn
import torch.nn.functional as F 
import random, os, sys
from SimplicitHelpers import *
import json
import subprocess

object_list = [obj_name for obj_name in os.listdir("./") if obj_name.startswith("SDF_Towaki")]

# Batch copy
# for obj in object_list:
#     # Define the Bash command you want to execute
#     bash_command = "copy dropstiffer.json "+obj+"/dropstiffer.json"  # Replace this with the desired command
#     print("Running-------:"+bash_command)
#     try:
#         # Execute the Bash command and capture the output
#         result = subprocess.check_output(bash_command, shell=True, text=True)
#         print(result)
#     except subprocess.CalledProcessError as e:
#         print(f"Command failed with error code {e.returncode}")

#Training Setup
# for obj in object_list:
#     # Define the Bash command you want to execute
#     bash_command = "python 2-SetupTraining.py "+obj+" test1"  # Replace this with the desired command
#     print("Running-------:"+bash_command)
#     try:
#         # Execute the Bash command and capture the output
#         result = subprocess.check_output(bash_command, shell=True, text=True)
#         print(result)
#     except subprocess.CalledProcessError as e:
#         print(f"Command failed with error code {e.returncode}")

# #Training
# for obj in object_list:
#     # Define the Bash command you want to execute
#     bash_command = "python 3-Training.py "+obj+" test1"  # Replace this with the desired command
#     print("Running-------:"+bash_command)
#     try:
#         # Execute the Bash command and capture the output
#         result = subprocess.check_output(bash_command, shell=True, text=True)
#         print(result)
#     except subprocess.CalledProcessError as e:
#         print(f"Command failed with error code {e.returncode}")

# # Simulate
for obj in object_list:
    # Define the Bash command you want to execute
    bash_command = "python 5-PhysicsSimMultiObject.py "+obj+" test1 dropstiffer"  # Replace this with the desired command
    # bash_command = "copy droplonger.json "+obj+"/droplonger.json"
    print("Running-------:"+bash_command)
    try:
        # Execute the Bash command and capture the output
        result = subprocess.check_output(bash_command, shell=True, text=True)
        print(result)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error code {e.returncode}")

