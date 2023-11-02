import numpy as np
import torch
import matplotlib.pyplot as plt
import random, math, sys, json
import matplotlib.colors as mcolors
import os
import pandas as pd

# 1. Read every CT_*, Nerf_*, PC_*, SDF_*, Mesh_* folder
# 2. Go through the <x>-training folders for each, read in clocktimes, timings, losses, training-settings.json, <y>_sim_timings.json

# Specify the directory where you want to start searching for folders
root_directory = '.'

# Define a function to list subfolder names in directories with specific prefixes
def list_subfolders_with_prefixes(prefixes, root_dir):
    subfolders = []
    for root, dirs, _ in os.walk(root_dir):
        for dir_name in dirs:
            for prefix in prefixes:
                if dir_name.startswith(prefix):
                    subfolders.append(os.path.join(root, dir_name))
    return subfolders

prefixes = ["CT_", "Nerf_", "SDF_", "PC_", "Mesh_"]
subfolders = list_subfolders_with_prefixes(prefixes, root_directory)

TrainingCSV = [["TrainingName", "NumHandles","NumLayers","LayerWidth","ActivationFunc",
               "NumTrainingSteps","NumSamplePts","LRStart","LREnd","TSamplingStdev",
               "TBatchSize","LossCurveMovingAvgWindow","SaveHandleIts","SaveSampleIts","NumSamplesToView","Timeit",
               "TotClockTime", "TotGPUTime"]]
SimCSV = [["SimName", "NumCubaturePts", "Steps", "NewtonIts", "BarrierIts",
                  "SetupClockTime", "StepClockTimes", "NewtonIterationClockTimes", "HessianAutodiffClockTimes", "SolveClockTimes"]]

def print_clocktimes_in_training_subfolder(subfolder):
    # List all subfolders that end with "-training"
    training_subfolders = [dir_name for dir_name in os.listdir(str(subfolder)) if dir_name.endswith("-training") ]

    for training_subfolder in training_subfolders:
        path_to_training_subfolder = subfolder + "/" + training_subfolder
        clocktimes_file = path_to_training_subfolder + "/clocktimes"
        gputimes_file = path_to_training_subfolder + "/timings"
        losses_file = path_to_training_subfolder + "/losses"

        if os.path.exists(clocktimes_file):
            clocktimes = torch.load(clocktimes_file)
            gputimes = torch.load(gputimes_file)
            losses = torch.load(losses_file)
            # Opening JSON file with training settings
            with open(path_to_training_subfolder+"/training-settings.json", 'r') as openfile:
                training_settings = json.load(openfile)
            
            training_data_timings_row = [path_to_training_subfolder] + list(training_settings.values()) + [np.sum(clocktimes), np.sum(gputimes)]
            TrainingCSV.append(training_data_timings_row)

            sim_time_files = [sim_time for sim_time in os.listdir(path_to_training_subfolder) if sim_time.endswith("_sim_timings.json")]
            for sim_time_file in sim_time_files:
                sim_name = sim_time_file.split("_sim_timings")[0]
                # print(path_to_training_subfolder+"/../"+sim_name+".json")
                # Opening JSON file with training settings
                with open(path_to_training_subfolder+"/../"+sim_name+".json", 'r') as openfile:
                    sim_settings = json.load(openfile) 
                # Opening JSON file with training settings
                with open(path_to_training_subfolder+"/"+sim_time_file, 'r') as openfile:
                    sim_timings = json.load(openfile) 

                SimCSV.append([path_to_training_subfolder, sim_settings["NumCubaturePts"], sim_settings["Steps"], sim_settings["NewtonIts"], sim_settings["BarrierIts"],
                  np.sum(sim_timings["SetupClockTime"]), np.sum(sim_timings["StepClockTimes"]), np.sum(sim_timings["NewtonIterationClockTimes"]), np.sum(sim_timings["HessianAutodiffClockTimes"]), np.sum(sim_timings["SolveClockTimes"])])
        else:
            print(f"'clocktimes' file not found in {training_subfolder}.")

if subfolders:
    print("Subfolders starting with the specified prefixes:")
    for subfolder in subfolders:
        print_clocktimes_in_training_subfolder(subfolder)
    
    print("Training Settings & Times")
    print(pd.DataFrame(TrainingCSV))
    print("")
    print("Sim settings & Time")
    print(pd.DataFrame(SimCSV))
    # print(TrainingCSV)
    # print(SimCSV)
else:
    print(f"No subfolders starting with the specified prefixes found in {root_directory}.")