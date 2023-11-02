import random, os, sys
import json
import subprocess

object_list = [obj_name for obj_name in os.listdir("./") if obj_name.startswith("SDF_Towaki")]

for obj in object_list:
    # Define the Bash command you want to execute
    bash_command = "python 5-PhysicsSimMultiObject.py "+obj+" test1 droplonger"  # Replace this with the desired command
    # bash_command = "copy droplonger.json "+obj+"/droplonger.json"
    print("Running-------:"+bash_command)
    try:
        # Execute the Bash command and capture the output
        result = subprocess.check_output(bash_command, shell=True, text=True)
        print(result)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error code {e.returncode}")
