# Simplicit: Simulating Implicit Objects
The implicit simulation pipeline is divided into 7 steps.
1. Creating the object.
    a. Custom function per object with customizable post-processing for each object.
    b. Can convert to a simplicit from a SDF, pointcloud, surface mesh, tetmesh
    c. The object structure is a json of properties: 
    {   "Name":string, "Dim":scalar 2 or 3, "BoundingBoxSamplePts": u x dim np.array, "BoundingBoxSignedDists": u x 1 np.array,
        "ObjectSamplePts": n x dim np.array, "ObjectSampleSignedDists": n x 1 np.array,
        "ObjectYMs": n x 1 np.array, "ObjectPRs": n x 1 np.arry, "ObjectRho": n x 1 np.array, "ObjectColors": n x 4 np.array of RGBA
        "ObjectVol": scalar, "SurfV": v x dim np.array/None, "SurfF", f x dim+1 np.array/None, "MarchingCubesRes": scalar grid res/-1
    }
    d. Writes the object to the a binary file in folder called "<Name>/<Name>-object.json" 
    e. Also writes an editable json file in the folder called "<Name>/<Name>-training-settings.json" which has properties
        {"Dim": scalar 2 or 3, "NumHandles":scalar, "NumLayers":scalar, "LayerWidth":scalar, "ActivationFunc": string ("ELU" or "Siren"), 
         "NumTrainingSteps": int, "NumSamplePts": int, "LRStart": scalar, "LREnd": scalar,
         "TSamplingStdev": scalar, "TBatchSize": int, 
         "LossCurveMovingAvgWindow": scalar, "SaveHandleIts": scalar, "SaveSampleIts" scalar, "NumSamplesToView": scalar
        }
    f. Run by doing "python 1-SetupObject.py"

2. Creating the training settings
    a. User edits the "<Name>/<Name>-training-settings.json" file to set the training procedure
    b. Sets up the network by running "python 2-SetupTraining.py <Name> <TrainingName>"
    c. Stores the network in "<Name>/<Name>-training-<TrainingName>" where <TrainingName> is unique for each training process

3. Starts the training process.
    a. Run "python 3-Training.py <Name>"
    b. Start training by reading the "<Name>/<Name>-training-settings.json" file
    c. Saves output to folder "<Name>/<Name>-<TrainingName>/" 
    d. Output to the folder includes: 
        i. "training-recorded.json" which records all the "training-settings.json" as well as timings (per-iteration) 
        ii. the network files "Handles_pre", "Handles_post", "Handles_its-<its>, losses, losses-<its>" which records handles and losses per <SaveHandleIts> iterations
        iii. the sampled "training-it-<it>-batchT-<handle_num>.ply files saved every <SaveSampleIts> 

4. Visualize Training Results
    a. Run "python 4-AnalyzeTrainingResults.py <Name> (optional)<training-id>" or "python 4-VisualizeTraining.py <Name> (optional)<training-id>"
    b. Displays loss-curve, trained handles, and sample deformations through the training process 

5. Simulate scene 
    a. Run "python 5-PhysicsSim.py <SceneFileName>.json <ObjectName> <TrainingName>"
    b. Scenefile.json inclues the parameters 
    { "Description": string, "BoundingX":[-100, 100], "BoundingY":[-100, 100], "BoundingZ":[-100,100],
      "Floor":scalar, "Gravity": [0, 0, 0], "dt": scalar, "steps": int, "newton_iters": int, "ls_iters": int,
      "NumCubaturePts": int, "penalty_spring_fixed_weight": int, "penalty_spring_moving_weight": int, "penalty_spring_floor_weight": int,
      "HessianSPDFix": bool,
      "SimplicitObject": [
            {
                "Name": string,
                "PositionDelta": [0,0,0],
                "SetFixedBC": python code as string,
                "SetMovingBC": python code as string,
                "MoveBC":python code as string
            }
        ],
        "CollisionObject": [
            {
                "Name": string,
                "Position": 3x1 np.array,
                "Radius":
                "BarrierStiffness": scalar
                "BarrierIts": scalar
                "BarrierUpdate": python code
            }
        ]
    }

6. Display simulation "python 6-AnimateWithPolyscope.py..."
