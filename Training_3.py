import torch
import torch.nn as nn
import torch.nn.functional as F 
import random, os, sys
from SimplicitHelpers import *
import json
from trainer import Trainer

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

#Read in the object (hardcoded for now)
args = sys.argv[1:]
object_name = str(args[0])
training_name = str(args[1])

trainer = Trainer(object_name, training_name)

print("Start Training")
STARTCUDATIME = torch.cuda.Event(enable_timing=True)
ENDCUDATIME = torch.cuda.Event(enable_timing=True)
losses = []
timings = []
clock = []

name_and_training_dir = trainer.name_and_training_dir
Handles_post = trainer.Handles_post
training_settings = trainer.training_settings

for e in range(1, trainer.TOTAL_TRAINING_STEPS):
    num_handles = Handles_post.num_handles
    batch_size = int(training_settings["TBatchSize"])
    batchTs = getBatchOfTs(num_handles, batch_size, e)
    t_batchTs = batchTs.to(device).float()*float(training_settings["TSamplingStdev"])
    STARTCLOCKTIME = time.time()
    STARTCUDATIME.record()
    l1, l2 = trainer.train_step(
        Handles_post,
        trainer.t_O,
        trainer.t_YMs,
        trainer.t_PRs,
        trainer.loss_fcn,
        t_batchTs,
        e
    )
    ENDCUDATIME.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    ENDCLOCKTIME = time.time()

    timings.append(STARTCUDATIME.elapsed_time(ENDCUDATIME))  # milliseconds
    clock.append(ENDCLOCKTIME - STARTCLOCKTIME)

    print("Step: ", e, "Loss:", l1+l2," > l1: ", l1, " > l2: ", l2)
    losses.append(np.array([l1+l2, l1, l2]))

    if e % int(training_settings["SaveHandleIts"]) == 0:
    # Compute the moving average

        # save loss and handle state at current its
        torch.save(clock, name_and_training_dir+"/clocktimes-its-"+str(e))
        torch.save(timings, name_and_training_dir+"/timings-its-"+str(e))
        torch.save(losses, name_and_training_dir+"/losses-its-"+str(e))
        torch.save(Handles_post, name_and_training_dir+"/Handles_post-its-"+str(e))


        torch.save(clock, name_and_training_dir+"/clocktimes")
        torch.save(timings, name_and_training_dir+"/timings")
        torch.save(losses, name_and_training_dir+"/losses")
        torch.save(Handles_post, name_and_training_dir+"/Handles_post")

    if e % int(training_settings["SaveSampleIts"]) == 0:
        O = torch.tensor(trainer.np_object["ObjectSamplePts"], dtype=torch.float32, device=device)
        for b in range(batchTs.shape[0]):
            Ts = batchTs[b,:,:,:]
            O_new = trainer.getX(Ts, O, Handles_post)
            write_ply(name_and_training_dir+"/training-epoch-"+str(e)+"-batch-"+str(b)+".ply", O_new)

torch.save(clock, name_and_training_dir+"/clocktimes")
torch.save(timings, name_and_training_dir+"/timings")
torch.save(losses, name_and_training_dir+"/losses")
torch.save(Handles_post, name_and_training_dir+"/Handles_post")
torch.save(trainer.Handles_pre, name_and_training_dir+"/Handles_pre")