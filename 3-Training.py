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

# Opening JSON file with training settings
with open(object_name+"/"+training_name+"-training/training-settings.json", 'r') as openfile:
    training_settings = json.load(openfile)

name_and_training_dir = object_name+"/"+training_name+"-training"

Handles_post = torch.load(name_and_training_dir+"/Handles_post")
Handles_pre = torch.load(name_and_training_dir+"/Handles_pre")

Handles_post.to_device(device)

t_O = torch.tensor(np_object["ObjectSamplePts"][:,0:3]).to(device)
t_YMs = torch.tensor(np_object["ObjectYMs"]).unsqueeze(-1).to(device)
t_PRs = torch.tensor(np_object["ObjectPRs"]).unsqueeze(-1).to(device)
# Use torch.where to find indices where the value is equal to 2e4
ym_min_val = t_YMs.min()
ym_max_val = t_YMs.max()
stiffer_indices = torch.where(t_YMs == ym_max_val)[0]

TOTAL_TRAINING_STEPS = int(training_settings["NumTrainingSteps"])

LE_End = 1
T_only = 0
if "LE" in training_name.split("_")[-1]:
    print("LE")
    LE_End = 0
if "T" in training_name.split("_")[-1]:
    print("T")
    T_only = 1
    
ENERGY_INTERP_LINSPACE = np.linspace(0, LE_End, TOTAL_TRAINING_STEPS, endpoint=False)
LR_INTERP_LINSPCE = np.linspace(float(training_settings["LRStart"]), float(training_settings["LREnd"]), TOTAL_TRAINING_STEPS, endpoint=True)
YM_INTERP_LINSPACE = np.linspace(ym_max_val.cpu().detach().numpy(), ym_max_val.cpu().detach().numpy(), TOTAL_TRAINING_STEPS, endpoint=True)
eps = 0.0001
eps0 = torch.tensor([eps, 0, 0]).to(device)
eps1 = torch.tensor([0, eps, 0]).to(device)
eps2 = torch.tensor([0, 0, eps]).to(device)

def interp(e, TOT):
    #logarithmic
    # mdpt = TOT/2
    # x = e-mdpt
    # val = 1/(1+np.exp(-(10/TOT)*x))

    #linear
    val =  float(e)/float(TOT)
    return val

def getX(Ts, lX0, Handles):
    def x(x0):
        x0_i = x0.unsqueeze(0)
        x03 = torch.cat((x0_i, torch.tensor([[1]], device=device)), dim=1)
        t_W = Handles.getAllWeightsSoftmax(x0_i).T
        
        def inner_over_handles(T, w):
            return w*T@x03.T

        wTx03s = torch.vmap(inner_over_handles)(Ts, t_W)
        x_i =  torch.sum(wTx03s, dim=0) 
        return x_i.T + x0_i
    
    X = torch.vmap(x, randomness="same")(lX0)
    return X[:,0,:]

def E_Coll_Penalty(X, col_indices):
    col_penalty = torch.tensor(0, dtype=torch.float32, device=device)

    for i in range(torch.max(col_indices)):
        j = i+1
        inds_where_object_i = torch.nonzero(col_indices == i).squeeze()
        inds_where_object_j = torch.nonzero(col_indices == j).squeeze()

        M = X[inds_where_object_i, :]
        V = X[inds_where_object_j, :]
        col_penalty += torch.sum(-10*torch.log(torch.cdist(M, V, p=2)**2 + 1e-9))
    return col_penalty

def E_pot(W,X0, mus, lams, Ts, Handles, interp_val):

    def elastic_energy(F, mu, lam):
        En = interp_val*neohookean_E(mu, lam, F[0,:,:])
        El = (1.0-interp_val)*linear_elastic_E(mu, lam, F[0,:,:])
        return En + El
    
    def x(x0):
        x0_i = x0.unsqueeze(0)
        x03 = torch.cat((x0_i, torch.tensor([[1]], device=device)), dim=1)
        t_W = Handles.getAllWeightsSoftmax(x0_i).T
        
        def inner_over_handles(T, w):
            return w*T@x03.T

        wTx03s = torch.vmap(inner_over_handles)(Ts, t_W)
        x_i =  torch.sum(wTx03s, dim=0) 
        return x_i.T + x0_i
    
    def fdF2(x0):
        #left
        xx = x0 + eps0
        left = x(xx)
        xx = x0 - eps0
        right = x(xx)
        col1 = (left - right)/(2*eps)

        xx = x0 + eps1
        left = x(xx)
        xx = x0 - eps1
        right = x(xx)
        col2 = (left - right)/(2*eps)
        
        xx = x0 + eps2
        left = x(xx)
        xx = x0 - eps2
        right = x(xx)
        col3 = (left - right)/(2*eps)
        
        # Create a PyTorch matrix from the columns
        return torch.stack((col1, col2, col3), dim=1)

    def fdF1(x0):
        right = x(x0)
        #left
        xx = x0 + eps0
        left = x(xx)
        col1 = (left - right)/(eps)

        xx = x0 + eps1
        left = x(xx)
        col2 = (left - right)/(eps)
        
        xx = x0 + eps2
        left = x(xx)
        col3 = (left - right)/(eps)
        
        # Create a PyTorch matrix from the columns
        return torch.stack((col1, col2, col3), dim=1)




    pt_wise_Fs = torch.vmap(torch.func.jacrev(x), randomness="same")(X0)
    # pt_wise_Fs = torch.vmap(fdF1, randomness="same")(X0)
    pt_wise_E = torch.vmap(elastic_energy, randomness="same")(pt_wise_Fs, mus, lams)
    totE = (np_object["ObjectVol"]/X0.shape[0])*(pt_wise_E.sum())
    return totE

def loss_w_sum(_Weights_):
    # dimensions should be |sample points| x |strandles|
    # weights summed at each sample point should be 1, with one strandle all weights = 1
    sum_weights_pointwise = torch.sum(_Weights_, dim=1)
    ones = torch.ones_like(sum_weights_pointwise)
    return nn.MSELoss()(sum_weights_pointwise, ones) 

def loss_handle_w(W, P0):
    # weights at each handle should be 1 and 0 for all other handles
    return nn.MSELoss()(W[W.shape[0]-P0.shape[0]:, :], torch.eye(P0.shape[0]).to(device))

def getBColiNorm(W, X0, i):
    # B matrix is the modes
    # 3*|verts| x num dofs (|z|)
    
    t_ind = torch.int(i/12) #row i gets weights from handle t_ind

    def nzBColi(wt_n, x_n):
        if i%4 ==0:
            return wt_n*x_n[0]
        elif i%4 ==1:
            return wt_n*x_n[1]
        elif i%4 ==2:
            return wt_n*x_n[2]
        elif i%4 ==3:
            return wt_n
    
    nonzero_col_entries = torch.vmap(nzBColi, randomness="same")(W[:, t_ind], X0)
    return nonzero_col_entries

# def orthogonality_loss(W, X0):
#     num_B_cols = 12*W.shape[1]

#     #BtB = torch.zeros((num_B_cols, num_B_cols), dtype=torch.float32).to(device) 
#     L = torch.tensor(0, dtype=torch.float32, device=device)
    
#     def BtB_i(i):
#         coli = getBColiNorm(W, X0, i)
#         def BTB_j(j):
#             colj = getBColiNorm(W, X0, j)
#             if i==j:
#                 L += (torch.dot(coli, colj) - torch.tensor(1, dtype=torch.float32).to(device))**2
#             else:
#                 L += (torch.dot(coli, colj) - torch.tensor(0, dtype=torch.float32).to(device))**2
#         torch.vmap(BTB_j)(torch.arange(i+1))
#     torch.vmap(BtB_i)(torch.arange(num_B_cols))
#     return L

def loss_fcn(W, X0, mus, lams, batchTs, Handles,  interp_val):
    L1 = torch.tensor(0, dtype=torch.float32).to(device)
    L3 = torch.tensor(0, dtype=torch.float32).to(device)
    for b in range(batchTs.shape[0]):
        Ts = batchTs[b,:,:,:]
        L1 += E_pot(W, X0, mus, lams, Ts, Handles, interp_val)/batchTs.shape[0]

    L1 *=0.1



    # num_handles = W.shape[1]
    # num_samples = X0.shape[0]
    # X03 = torch.cat((X0, torch.ones(X0.shape[0], device=device).unsqueeze(-1)), dim=1)
    # X03reps = X03.repeat_interleave(3, dim=0).repeat((1, 3*num_handles))
    # Wreps = W.repeat_interleave(12, dim=1).repeat_interleave(3, dim=0)
    # WX03reps = torch.mul(Wreps, X03reps)
    # Bsetup = torch.kron(torch.ones(num_samples).unsqueeze(-1), torch.eye(3)).repeat((1,num_handles)).to(device)
    # Bmask = torch.repeat_interleave(Bsetup, 4, dim=1).to(device)

    # B = torch.mul(Bmask, WX03reps)

    # BtB = B.T@B
    # L2 = 1e6*nn.MSELoss()(BtB, torch.eye(B.shape[1]).to(device))

    L2 = 1e6*nn.MSELoss()(W.T@W, torch.eye(W.shape[1]).to(device))

    # L2 = 1000000*orthogonality_loss(W, X0)

    return L1, L2

def train(Handles, O, YoungsMod, PRs, loss_fcn, batchTs, step):
    Handles.train()
    random_batch_indices = torch.randint(low=0, high=int(t_O.shape[0]), size=(int(training_settings["NumSamplePts"]),))
    X0 = O[random_batch_indices].float().to(device)
    X0.requires_grad = True

    YMs = YoungsMod[random_batch_indices,:].float().to(device)
    poisson = PRs[random_batch_indices,:].float().to(device)
    mus = YMs/(2*(1+poisson)) #shead modulus
    lams = YMs*poisson/((1+poisson)*(1-2*poisson)) #
    W = Handles.getAllWeightsSoftmax(X0)
    
    # Backpropagation
    Handles.optimizers_zero_grad()

    interp_val = ENERGY_INTERP_LINSPACE[step]

    loss1, loss2 = loss_fcn(W, X0, mus, lams, batchTs, Handles, interp_val)
    loss = loss1 + loss2
    loss.backward()
    
    # Backpropagation
    Handles.optimizers_step()
    Handles.updateLR(LR_INTERP_LINSPCE[step])
    
    return loss1.item(), loss2.item()

# train(Handles, np_P0_, np_O_, loss_fcn, [getRandomTransform(2, 1) for i in range(Handles.num_handles)])
print("Start Training")

STARTCUDATIME = torch.cuda.Event(enable_timing=True)
ENDCUDATIME = torch.cuda.Event(enable_timing=True)
losses = []
timings = []
clock = []

steps = TOTAL_TRAINING_STEPS
for e in range(1, steps):
    batchTs = getBatchOfTs(Handles_post.num_handles, int(training_settings["TBatchSize"]), t_only=T_only)
    t_batchTs = batchTs.to(device).float()*float(training_settings["TSamplingStdev"])
    STARTCLOCKTIME = time.time()
    STARTCUDATIME.record()
    l1, l2 = train(Handles_post, t_O, t_YMs, t_PRs, loss_fcn, t_batchTs, e)
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
        O = torch.tensor(np_object["ObjectSamplePts"], dtype=torch.float32, device=device)
        for b in range(batchTs.shape[0]):
            Ts = batchTs[b,:,:,:]
            O_new = getX(Ts, O, Handles_post)
            write_ply(name_and_training_dir+"/training-epoch-"+str(e)+"-batch-"+str(b)+".ply", O_new)

torch.save(clock, name_and_training_dir+"/clocktimes")
torch.save(timings, name_and_training_dir+"/timings")
torch.save(losses, name_and_training_dir+"/losses")
torch.save(Handles_post, name_and_training_dir+"/Handles_post")
torch.save(Handles_pre, name_and_training_dir+"/Handles_pre")