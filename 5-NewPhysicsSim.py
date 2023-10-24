import torch
import torch.nn as nn
import torch.nn.functional as F 
import random, os, sys, json
from SimplicitHelpers import *
from PhysicsHelpers import *

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
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
# scene = json.loads(open(name_and_training_dir + "/../"+"squish"+".json", "r").read())

Handles_post = torch.load(object_name+"/"+training_name+"-training" + "/Handles_post")
Handles_post.to_device(device)
Handles_post.eval()

t_O = torch.tensor(np_object["ObjectSamplePts"], dtype=torch.float32).to(device)
t_YMs = torch.tensor(np_object["ObjectYMs"], dtype=torch.float32).to(device)

simulation_iteration = 0

#timestep 
dt = float(scene["dt"]) #s
move_mask = None
explicit_W = None # torch.tensor(np_object["biharmonic_weights"]).float().to(device)#Handles.getAllWeightsSoftmax(X0)

def E_Coll_Penalty(X, col_indices):
    col_penalty = torch.tensor(0, dtype=torch.float32, device=device)

    for i in range(torch.max(col_indices)):
        j = i+1
        inds_where_object_i = torch.nonzero(col_indices == i).squeeze()
        inds_where_object_j = torch.nonzero(col_indices == j).squeeze()

        M = X[inds_where_object_i, :]
        V = X[inds_where_object_j, :]

        col_penalty += torch.sum(-1*torch.log(torch.cdist(M, V, p=2)**2))
    return col_penalty

def E_pot(X0,  W, Faces, YMs, Ts, Handles):
    poisson = 0.45
    mus = YMs/(2*(1+poisson)) #shead modulus
    lams = YMs*poisson/((1+poisson)*(1-2*poisson)) #

    def elastic_energy(F, mu, lam):
        E = neohookean_E2(mu, lam, F[0,:,:])
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
    totE = (np_object["ObjectVol"]/X0.shape[0])*pt_wise_E.sum()
    return totE

def penalty(X0, x0_flat, B, J, z):

    global simulation_iteration, move_mask, dt
    x_flat = B@z + x0_flat

    #### Fixed vertices
    totE = float(scene["penalty_spring_fixed_weight"])*z.T@B.T@J@J.T@B@z

    # collE = float(scene["penalty_log_barrier_collisions"])*E_Coll_Penalty(x_flat.reshape(-1,3), col_indices=ColMap)

    ##### Moving vertices
    movE = torch.tensor([0]).to(device)
    moving_verts = x0_flat[move_mask].reshape(-1,3).detach()
    l_dict = {"moving_verts": moving_verts, "dt":dt, "simulation_iteration": simulation_iteration}
    exec(scene["SimplicitObjects"][0]["MoveBC_code"], globals(), l_dict)
    updated_vert_positions = l_dict["updated_vert_positions"]
    if updated_vert_positions != None:
        Xs = x_flat[move_mask].reshape(-1, 3)
        movE = float(scene["penalty_spring_moving_weight"])*torch.sum(torch.square(updated_vert_positions - Xs))

    #### Floor
    ys = x_flat[1::3]
    # Apply mask to select values less than floor height
    masked_values = ys[ys<float(scene["Floor"])] - float(scene["Floor"])

    # Compute the square of the masked values and sum them
    floorE = float(scene["penalty_spring_floor_weight"])*torch.sum(masked_values**2)
    # print("Energies: ", totE.item(), floorE.item(), movE.item())
    return totE + floorE + movE # + collE

def potential_energy(Phi, W, z, newx, Mg, X0, Faces, YMs,  x0_flat, J, B, Handles):  
    pe = penalty(X0, x0_flat, B, J, z)
    T = z.reshape(-1, 3,4)
    le = Phi(X0, W, Faces, YMs, T, Handles) #E_pot(X0, T, Handles)
    ge = newx.T @ Mg
    return le + ge + pe

def line_search(func, x, direction, gradient, alpha=0.5, beta=0.5):
    t = 10.0  # Initial step size
    for _ in range(int(scene["LSIts"])):
        x_new = x + t * direction
        f_new = func(x_new)
        f = func(x)
        gTd = gradient.T@direction
        if f_new <= f + alpha * t * gTd:
            return t
        t *= beta  # Reduce the step size

    return t  # Return the final step size if max_iters is reached


def getBColiNorm(W, X0, i):
    # B matrix is the modes
    # 3*|verts| x num dofs (|z|)
    
    t_ind = int(i/12) #row i gets weights from handle t_ind

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

    return torch.sum(nonzero_col_entries.square())

def simulate(np_X, Faces, np_YMs, WW, Phi, Handles):
    global simulation_iteration, move_mask, dt
    
    states = []
    
    #nx2 sample points
    X0 = torch.tensor(np_X, dtype=torch.float32, requires_grad = True, device=device)
    ones_column = torch.ones(X0.shape[0], 1).to(device)
    W = torch.cat((Handles.getAllWeightsSoftmax(X0), ones_column), dim=1)

    tYMs = torch.tensor(np_YMs, dtype=torch.float32).to(device)
    tFaces = torch.tensor(Faces).to(device)
    
    # number of handles + 1 for hard coded rigid translations
    num_handles = W.shape[1]
    num_samples = np_X.shape[0]
    

    #### Set moving boundary conditions
    move_mask = torch.zeros(num_samples*3, dtype=torch.bool)
    l_dict = {"X0": X0}
    exec(scene["SimplicitObjects"][0]["SetMovingBC_code"], globals(), l_dict)
    indices = l_dict["indices"]
    indices = (3*indices).repeat_interleave(3)
    indices[1::3] += 1
    indices[2::3] += 2; 
    move_mask[indices] = True 

    #### Set fixed boundary conditions
    # Create a mask to identify rows to be REMOVED
    mask = torch.zeros(num_samples*3, dtype=torch.bool)
    # execute code in json file to set object's fixed bc
    l_dict = {"X0": X0}
    exec(scene["SimplicitObjects"][0]["SetFixedBC_code"], globals(), l_dict)
    indices = l_dict["indices"]
    indices = (3*indices).repeat_interleave(3)
    indices[1::3] += 1
    indices[2::3] += 2; 
    # Create a mask to identify rows to be removed; 
    mask[indices] = True
    
    #### Create an nxn identity matrix
    Id = torch.eye(num_samples*3)
    # Use the mask to remove rows from the identity matrix
    redID = Id[mask]
    J = redID.T.to(device)

    # 2*num samples gravities per sample point
    grav = torch.zeros(num_samples*3).to(device)
    grav[0::3] = float(scene["Gravity"][0]) #acc from gravity
    grav[1::3] = float(scene["Gravity"][1]) #acc from gravity
    grav[2::3] = float(scene["Gravity"][2]) #acc from gravity
    
    
    #total mass of object
    density_m = float(10) #kg/m^2
    total_m = density_m*np_object["ObjectVol"]
    
    #mass matrix created  from masses assuming uniform density over mesh
    m = (total_m/X0.shape[0])*torch.ones(X0.shape[0], dtype=torch.float32, device=device)#torch.sum((total_m/W.shape[0])*W, dim=1).to(device)
    M = torch.diag(m.repeat_interleave(3)).to(device).float()
        
    # affine dofs and velocities
    z = torch.tensor([[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=torch.float32, requires_grad=True, device=device).flatten().repeat(num_handles).unsqueeze(-1)
    z_dot = torch.zeros_like(z).to(device)

    # 3*samples x 4*3*handles
    X03 = torch.cat((X0, torch.ones(X0.shape[0], device=device).unsqueeze(-1)), dim=1)
    X03reps = X03.repeat_interleave(3, dim=0).repeat((1, 3*num_handles))
    Wreps = W.repeat_interleave(12, dim=1).repeat_interleave(3, dim=0)
    WX03reps = torch.mul(Wreps, X03reps)
    Bsetup = torch.kron(torch.ones(num_samples).unsqueeze(-1), torch.eye(3)).repeat((1,num_handles)).to(device)
    Bmask = torch.repeat_interleave(Bsetup, 4, dim=1).to(device)

    B = torch.mul(Bmask, WX03reps)

    x0_flat = (X0.flatten().unsqueeze(-1))

    BMB = B.T@M@B
    BJMJB = BMB #B.T@J@J.T@M@J@J.T@B

    Mg = M@grav
    states.append(z.clone().cpu().detach())

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for step in range(int(scene["Steps"])):
        # set u prev at start of step
        z_prev = z.clone().detach()

        for iter in range(int(scene["NewtonIts"])):
            # zero out the gradients of u
            z.grad = None

            def partial_newton_E(z): 
                newx = B@z + x0_flat
                PE = potential_energy(Phi,W, z, newx, Mg, X0, tFaces, tYMs, x0_flat, J, B, Handles)
                return 0.5*z.T@BJMJB@z - z.T@BJMJB@ z_prev - dt*z.T@BJMJB@ z_dot + dt*dt*PE

            
            # Newton's method minimizes this energy
            newton_E = partial_newton_E(z)
            newton_gradE = torch.autograd.grad(newton_E, inputs = z, allow_unused=True)

            newton_hessE = torch.autograd.functional.hessian(partial_newton_E, inputs = z)

            # 18885544732 PEC
            # print(newton_E)
            # print(newton_gradE)
            # print(torch.sum(newton_hessE))
            with torch.no_grad():
                newton_H = newton_hessE[:,0,:,0]

                # Simple PSD fix
                # L, V = torch.linalg.eig(newton_H)
                # L = torch.real(L)
                # L[L<1e-2] = 1e-2
                # L = torch.complex(L, torch.zeros_like(L))
                # fixed_H = torch.real(V@torch.diag(L)@torch.linalg.inv(V))
            
                # print(step, torch.dist(fixed_H, newton_H))

                newton_g = torch.cat(newton_gradE)
                newx = B@z + x0_flat
                # Solve for x using J and g
                dz = -torch.linalg.solve(newton_H[:, :], newton_g[:])

                print(torch.norm(newton_g))
                if (torch.norm(newton_g)<2e-4):
                    print("-----------Converged")
                    break
                
                # Line Search
                alpha = line_search(partial_newton_E, z, dz, newton_g)
    
                print("ls alpha: ", alpha)
                # Update positions 
                z[:] += alpha*dz
        with torch.no_grad():
            z_dot = (z - z_prev)/dt
        states.append(z.clone().cpu().detach())
        simulation_iteration += 1
        torch.save(states, name_and_training_dir+"/" + scene_name + "-sim_states")

    return states, X0.cpu().detach(), W.detach()

random_batch_indices = np.random.randint(0, np_object["ObjectSamplePts"].shape[0], size=int(scene["NumCubaturePts"])) 
np_V = None #np_object["surfV"][:, 0:3]
np_F = 0 #np_object["surfF"]
np_X = np_object["ObjectSamplePts"][:, 0:3][random_batch_indices, :]
np_YMs = np_object["ObjectYMs"][random_batch_indices, np.newaxis]
states,  t_X0, explicit_W = simulate(np_X, np_F, np_YMs, explicit_W, E_pot, Handles_post)

torch.save(states, name_and_training_dir+"/" + scene_name + "-sim_states")
torch.save(t_X0, name_and_training_dir+"/" + scene_name + "-sim_X0")
torch.save(explicit_W, name_and_training_dir+"/" + scene_name + "-sim_W")