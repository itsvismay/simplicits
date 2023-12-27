import torch
import torch.nn as nn
import torch.nn.functional as F 
import random, os, sys, json
from typing import Dict
from SimplicitHelpers import *
from PhysicsHelpers import *

class PhysicsSimulator:

    def __init__(self, scene_name: str, np_object: Dict, name_and_training_dir: str):
        self.np_object = np_object
        self.name_and_training_dir = name_and_training_dir

        self.scene_name = scene_name
        self.scene = json.loads(open(f'scenes/{scene_name}.json', "r").read())
        self.dt = float(self.scene["dt"])  # s
        self.simulation_iteration = 0
        self.barrier_T = 0
        self.move_mask = None
        self.explicit_W = None

    def E_Coll_Penalty(self, X, col_indices):
        col_penalty = torch.tensor(0, dtype=torch.float32, device=device)

        for i in range(torch.max(col_indices)):
            j = i+1
            inds_where_object_i = torch.nonzero(col_indices == i).squeeze()
            inds_where_object_j = torch.nonzero(col_indices == j).squeeze()

            M = X[inds_where_object_i, :]
            V = X[inds_where_object_j, :]

            col_penalty += torch.sum(-1*torch.log(torch.cdist(M, V, p=2)**2))
        return col_penalty

    def E_pot(self, X0, YMs, Ts, Handles):
        poisson = float(0.45)
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
        totE = (self.np_object["ObjectVol"]/X0.shape[0])*pt_wise_E.sum()
        return totE

    def penalty(self, X0, x0_flat, B, J, z):

        scene = self.scene
        x_flat = B@z + x0_flat

        #### Fixed vertices
        totE = float(scene["penalty_spring_fixed_weight"])*z.T@B.T@J@J.T@B@z

        # collE = float(scene["penalty_log_barrier_collisions"])*E_Coll_Penalty(x_flat.reshape(-1,3), col_indices=ColMap)

        ##### Moving vertices
        movE = torch.tensor([[0]], dtype=torch.float32).to(device)
        if float(scene["penalty_spring_moving_weight"])>0:
            moving_verts = x0_flat[self.move_mask].reshape(-1,3).detach()
            l_dict = {"moving_verts": moving_verts, "dt":self.dt, "simulation_iteration": self.simulation_iteration}
            exec(scene["SimplicitObjects"][0]["MoveBC_code"], globals(), l_dict)
            updated_vert_positions = l_dict["updated_vert_positions"]
            if updated_vert_positions != None:
                Xs = x_flat[self.move_mask].reshape(-1, 3)
                movE = float(scene["penalty_spring_moving_weight"])*torch.sum(torch.square(updated_vert_positions - Xs))


        #pokes
        pokyE = torch.tensor([[0]], dtype=torch.float32).to(device)
        if len(scene["CollisionObjects"])>0:
            dist_to_poky = torch.tensor([0], dtype=torch.float32).to(device)
            for p in range(len(scene["CollisionObjects"])):
                collision_object = torch.tensor(scene["CollisionObjects"][p]["Position"], dtype=torch.float32, device=device)
                l_dict = {"collision_obj": collision_object, "dt":self.dt, "simulation_iteration": self.simulation_iteration}
                exec(scene["CollisionObjects"][p]["Update_code"], globals(), l_dict)
                collision_object = l_dict["pos"]
                dist_to_poky = torch.sqrt(torch.sum((x_flat.reshape(-1,3) - collision_object)**2, dim=1))
                min_dist_to_poky = torch.min(dist_to_poky)
                collision_object_rad = float(scene["CollisionObjects"][0]["Radius"])
                if torch.min(dist_to_poky)<collision_object_rad:
                    pokyE += torch.tensor(float("inf"), dtype=torch.float32).to(device)
                else:
                    log_dist_to_poky = -(float(scene["BarrierInitStiffness"])/(float(scene["BarrierDec"])**self.barrier_T))*torch.log(min_dist_to_poky - collision_object_rad) #log barrier is inf at collision_object_rad
                    if log_dist_to_poky>0:
                        pokyE += log_dist_to_poky
                    # pos_indx = log_dist_to_poky>0
                    # pokyE += 1*torch.sum(log_dist_to_poky[pos_indx])


        #### Floor
        floorE = torch.tensor([[0]], dtype=torch.float32).to(device)
        if float(scene["penalty_spring_floor_weight"])>0:
            ys = x_flat[1::3]
            masked_values = ys[ys<float(scene["Floor"])] - float(scene["Floor"])
            floorE = float(scene["penalty_spring_floor_weight"])*torch.sum(masked_values**2)

        if len(scene["CollisionObjects"])>0:
            print("         PokeE: ", pokyE, self.barrier_T, torch.min(dist_to_poky))
        return totE + floorE + movE + pokyE# + collE

    def potential_energy(self, Phi, z, newx, Mg, X0, YMs,  x0_flat, J, B, Handles):
        pe = self.penalty(X0, x0_flat, B, J, z)
        T = z.reshape(-1, 3,4)
        le = Phi(X0, YMs, T, Handles) #E_pot(X0, T, Handles)
        ge = newx.T @ Mg
        return le + ge + pe

    def line_search(self, func, x, direction, gradient, alpha=0.5, beta=0.5):
        t = 10.0  # Initial step size
        for _ in range(int(self.scene["LSIts"])):
            x_new = x + t * direction
            f_new = func(x_new)
            f = func(x)
            gTd = gradient.T@direction
            if f_new <= f + alpha * t * gTd:
                return t
            t *= beta  # Reduce the step size

        return t  # Return the final step size if max_iters is reached

    def getBColiNorm(self, W, X0, i):
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

    def simulate(self, np_X, np_YMs, np_PRs, np_Rho, Phi, Handles):

        states = []

        #nx2 sample points
        X0 = torch.tensor(np_X, dtype=torch.float32, requires_grad = True, device=device)
        ones_column = torch.ones(X0.shape[0], 1).to(device)
        W = torch.cat((Handles.getAllWeightsSoftmax(X0), ones_column), dim=1)

        tYMs = torch.tensor(np_YMs, dtype=torch.float32).to(device)
        tPRs = torch.tensor(np_PRs, dtype=torch.float32).to(device)

        # Use torch.nn.functional.one_hot to create the one-hot matrix for object-wise rigid deformations
        # one_hot_matrix = torch.nn.functional.one_hot(tColMap, num_classes=torch.max(tColMap)+1)
        # W = torch.cat((W, one_hot_matrix), dim=1)

        # number of handles + 1 for hard coded rigid translations
        num_handles = W.shape[1]
        num_samples = np_X.shape[0]


        #### Set moving boundary conditions
        move_mask = torch.zeros(num_samples*3, dtype=torch.bool)
        l_dict = {"X0": X0}
        exec(self.scene["SimplicitObjects"][0]["SetMovingBC_code"], globals(), l_dict)
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
        exec(self.scene["SimplicitObjects"][0]["SetFixedBC_code"], globals(), l_dict)
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
        grav[0::3] = float(self.scene["Gravity"][0]) #acc from gravity
        grav[1::3] = float(self.scene["Gravity"][1]) #acc from gravity
        grav[2::3] = float(self.scene["Gravity"][2]) #acc from gravity


        #total mass of object
        density_m = torch.tensor(np_Rho, dtype=torch.float32, device=device) #kg/m^2
        sample_vol = self.np_object["ObjectVol"]/density_m.shape[0]

        #mass matrix created  from masses assuming uniform density over mesh
        m = density_m*sample_vol#torch.sum((total_m/W.shape[0])*W, dim=1).to(device)
        M = torch.diag(m.repeat_interleave(3)).to(device).float()

        #total mass of object
        # density_m = float(10) #kg/m^2
        # total_m = density_m*np_object["ObjectVol"]

        # #mass matrix created  from masses assuming uniform density over mesh
        # m = (total_m/X0.shape[0])*torch.ones(X0.shape[0], dtype=torch.float32, device=device)#torch.sum((total_m/W.shape[0])*W, dim=1).to(device)
        # M = torch.diag(m.repeat_interleave(3)).to(device).float()

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

        for step in range(int(self.scene["Steps"])):
            # set u prev at start of step
            z_prev = z.clone().detach()

            barrier_T = 0
            for barrier_its in range(int(self.scene["BarrierIts"])):
                for iter in range(int(self.scene["NewtonIts"])):
                    # zero out the gradients of u
                    z.grad = None

                    potential_energy = self.potential_energy
                    dt = self.dt

                    def partial_newton_E(z):
                        newx = B@z + x0_flat
                        PE = potential_energy(Phi, z, newx, Mg, X0, tYMs, x0_flat, J, B, Handles)
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

                        if bool(self.scene["HessianSPDFix"]):
                            # Simple PSD fix
                            L, V = torch.linalg.eig(newton_H)
                            L = torch.real(L)
                            L[L<1e-2] = 1e-2
                            L = torch.complex(L, torch.zeros_like(L))
                            fixed_H = torch.real(V@torch.diag(L)@torch.linalg.inv(V))
                        else:
                            fixed_H = newton_H

                        # print(step, torch.dist(fixed_H, newton_H))

                        newton_g = torch.cat(newton_gradE)
                        newx = B@z + x0_flat
                        # Solve for x using J and g
                        dz = -torch.linalg.solve(fixed_H[:, :], newton_g[:])

                        print(torch.norm(newton_g))
                        if (torch.norm(newton_g)<2e-4):
                            print("-----------Converged")
                            break

                        # Line Search
                        alpha = self.line_search(partial_newton_E, z, dz, newton_g)

                        print("ls alpha: ", alpha)
                        # Update positions
                        z[:] += alpha*dz

                with torch.no_grad():
                    z_dot = (z - z_prev)/dt

                barrier_T += 1
            states.append(z.clone().cpu().detach())
            self.simulation_iteration += 1
            torch.save(states, self.name_and_training_dir+"/" + self.scene_name + "-sim_states")

        return states, X0.cpu().detach(), W.detach()
