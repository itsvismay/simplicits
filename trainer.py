import torch
import torch.nn as nn
import torch.nn.functional as F
import random, os, sys
from SimplicitHelpers import *
import json


class Trainer:

    def __init__(self, object_name, training_name):
        # Get cpu or gpu device for training.
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        print(f"Using {device} device")

        # Read in the object (hardcoded for now)
        fname = object_name + "/" + object_name
        self.np_object = torch.load(fname + "-object")

        # Opening JSON file with training settings
        with open(fname + "-training-settings.json", 'r') as openfile:
            self.training_settings = json.load(openfile)

        self.name_and_training_dir = object_name + "/" + training_name + "-training"

        self.Handles_post = torch.load(self.name_and_training_dir + "/Handles_post")
        self.Handles_pre = torch.load(self.name_and_training_dir + "/Handles_pre")

        self.Handles_post.to_device(device)

        self.t_O = torch.tensor(self.np_object["ObjectSamplePts"][:, 0:3]).to(device)
        self.t_YMs = torch.tensor(self.np_object["ObjectYMs"]).unsqueeze(-1).to(device)
        self.t_PRs = torch.tensor(self.np_object["ObjectPRs"]).unsqueeze(-1).to(device)
        # Use torch.where to find indices where the value is equal to 2e4
        self.ym_min_val = self.t_YMs.min()
        self.ym_max_val = self.t_YMs.max()
        self.stiffer_indices = torch.where(self.t_YMs == self.ym_max_val)[0]

        self.TOTAL_TRAINING_STEPS = int(self.training_settings["NumTrainingSteps"])

        self.ENERGY_INTERP_LINSPACE = np.linspace(0, 1, self.TOTAL_TRAINING_STEPS, endpoint=False)
        self.LR_INTERP_LINSPCE = np.linspace(float(self.training_settings["LRStart"]),
                                             float(self.training_settings["LREnd"]),
                                             self.TOTAL_TRAINING_STEPS, endpoint=True)
        self.YM_INTERP_LINSPACE = np.linspace(self.ym_max_val.cpu().detach().numpy(),
                                              self.ym_max_val.cpu().detach().numpy(),
                                              self.TOTAL_TRAINING_STEPS, endpoint=True)

    def interp(self, e, TOT):
        # logarithmic
        # mdpt = TOT/2
        # x = e-mdpt
        # val = 1/(1+np.exp(-(10/TOT)*x))

        # linear
        val = float(e) / float(TOT)
        return val

    def getX(self, Ts, lX0, Handles):
        def x(x0):
            x0_i = x0.unsqueeze(0)
            x03 = torch.cat((x0_i, torch.tensor([[1]], device=x0.device)), dim=1)
            t_W = Handles.getAllWeightsSoftmax(x0_i).T.to(x0.device)

            def inner_over_handles(T, w):
                return w * T @ x03.T

            wTx03s = torch.vmap(inner_over_handles)(Ts, t_W)
            x_i = torch.sum(wTx03s, dim=0)
            return x_i.T + x0_i

        X = torch.vmap(x, randomness="same")(lX0)
        return X[:, 0, :]

    def E_Coll_Penalty(self, X, col_indices):
        col_penalty = torch.tensor(0, dtype=torch.float32, device=self.device)

        for i in range(torch.max(col_indices)):
            j = i + 1
            inds_where_object_i = torch.nonzero(col_indices == i).squeeze()
            inds_where_object_j = torch.nonzero(col_indices == j).squeeze()

            M = X[inds_where_object_i, :]
            V = X[inds_where_object_j, :]
            col_penalty += torch.sum(-10 * torch.log(torch.cdist(M, V, p=2) ** 2 + 1e-9))
        return col_penalty

    def E_pot(self, W, X0, mus, lams, Ts, Handles, interp_val):

        def elastic_energy(F, mu, lam):
            En = interp_val * neohookean_E(mu, lam, F[0, :, :])
            El = (1.0 - interp_val) * linear_elastic_E(mu, lam, F[0, :, :])
            return En + El

        def x(x0):
            x0_i = x0.unsqueeze(0)
            x03 = torch.cat((x0_i, torch.tensor([[1]], device=self.device)), dim=1)
            t_W = Handles.getAllWeightsSoftmax(x0_i).T

            def inner_over_handles(T, w):
                return w * T @ x03.T

            wTx03s = torch.vmap(inner_over_handles)(Ts, t_W)
            x_i = torch.sum(wTx03s, dim=0)
            return x_i.T + x0_i

        pt_wise_Fs = torch.vmap(torch.func.jacrev(x), randomness="same")(X0)
        pt_wise_E = torch.vmap(elastic_energy, randomness="same")(pt_wise_Fs, mus, lams)
        totE = (self.np_object["ObjectVol"] / X0.shape[0]) * (pt_wise_E.sum())
        return totE

    def loss_w_sum(self, _Weights_):
        # dimensions should be |sample points| x |strandles|
        # weights summed at each sample point should be 1, with one strandle all weights = 1
        sum_weights_pointwise = torch.sum(_Weights_, dim=1)
        ones = torch.ones_like(sum_weights_pointwise)
        return nn.MSELoss()(sum_weights_pointwise, ones)

    def loss_handle_w(self, W, P0):
        # weights at each handle should be 1 and 0 for all other handles
        return nn.MSELoss()(W[W.shape[0] - P0.shape[0]:, :], torch.eye(P0.shape[0]).to(self.device))

    def getBColiNorm(self, W, X0, i):
        # B matrix is the modes
        # 3*|verts| x num dofs (|z|)

        t_ind = torch.int(i / 12)  # row i gets weights from handle t_ind

        def nzBColi(wt_n, x_n):
            if i % 4 == 0:
                return wt_n * x_n[0]
            elif i % 4 == 1:
                return wt_n * x_n[1]
            elif i % 4 == 2:
                return wt_n * x_n[2]
            elif i % 4 == 3:
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

    def loss_fcn(self, W, X0, mus, lams, batchTs, Handles, interp_val):
        L1 = torch.tensor(0, dtype=torch.float32).to(self.device)
        L3 = torch.tensor(0, dtype=torch.float32).to(self.device)
        for b in range(batchTs.shape[0]):
            Ts = batchTs[b, :, :, :]
            L1 += self.E_pot(W, X0, mus, lams, Ts, Handles, interp_val) / batchTs.shape[0]

        L1 *= 0.1

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

        L2 = 1e6 * nn.MSELoss()(W.T @ W, torch.eye(W.shape[1]).to(self.device))

        # L2 = 1000000*orthogonality_loss(W, X0)

        return L1, L2

    def train_step(self, Handles, O, YoungsMod, PRs, loss_fcn, batchTs, step):
        Handles.train()
        random_batch_indices = torch.randint(low=0, high=int(self.t_O.shape[0]),
                                             size=(int(self.training_settings["NumSamplePts"]),))
        X0 = O[random_batch_indices].float().to(self.device)
        X0.requires_grad = True

        YMs = YoungsMod[random_batch_indices, :].float().to(self.device)
        poisson = PRs[random_batch_indices, :].float().to(self.device)
        mus = YMs / (2 * (1 + poisson))  # shead modulus
        lams = YMs * poisson / ((1 + poisson) * (1 - 2 * poisson))  #
        W = Handles.getAllWeightsSoftmax(X0)

        # Backpropagation
        Handles.optimizers_zero_grad()

        interp_val = self.ENERGY_INTERP_LINSPACE[step]

        loss1, loss2 = loss_fcn(W, X0, mus, lams, batchTs, Handles, interp_val)
        loss = loss1 + loss2
        loss.backward()

        # Backpropagation
        Handles.optimizers_step()
        Handles.updateLR(self.LR_INTERP_LINSPCE[step])

        return loss1.item(), loss2.item()