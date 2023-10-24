import igl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from functorch import vmap
import random, math
import matplotlib.tri as tri
import polyscope as ps
import polyscope.imgui as psim
import inspect
from scipy.interpolate import RBFInterpolator
import matplotlib.colors as mcolors
# from vectoradam import * 
from siren_pytorch import SirenNet
import time


global device
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

def printarr(*arrs, data=True, short=True, max_width=200):

    # helper for below
    def compress_str(s):
        return s.replace('\n', '')
    name_align = ">" if short else "<"

    # get the name of the tensor as a string
    frame = inspect.currentframe().f_back
    try:
        # first compute some length stats
        name_len = -1
        dtype_len = -1
        shape_len = -1
        default_name = "[unnamed]"
        for a in arrs:
            name = default_name
            for k,v in frame.f_locals.items():
                if v is a:
                    name = k
                    break
            name_len = max(name_len, len(name))
            dtype_len = max(dtype_len, len(str(a.dtype)))
            shape_len = max(shape_len, len(str(a.shape)))
        len_left = max_width - name_len - dtype_len - shape_len - 5

        # now print the acual arrays
        for a in arrs:
            name = default_name
            for k,v in frame.f_locals.items():
                if v is a:
                    name = k
                    break
            print(f"{name:{name_align}{name_len}} {str(a.dtype):<{dtype_len}} {str(a.shape):>{shape_len}}", end='') 
            if data:
                # print the contents of the array
                print(": ", end='')
                flat_str = compress_str(str(a))
                if len(flat_str) < len_left:
                    # short arrays are easy to print
                    print(flat_str)
                else:
                    # long arrays
                    if short:
                        # print a shortented version that fits on one line
                        if len(flat_str) > len_left - 4:
                            flat_str = flat_str[:(len_left-4)] + " ..."
                        print(flat_str)
                    else:
                        # print the full array on a new line
                        print("")
                        print(a)
            else:
                print("") # newline
    finally:
        del frame


# Plotting functions
#--------------------
# Plot the implicit object (blue scatter samples) and strandle (red)
def plot_implicit(sample_points, scalars=None, colors = None):
    ps.init()
    ps.remove_all_structures()
    ps_unit_sphere = ps.register_point_cloud("Unit Sphere", np.array([[0,0,0],[1,0,0]]), radius=0.1)
    ps_cloud1 = ps.register_point_cloud("sample points", sample_points)
    if scalars is not None:
        ps_cloud1.add_scalar_quantity("scalar vals", scalars,datatype="symmetric")
    if colors is not None:
        ps_cloud1.add_color_quantity("color vals", colors)
    ps.show()  

def plot_mesh(V,  E):
    ps.init()
    ps.remove_all_structures()
    ps_vol = ps.register_surface_mesh("test surf mesh", V, E)
    ps.show()  

def plot_handle_regions(points, weights, title):
    handle_num = 0

    max_all_w = np.max(np.max(weights))
    min_all_w = np.min(np.min(weights))

    # Combo box to choose from options
    # There, the options are a list of strings in `ui_options`,
    # and the currently selected element is stored in `ui_options_selected`.
    def callback():
        nonlocal handle_num
        changed, handle_num = psim.SliderInt("Handles:", handle_num, v_min=0, v_max=weights.shape[1]-1)
        if changed:
            ii = int(handle_num)
            w = weights[:, ii]
            # ps_cloud.add_scalar_quantity("Weight: "+str(ii), w, enabled=True)
            # Normalize the weights to the range [0, 1]
            w_norm = (w - min_all_w) / (max_all_w - min_all_w)
            ps_cloud.add_scalar_quantity("Weight: "+str(ii), w_norm, enabled=True)
            # color_mat = np.column_stack((w_norm, w_norm, w_norm))
            # color_mat[:,1:2] = 1
            # # print(color_mat)
            # ps_cloud.add_color_quantity("weights: "+str(ii), color_mat, enabled=True)




    # Show the plot
    ps.init()
    ps.remove_all_structures()
    ps_cloud = ps.register_point_cloud("samples", points, enabled=True)
    ps_cloud.add_scalar_quantity("Weight: "+str(0), weights[:,0], enabled=True)
    ps.set_user_callback(callback)
    ps.show()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_deformation(X0, W, T, L1, L2, L3):
    # 2*samples x 3*2*handles
    X03 = torch.cat((X0, torch.ones(X0.shape[0]).unsqueeze(-1)), dim=1)
    X03reps = X03.repeat_interleave(2, dim=0).repeat((1, 2*W.shape[1]))

    Wreps = W.repeat_interleave(6, dim=1).repeat_interleave(2, dim=0)
    WX03reps = torch.mul(Wreps, X03reps)

    Bsetup = torch.kron(torch.ones(X0.shape[0]).unsqueeze(-1), torch.eye(2)).repeat((1,W.shape[1]))
    Bmask = torch.repeat_interleave(Bsetup, 3, dim=1)
    B = torch.mul(Bmask, WX03reps)

    z = T.flatten()

    X = torch.matmul(B, z)
    np_X = X.reshape((-1, 2)).cpu().detach().numpy()
    sample_points = np_X[0:np_X.shape[0]-W.shape[1], :]
    P0 = np_X[sample_points.shape[0]:, :]
    
    fig, ax = plt.subplots()
    ax.scatter(sample_points[:,0], sample_points[:,1], s=10)
    ax.plot(P0[:,0], P0[:,1], "ro")
    ax.set_xlim([-1, 25])
    ax.set_ylim([-1, 25])
    ax.set_aspect('equal')
    ax.set_title("Elastic: " + str(L1.data) +", POU: " + str(L2.data) + ", Handle=1: "+str(L3.data))
    plt.show()

def write_ply(file_path, vertices, rgb=None, faces=None):
    with open(file_path, "w") as f:
        num_vertices = len(vertices)

        # Check if RGB values are provided, and adjust the header accordingly
        has_rgb = rgb is not None
        if(has_rgb):
            vertices = np.concatenate((vertices, rgb), axis=1)
  
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if has_rgb:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        if faces is not None:
            num_faces = len(faces)
            f.write(f"element face {num_faces}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Write vertices
        for vertex in vertices:
            x, y, z = vertex[:3]
            f.write(f"{x} {y} {z}")

            if has_rgb:
                r, g, b =  vertex[3:]
                f.write(f" {int(r)} {int(g)} {int(b)}")

            f.write("\n")

        if faces is not None:
            # Write faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
                
def read_ply(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise Exception("File not found: {}".format(file_path))

    vertices = []
    colors = []
    data_started = False

    for line in lines:
        line = line.strip()

        if data_started:
            if len(line) == 0:
                break  # End of data
            values = line.split()
            x, y, z, r, g, b, a = map(float, values)
            vertices.append([x, y, z])
            colors.append([r, g, b])

        if line == "end_header":
            data_started = True

    return np.array(vertices), np.array(colors)
                

def voxelize_point_cloud(points, voxel_size, fcn, sdf_vals = None):
    """
    Voxelize a point cloud into a voxel grid.

    Args:
        points (numpy.ndarray): Point cloud array of shape (N, 3).
        voxel_size (float): Size of each voxel.

    Returns:
        numpy.ndarray: Voxel grid array of shape (W, H, D) with boolean values.

    """
    # Determine the extents of the point cloud
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # Compute the size of the voxel grid
    voxel_grid_size = np.ceil((max_coords - min_coords) / voxel_size).astype(int)

    x = np.arange(min_coords[0], min_coords[0] + voxel_grid_size[0] * voxel_size, voxel_size)
    y = np.arange(min_coords[1], min_coords[1] + voxel_grid_size[1] * voxel_size, voxel_size)
    z = np.arange(min_coords[2], min_coords[2] + voxel_grid_size[2] * voxel_size, voxel_size)

    # Create 3D coordinate arrays using meshgrid
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    # Compute the nodes of the voxel grid
    # Stack the coordinate arrays and reshape to nx3
    voxel_node_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T    
    voxel_node_sdfs = np.apply_along_axis(fcn, 1, voxel_node_points)

    # Create an empty voxel grid
    voxel_grid = np.zeros(voxel_grid_size, dtype=np.float32)

    # Compute the voxel indices for each point
    voxel_indices = ((points - min_coords) / voxel_size).astype(int)

    voxel_grid = voxel_node_sdfs.reshape(x.shape)
    # # Set the corresponding voxels to True
    # if sdf_vals.any() == None:
    #     voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
    # else:
    #     voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = voxel_node_sdfs

    return voxel_grid, voxel_node_points, voxel_node_sdfs

def interpolate_point_cloud(points, voxel_size, signed_distances):
    """
    Voxelize a point cloud into a voxel grid.

    Args:
        points (numpy.ndarray): Point cloud array of shape (N, 3) - not rest state.
        voxel_size (float): Size of each voxel.
        signed_distances (numpy.ndarray): signed distances at rest state

    Returns:
        numpy.ndarray: Voxel grid array of shape (W, H, D) with boolean values.

    """
    # Determine the extents of the point cloud
    min_coords = np.min(points, axis=0)-2e-1
    max_coords = np.max(points, axis=0)+2e-1

    print(min_coords, max_coords, voxel_size)

    # Compute the size of the voxel grid
    voxel_grid_size = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
    print(voxel_grid_size)

    x = np.arange(min_coords[0], min_coords[0] + voxel_grid_size[0] * voxel_size, voxel_size)
    y = np.arange(min_coords[1], min_coords[1] + voxel_grid_size[1] * voxel_size, voxel_size)
    z = np.arange(min_coords[2], min_coords[2] + voxel_grid_size[2] * voxel_size, voxel_size)

    print("premeshgrid", x.shape, y.shape, z.shape)
    # Create 3D coordinate arrays using meshgrid
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    print("postmeshgrid", x.shape, y.shape, z.shape)

    # Compute the nodes of the voxel grid
    # query points to get the interpolated sdf values
    voxel_node_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T    
    print(voxel_node_points.shape)
    # Interpolate the SDs at the voxel_node_points from the sample points
    voxel_node_sdfs = RBFInterpolator(points, signed_distances, kernel='linear')(voxel_node_points)

    print("Voxels min, max: ", np.min(voxel_node_sdfs), np.max(voxel_node_sdfs))
    voxel_grid = voxel_node_sdfs.reshape(x.shape)

    return voxel_grid, voxel_node_points, voxel_node_sdfs 

def rescale_and_recenter(points, bbox_min, bbox_max):
    # Calculate the current minimum and maximum values in each dimension
    current_min = np.min(points, axis=0)
    current_max = np.max(points, axis=0)

    # Calculate the scaling factors for each dimension
    scaling_factors = (bbox_max - bbox_min) / (current_max - current_min)

    # Rescale the points
    scaled_points = points * scaling_factors

    # Calculate the scaled minimum and maximum values in each dimension
    scaled_min = np.min(scaled_points, axis=0)
    scaled_max = np.max(scaled_points, axis=0)

    # Calculate the center of the current bounding box and target bounding box
    current_center = (scaled_min + scaled_max) / 2.0
    target_center = (bbox_min + bbox_max) / 2.0

    # Recenter the points
    recentered_points = scaled_points - current_center + target_center

    return recentered_points

#--------------------
def cauchy_strain(F):
  return 0.5*(torch.transpose(F,0,1) + F) - torch.eye(3).to(F.device)

def green_strain(F):
    return 0.5*(torch.matmul(torch.transpose(F,0,1), F) - torch.eye(3).to(device))
def E_pot(mu, lam, F):
  Eps = cauchy_strain(F)
  return mu*torch.trace(torch.matmul(torch.transpose(Eps,0,1),Eps)) + (lam/2)*torch.trace(Eps)*torch.trace(Eps)

def linear_elastic_E(mu, lam, F):
  Eps = cauchy_strain(F)
  return mu*torch.trace(torch.matmul(torch.transpose(Eps,0,1),Eps)) + (lam/2)*torch.trace(Eps)*torch.trace(Eps)

def simple_neohookean_E(mu, lam, F):
    I1 = torch.trace(torch.matmul(torch.transpose(F, 0,1), F))
    J = torch.det(F)
    return (mu/2.0)*(I1 - 3) - mu*torch.log(J) + (lam/2.0)*torch.log(J)*torch.log(J)

def neohookean_E2(mu, lam, F):
    J = torch.det(F)
    C = torch.matmul(torch.transpose(F, 0,1), F)
    IC = torch.trace(C)
    a = 1.0 + mu/lam 
    Ja = J- a 
    W = (mu/2.0)*(IC - 3.0) + (lam/2.0)*(Ja*Ja)
    return W

def neohookean_E(mu, lam, F):
    C1 = mu/2 
    D1 = lam/2
    I1 = torch.trace(torch.matmul(torch.transpose(F, 0,1), F))
    J = torch.det(F)
    #alpha = (1 + (C1/D1) - (C1/(D1*4)))
    W = C1*(I1 -3) + D1*(J-1)*(J-1) #- 0.5*C1*torch.log(I1 + 1)
    return W

#--------------------
def face_coords(verts, faces):
    coords = verts[faces]
    return coords

def cross(vec_A, vec_B):
    return torch.cross(vec_A, vec_B, dim=-1)

def face_area(verts, faces):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)
    return 0.5 * norm(raw_normal)

def norm(x, highdim=False):

    if(len(x.shape) == 1):
        raise ValueError("called norm() on single vector of dim " + str(x.shape) + " are you sure?")
    if(not highdim and x.shape[-1] > 4):
        raise ValueError("called norm() with large last dimension " + str(x.shape) + " are you sure?")

    return torch.norm(x, dim=len(x.shape)-1)

def sample_points_on_surface(verts, faces, n_pts, elem_vols):
    # Choose faces

    # Weight directly by vols to uniformly sample surface
    sample_probs = elem_vols
    sample_probs = torch.clamp(sample_probs, 1e-30, float('inf')) # avoid -eps area
    face_distrib = torch.distributions.categorical.Categorical(sample_probs)

    face_inds = face_distrib.sample(sample_shape=(n_pts,))

    # Get barycoords for each sample
    r1 = torch.rand(n_pts, device=verts.device)
    r2 = torch.rand(n_pts, device=verts.device)
    r3 = torch.rand(n_pts, device=verts.device)
    bary_vals = torch.zeros((n_pts, 4), device=verts.device)
    # bary_vals[:, 0] = r1
    # bary_vals[:, 1] = r2
    # bary_vals[:, 2] = r3
    # bary_vals[:, 3] = 1 - r1 - r2 - r3
    bary_vals = torch.rand((n_pts, 4), device=verts.device)
    # Normalize each row to ensure their sum is 1
    row_sums = torch.sum(bary_vals, dim=1, keepdim=True)
    bary_vals = bary_vals / row_sums
    print(bary_vals)
    # Get position in face for each sample
    coords = face_coords(verts, faces)
    sample_coords = coords[face_inds, :, :]
    sample_pos = torch.sum(bary_vals.unsqueeze(-1) * sample_coords, dim=1)
    print(coords.shape)
    print(bary_vals.unsqueeze(-1).shape)
    return sample_pos

def closest_points_indices(V, P):
    # Calculate the squared Euclidean distance between each point in P and all points in V
    distances = np.sum((V[:, np.newaxis, :] - P) ** 2, axis=2)

    # Find the indices of the rows in V that correspond to the closest points to P
    indices = np.argmin(distances, axis=0)

    return indices
#--------------------

# Define model
class StrandleWeightsMLP(nn.Module):
    # Models weighting function over implicit object for the strandle
    #  - f(point) = weight
    #  - input: 2D point
    #  - output: scalar weight at point
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(StrandleWeightsMLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ELU())

        for n in range(0, num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ELU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        strandle_weight = self.linear_relu_stack(x)
        return strandle_weight

class HandleModels():
    def __init__(self, num_handles, num_layers, layer_width, activation_func, dim, lr_start):
        self.num_handles = num_handles
        self.num_layers = num_layers 
        self.layer_width = layer_width
        self.activation_func = activation_func
        self.lr_start = lr_start
        
        if activation_func == "ELU":
            self.model = StrandleWeightsMLP(input_dim = dim, 
                                            hidden_dim = layer_width, 
                                            output_dim = num_handles, 
                                            num_layers = num_layers)
        
        elif activation_func == "SIREN":
            self.model = SirenNet(
                                dim_in = dim,                        # input dimension, ex. 2d coor
                                dim_hidden = layer_width,                  # hidden dimension
                                dim_out = num_handles,       # output dimension, ex. rgb value
                                num_layers = num_layers,                    # number of layers
                                w0_initial = 0.3                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
                            )
            exit()
        else:
            print("uknown activation function")
            exit()        
        self.optimizers =  torch.optim.Adam(self.model.parameters(), lr = lr_start)
        self.device = "cpu"

    # def getAllWeightsSoftmax(self, X):
    #     matrix = -100*torch.log(torch.norm(X[:, None, :] - self.P0[None, :, :], dim=-1) + 1e-9).to(device)
    #     # Find the indices of the maximum values along each row
    #     # Find the maximum values along each row
    #     max_values = torch.max(matrix, dim=1, keepdim=True).values

    #     # Create a mask where the maximum values are set to 1 and the rest to 0
    #     mask = torch.eq(matrix, max_values).float()

    #     # If there are repeated maximum values, select only one position by using torch.any
    #     has_max = torch.any(mask, dim=1, keepdim=True)
    #     mask *= has_max.float()
    #     return mask

    def updateLR(self, newLR):
        for g in self.optimizers.param_groups:
            g['lr'] = newLR
    
    def getAllWeightsSoftmax(self, X):
        MLP = self.model(X) 
        return MLP 

    def getAllGrads(self, W, X):
        #3d tensor: handles x samples x 2
        GG = torch.stack([torch.autograd.grad(torch.sum(W[:,i]), X, create_graph=True, allow_unused=True)[0] for i in range(W.shape[1])])
        return GG

    def train(self):
      self.model.train()
    
    def eval(self):
      self.model.eval()
    
    def optimizers_zero_grad(self):
      self.optimizers.zero_grad()

    def optimizers_step(self):
      self.optimizers.step()

    def to_device(self, dev):
        self.model = self.model.to(dev)
        self.device = dev
        self.optimizers = torch.optim.Adam(self.model.parameters(), lr = 1e-3)

def biased_random():
    #x = random.random()
    #probability of 0
    p = 0.5 #1 / (1 + math.exp(-10 * (x - 0.5)))
    if random.random() < p:
        return 0
    else:
        return 1
# Random transforms samples
# Index 0: Identity
# Index 1: Stretch by 5%
# Index 2: Rotation by 15 degrees
def getRandomDeltas(ind):
    if ind == 0:
        return 0
   
    if ind == 1 or ind == 2:
         #stretch x or stretch y
        if random.random() < 0.5:
            return (1, 0)
        else:
            return (2, 0)
    
    # if ind == 2:
    #     angle = math.radians(random.uniform(-40, 40))
    #     return (3, angle)

def getTransformsFromDeltas(d, num_handles):
    # return [np.array([[0, 0, 0, 0.2], 
    #                       [0, 0, 0, 0.2],
    #                       [0, 0, 0, 0.2]]) for i in range(num_handles)]
    if d==0:
        return [np.array([[np.random.randn(1)[0], np.random.randn(1)[0], np.random.randn(1)[0], 1*np.random.randn(1)[0]], 
                          [np.random.randn(1)[0], np.random.randn(1)[0], np.random.randn(1)[0], 1*np.random.randn(1)[0]],
                          [np.random.randn(1)[0], np.random.randn(1)[0], np.random.randn(1)[0], 1*np.random.randn(1)[0]]]) for i in range(num_handles)]
        
    else:
        return [np.array([[np.random.randn(1)[0], np.random.randn(1)[0], np.random.randn(1)[0], 1*np.random.randn(1)[0]], 
                          [np.random.randn(1)[0], np.random.randn(1)[0], np.random.randn(1)[0], 1*np.random.randn(1)[0]],
                          [np.random.randn(1)[0], np.random.randn(1)[0], np.random.randn(1)[0], 1*np.random.randn(1)[0]]]) for i in range(num_handles)]

    
def getBatchOfTs(num_handles, batch_size, epoch):
    batchTs = []
    for b in range(batch_size):
        Ts =getTransformsFromDeltas(biased_random(), num_handles)
        # Ts =getTransformsFromDeltas(-3, num_handles) #hard coded to Id deformations
        batchTs.append(Ts)
    return torch.tensor(np.array(batchTs))
        
def test(Handles, O, num_samples):
    Handles.eval()
    
    if num_samples == O.shape[0]:
        if num_samples>10000:
            num_samples = 10000
            random_batch_indices = torch.randint(low=0, high= O.shape[0], size=(num_samples,))
            X = O[random_batch_indices].float().to(Handles.device)
        else:
            X = torch.tensor(O, dtype=torch.float32, device=Handles.device)
        X.requires_grad = True
    else:
        if num_samples>10000:
            num_samples = 10000
        random_batch_indices = torch.randint(low=0, high= O.shape[0], size=(num_samples,))
        X = O[random_batch_indices].float().to(Handles.device)
        X.requires_grad = True

    # Compute prediction error
    W = Handles.getAllWeightsSoftmax(X)
    G = torch.transpose(Handles.getAllGrads(W, X), 0, 1)

    return W.cpu().detach().numpy(), X.cpu().detach().numpy(), G.cpu().detach().numpy()
