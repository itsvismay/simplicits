import igl
import torch
import numpy as np
from SimplicitHelpers import *
import os
import skimage
from PhysicsHelpers import *
# import potpourri3d as pp3d
import json
# from plyfile import PlyData

# SDFs
global SDBOXSIZE
global SPHERERAD
SDBOXSIZE = [1,0.1, 0.1]
SPHERERAD = 0.25

# General setup
def getDefaultTrainingSettings():
    return { "NumHandles":10, 
             "NumLayers":10, 
             "LayerWidth":64, 
             "ActivationFunc": "ELU", 
             "NumTrainingSteps": 20000, 
             "NumSamplePts": 1000, 
             "LRStart": 1e-4, 
             "LREnd": 1e-4,
             "TSamplingStdev": 1, 
             "TBatchSize": 10, 
             "LossCurveMovingAvgWindow": 100, 
             "SaveHandleIts": 1000, 
             "SaveSampleIts": 20000, 
             "NumSamplesToView": 5,
             "Timeit": True
    }

# SDFs
#------------------------
def sdSphere(p):
    global SPHERERAD
    return np.linalg.norm(p)-SPHERERAD

def sdLink(p):
    #parameters
    le = 0.2
    r1 = 0.21
    r2 = 0.1
    q = np.array([p[0], max(abs(p[1])-le,0.0), p[2]])
    return np.linalg.norm(np.array([np.linalg.norm(q[0:2])-r1, q[2]])) - r2

def sdBox(p):
    global SDBOXSIZE
    b = np.array(SDBOXSIZE)
    q = np.absolute(p) - b
    return  np.linalg.norm(np.array([max(q[0], 0.0), max(q[1], 0.0), max(q[2], 0.0)])) + min(max(q[0],max(q[1],q[2])),0.0)

def sdChain(p):
    def fract(x):
        return x - np.floor(x)

    # make a chain out of sdLink's
    a = np.copy(p); 
    a[1] = fract(a[1])-0.5
    b = np.copy(p); 
    b[1] = fract(b[1]+0.5) - 0.5
    
    # evaluate two links
    return min(sdLink(np.array([a[0], a[1], a[2]])), sdLink(np.array([b[2], b[1], b[0]])))

def distance_to_surface(P, mandelbulb_minimumDistanceToSurface, ITERATIONS, power):
    AO = 1.0
    externalBoundingRadius = 1.2
    internalBoundingRadius = 0.72
    derivative = 1.0
    Q = P.copy()
    
    for i in range(ITERATIONS):
        AO *= 0.725
        r = np.linalg.norm(Q)
        
        if r > 2.0:
            AO = min((AO + 0.075) * 4.1, 1.0)
            return min(np.linalg.norm(P) - internalBoundingRadius, 0.5 * np.log(r) * r / derivative)
        else:
            theta = np.arccos(Q[2] / r) * power
            phi = np.arctan2(Q[1], Q[0]) * power
            
            derivative = pow(r, power - 1.0) * power * derivative + 1.0
            
            sinTheta = np.sin(theta)
            cosTheta = np.cos(theta)
            cosPhi = np.cos(phi)
            sinPhi = np.sin(phi)
            
            Q = np.array([sinTheta * cosPhi,
                          sinTheta * sinPhi,
                          cosTheta]) * pow(r, power) + P
    
    return mandelbulb_minimumDistanceToSurface

def sdMandelbulb(p):
    scale = 0.6
    p *= 1.0 / scale
    return distance_to_surface(p, 0.0003, 8, 8.0) * scale

def DE(pos, Bailout=2., Iterations=5, Power=8):
    z = pos.copy()
    dr = 1.0
    r = 0.0
    for i in range(Iterations):
        r = np.linalg.norm(z)
        if r > Bailout:
            break
        
        theta = np.arccos(z[2] / r)
        phi = np.arctan2(z[1], z[0])
        dr = pow(r, Power - 1.0) * Power * dr + 1.0
        
        zr = pow(r, Power)
        theta = theta * Power
        phi = phi * Power
        
        z = zr * np.array([np.sin(theta) * np.cos(phi),
                           np.sin(theta) * np.sin(phi),
                           np.cos(theta)])
        z += pos
    
    return 0.5 * np.log(r) * r / dr

# Generators for different types
#------------------------
def generate_object_and_handles_from_sdf(name, num_handles, postprocess=False, yms=1e5):
    global SDBOXSIZE
    global SPHERERAD
    # 1. set fcn to the correct sdf fcn, for now just sdLink
    fcn = None
    bounds = [[-1,-1,-1], [1,1,1]]
    if name == "Link":
        fcn = sdLink
        bounds = [[-0.61,-0.61,-0.61], [0.61,0.61,0.61]]
    elif name == "Box":
        fcn = sdBox
        SDBOXSIZE = [1,1,1]
        bounds = [[-2,-2,-2], [2,2,2]]
    elif name == "BigBox":
        fcn = sdBox
        SDBOXSIZE = [10,10,10]
        bounds = [[-20,-20,-20], [20,20,20]]
    elif name == "ThinSheet":
        SDBOXSIZE = [3,0.4, 0.04]
        # bounds = [[-3,-0.5,-0.05], [3,0.5,0.05]]
        # fcn = sdBox
    elif name == "PaperSheet":
        SDBOXSIZE = [3,0.4, 0.04]
        bounds = [[-3,-0.5,-0.5], [3,0.5,0.5]]
        fcn = sdBox
    elif name == "ThickerSheet":
        SDBOXSIZE = [3, 0.4, 0.2]
        bounds = [[-3,-0.5,-0.5], [3,0.5,0.5]]
        fcn = sdBox
    elif name == "Mandelbulb":
        fcn = DE
        bounds = [[-2,-2,-2], [2,2,2]]
    elif name == "Sphere":
        fcn=sdSphere 
        bounds = [[-2,-2,-2], [2,2,2]]
    else:
        print("Undefined SDF")
        return

    # 2. uniformly sample bounding box of mesh, throw away sample points outside the mesh
    uniform_points = np.random.uniform(bounds[0], bounds[1], size=(500000, 3))
    sdf_vals = np.apply_along_axis(fcn, 1, uniform_points)
    keep_points = np.nonzero(sdf_vals <= 0)[0] # keep points where sd is not positive
    np_O = uniform_points[keep_points, :]
    np_O_sdfval = sdf_vals[keep_points]
    plot_implicit(np_O, np.zeros((2,3)), np_O_sdfval)

    YMs = yms*np.ones(np_O.shape[0])
    surfV = None
    surfF = None

    # 3. Marching cubes to recreate the surface mesh from sdf
    voxel_grid, voxel_pts, voxel_sdf = voxelize_point_cloud(uniform_points, 0.05, fcn, sdf_vals)
    tempV, surfF, normals, other = skimage.measure.marching_cubes(voxel_grid, level=0)
    interior_voxel_pts = voxel_pts[np.nonzero(voxel_sdf<=0)[0], :]
    surfV = rescale_and_recenter(tempV, np.min(interior_voxel_pts, axis=0), np.max(interior_voxel_pts, axis=0))
    
    # 3. Get approximate vol by summing uniformly sampled verts
    bb_vol = (np.max(uniform_points[:,0]) - np.min(uniform_points[:,0])) * (np.max(uniform_points[:,1]) - np.min(uniform_points[:,1])) * (np.max(uniform_points[:,2]) - np.min(uniform_points[:,2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample*np_O.shape[0]
    

    # 4. Generate handle points using furthest point sampling
    indices= np.arange(num_handles) #farthest_point_sampler(torch.tensor(np_O, device="cpu").unsqueeze(0), num_handles).flatten()
    np_P0_ = np_O[indices, :]
    np_P0_YM = YMs[indices]
    plot_implicit(uniform_points, np_P0_)

    # 5. Create and save object
    np_object = {"O": np_O,
                 "uniform_sampling_points": uniform_points, 
                 "uniform_sampling_sdf_vals": sdf_vals,
                 "P0": np_P0_,
                 "P0YM": np_P0_YM,
                 "surfV": surfV,
                 "surfF": surfF,
                 "surfYM": None,
                 "YM": YMs,
                 "poisson": 0.45,
                 "vol": appx_vol,
                 "name": name,
                 "indices": indices,
                 "MC_res": 0.02
                }
    if postprocess:
        return np_object
    else:
        if not os.path.exists(name):
            os.makedirs(name)
        torch.save(np_object, name + "/" +name+"-"+str(num_handles)+"-"+"object")
        return np_object

def generate_object_and_handles_from_surf_mesh(name, num_handles, postprocess = False, yms = 5e6):
    # 1. read surface mesh obj file
    surfV, _, _, surfF, _, _ = igl.read_obj(name+".obj")

    # 2. uniformly sample bounding box of mesh, throw away sample points outside the mesh
    uniform_points = np.random.uniform([np.min(surfV[:,0]), np.min(surfV[:,1]), np.min(surfV[:,2])], [np.max(surfV[:,0]), np.max(surfV[:,1]), np.max(surfV[:,2])], size=(10000, 3))
    sdf_vals, _, closest = igl.signed_distance(uniform_points, surfV, surfF)
    keep_points = np.nonzero(sdf_vals <= 0)[0] # keep points where sd is not positive
    np_O = uniform_points[keep_points, :]

    YMs = yms*np.ones(np_O.shape[0])
    # for i in range(5):
    #     start_split = -1e-2+np.min(np_O[:,0]) + i*(np.max(np_O[:,0]) - np.min(np_O[:,0]))/5.0
    #     end_split = start_split + (np.max(np_O[:,0]) - np.min(np_O[:,0]))/5.0
    #     print(start_split, end_split)
    #     start_bools = np_O[:,0]> start_split 
    #     end_bools = np_O[:,0]< end_split
    #     bools = np.logical_and(start_bools, end_bools)
    #     indices = np.nonzero(bools)[0]
    #     if i%2==1:
    #         YMs[indices] *= 0.01

    surfYMs = yms*np.ones(surfV.shape[0])
    # halfway = (np.max(surfV[:,0]) + np.min(surfV[:,0]))/2.0
    # indices = np.nonzero(surfV[:,0]> halfway)[0]
    # surfYMs[indices] *= 100

    # 3. Get approximate vol by summing uniformly sampled verts
    bb_vol = (np.max(surfV[:,0]) - np.min(surfV[:,0])) * (np.max(surfV[:,1]) - np.min(surfV[:,1])) * (np.max(surfV[:,2]) - np.min(surfV[:,2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample*np_O.shape[0]

    # 4. Generate handle points using furthest point sampling
    indices= np.arange(num_handles) #farthest_point_sampler(torch.tensor(np_O, device="cpu").unsqueeze(0), num_handles).flatten()
    np_P0_ = np_O[indices, :]
    np_P0_YM = YMs[indices]
    plot_implicit(np_O, np_P0_, YMs)

    # 5. Create and save object
    np_object = {"O": np_O,
                 "uniform_sampling_points": uniform_points, 
                 "uniform_sampling_sdf_vals": sdf_vals,
                 "P0": np_P0_,
                 "P0YM": np_P0_YM,
                 "surfV": surfV,
                 "surfF": surfF,
                 "surfYM": surfYMs,
                 "YM": YMs,
                 "poisson": 0.45,
                 "vol": appx_vol,
                }
    if postprocess:
        return np_object
    else:
        if not os.path.exists(name):
            os.makedirs(name)
        torch.save(np_object, name + "/" +name+"-"+str(num_handles)+"-"+"object")
        return np_object

def generate_object_and_handles_from_pointcloud(name, num_handles, postprocess=False):
    fnearname = "C:/Users/vismay/Downloads/samples/near/" + name + ".npz"
    frandname = "C:/Users/vismay/Downloads/samples/rand/" + name + ".npz"
    near_npz_file = np.load(fnearname)
    rand_npz_file = np.load(frandname)
    positions = np.concatenate((near_npz_file['position'][0::10,:], rand_npz_file['position']))
    distances = np.concatenate((near_npz_file['distance'][0::10,:], rand_npz_file['distance']))
    
    
    random_batch_indices = np.random.randint(low=0, high= positions.shape[0], size=(1000000,))
    uniform_points_wNans = positions[random_batch_indices,:]
    sdfs_wNans = distances[random_batch_indices, :]

    # Create a boolean mask to identify rows with NaN values
    nan_mask = np.logical_or(np.isnan(uniform_points_wNans).any(axis=1), sdfs_wNans[:,0]>1e3)

    # Remove rows with NaN values using boolean indexing
    uniform_points = uniform_points_wNans[~nan_mask]
    sdf_vals = sdfs_wNans[~nan_mask].squeeze()
    keep_indices = np.nonzero(sdf_vals <= 0.001)[0]
    np_O = uniform_points[keep_indices, :]
    np_O_sdf = sdf_vals[keep_indices]
    plot_implicit(np_O, np.zeros((2,3)), np_O_sdf)
     

    # 3. Marching cubes to recreate the surface mesh from sdf
    # random_batch_indices = np.random.randint(low=0, high= uniform_points.shape[0], size=(3000,))
    # fewer_uniform_points = uniform_points[random_batch_indices, :]
    # fewer_sdf_vals = sdf_vals[random_batch_indices]*1000000
    # # normalized_vector = 2 * (fewer_sdf_vals - np.min(fewer_sdf_vals)) / (np.max(fewer_sdf_vals) - np.min(fewer_sdf_vals)) - 1

    # voxel_grid, voxel_pts, voxel_sdf = interpolate_point_cloud(points = fewer_uniform_points, voxel_size = 0.05, signed_distances = fewer_sdf_vals)
    # reconstructed_V, reconstructed_F, normals, other = skimage.measure.marching_cubes(voxel_grid, level=0)
    # interior_voxel_pts = voxel_pts[np.nonzero(voxel_sdf<=0.001)[0], :]
    # reconstructed_V = rescale_and_recenter(reconstructed_V, np.min(interior_voxel_pts, axis=0), np.max(interior_voxel_pts, axis=0))
    # plot_implicit(fewer_uniform_points, np.zeros((2,3)), fewer_sdf_vals)
    # plot_mesh(reconstructed_V, reconstructed_F)

    # 3. Get approximate vol by summing uniformly sampled verts
    bb_vol = (np.max(uniform_points[:,0]) - np.min(uniform_points[:,0])) * (np.max(uniform_points[:,1]) - np.min(uniform_points[:,1])) * (np.max(uniform_points[:,2]) - np.min(uniform_points[:,2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample*np_O.shape[0]
    YMs = 1e4*np.ones(np_O.shape[0])

    # 4. Generate handle points using furthest point sampling
    indices= np.arange(num_handles) #farthest_point_sampler(torch.tensor(np_O, device="cpu").unsqueeze(0), num_handles).flatten()
    np_P0_ = np_O[indices.cpu().numpy(), :]
    np_P0_YM = YMs[indices.cpu().numpy()]

  
    # 5. Create and save object
    np_object = {"O": np_O,
                 "uniform_sampling_points": uniform_points, 
                 "uniform_sampling_sdf_vals": sdf_vals,
                 "P0": np_P0_,
                 "P0YM": np_P0_YM,
                 "surfV": None,
                 "surfF": None,
                 "surfYM": None,
                 "YM": YMs,
                 "poisson": 0.4,
                 "vol": appx_vol,
                 "name": name,
                 "P0_indices": indices
                }
    if postprocess:
        return np_object
    else:
        if not os.path.exists(name):
            os.makedirs(name)
        torch.save(np_object, name + "/" +name+"-"+str(num_handles)+"-"+"object")
        return np_object

def generate_object_and_handles_from_tet_mesh(name, num_handles):
    # 1. read surface mesh obj file
    Verts, _, _, Tets, _, _ = igl.read_obj(name+".obj")
    
    # 3. Get approximate vol by summing uniformly sampled verts
    appx_vols = igl.volume(Verts, Tets)
    
    # 2. uniformly sample bounding box of mesh, throw away sample points outside the mesh
    np_O = sample_points_on_surface(torch.tensor(Verts), torch.tensor(Tets), 100000, torch.tensor(appx_vols))

    

    # 4. Generate handle points using furthest point sampling
    indices= np.arange(num_handles) #farthest_point_sampler(torch.tensor(np_O, device="cpu").unsqueeze(0), num_handles).flatten()
    np_P0_ = np_O[indices.cpu().numpy(), :]
    plot_implicit(np_O, np_P0_)

    # 5. Create and save object
    np_object = {"O": np_O,
                 "P0": np_P0_,
                 "surfV": Verts,
                 "surfF": Tets,
                 "YM": 1e4*np.ones(),
                 "poisson": 0.4,
                 "vol": np.sum(appx_vols),
                }
    if not os.path.exists(name):
        os.makedirs(name)
    torch.save(np_object, name + "/" +name+"-"+str(num_handles)+"-"+"object")

def generate_mandelbulb(sim_obj_name, num_handles):
    np_object = generate_object_and_handles_from_sdf("Mandelbulb", num_handles=num_handles, yms=1e6)

def generate_tree_uniform(sim_obj_name, num_handles):
    np_object = generate_object_and_handles_from_pointcloud("Tree", num_handles=num_handles, postprocess=True)
    O = np_object["O"]
    YMs = np_object["YM"]
    P0inds = np_object["P0_indices"]

    #stiffen the trunk and branches
    boolTrunkY = O[:,1]<0.5
    boolTrunkX = np.logical_and(-0.19 < O[:,0], O[:,0] < 0.19) 
    boolTrunkZ = np.logical_and(-0.19 < O[:,2], O[:,2] < 0.19)
    boolAll = np.logical_and(np.logical_and(boolTrunkX, boolTrunkZ), boolTrunkY)
    branchInds = np.nonzero(boolAll)[0]
    YMs[branchInds] *= 1.0001
    np_object["YM"] = YMs 
    np_P0_YM = YMs[P0inds]
    np_object["P0_indices"] = np_P0_YM
    plot_implicit(O, np_object["P0"], YMs)
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+str(num_handles)+"-"+"object")

def generate_tree_trunks(sim_obj_name, num_handles):
    np_object = generate_object_and_handles_from_pointcloud("Tree", num_handles=num_handles, postprocess=True)
    O = np_object["O"]
    YMs = np_object["YM"]
    P0inds = np_object["P0_indices"]

    #stiffen the trunk and branches
    boolTrunkY = O[:,1]<0.5
    boolTrunkX = np.logical_and(-0.19 < O[:,0], O[:,0] < 0.19) 
    boolTrunkZ = np.logical_and(-0.19 < O[:,2], O[:,2] < 0.19)
    boolAll = np.logical_and(np.logical_and(boolTrunkX, boolTrunkZ), boolTrunkY)
    branchInds = np.nonzero(boolAll)[0]
    YMs[branchInds] *= 100
    np_object["YM"] = YMs 
    np_P0_YM = YMs[P0inds]
    np_object["P0_indices"] = np_P0_YM
    plot_implicit(O, np_object["P0"], YMs)
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+str(num_handles)+"-"+"object")

def generate_ct_object(sim_obj_name, num_handles):
    fname = "ExampleSetupScripts/Brain/ParaviewOutput/allfacesamplesinverted"
    allfacepc = torch.load(fname, map_location=torch.device("cpu"))
    #remove shoulder verts
    allfacepc = allfacepc[allfacepc[:, 2]<210, :]
    # allfacepc = allfacepc[allfacepc[:, 1]>130, :]   
    random_batch_indices = np.random.randint(low=0, high= allfacepc.shape[0], size=(int(allfacepc.shape[0]/50),))

    allfacepc = allfacepc[random_batch_indices,:]

    bone = allfacepc[allfacepc[:, 3]>0.1, :] # bone is darker colors (towards 0)
    softtemp = allfacepc[allfacepc[:, 3]<0.02, :] # soft tissue is lighter (towards 1)
    soft = softtemp[softtemp[:,3]>0.005,:]

    boneYM = 1e7*np.ones(bone.shape[0])
    softYM = 1e4*np.ones(soft.shape[0])
    print(bone.shape, soft.shape)

    np_O = np.concatenate((bone[:,0:3], soft[:, 0:3]), axis=0)/1000.0
    np_O = rotate_points_x(torch.tensor(np_O, device=device, dtype=torch.float32), -30, axis = 0).cpu().detach().numpy()

    np_YM = np.concatenate((boneYM, softYM)) 
    np_P0 = np_O[0:num_handles, :]
    np_P0_YM = np_YM[0:num_handles]


    # 3. Get approximate vol by summing uniformly sampled verts
    # Create a boolean mask for points within the bounding box
    # Define the two corner points of the bounding box
    min_corner = np.array([0.240, 0.26, 0.075])
    max_corner = np.array([0.290, 0.31, 0.125])
    mask = np.all((np_O >= min_corner) & (np_O <= max_corner), axis=1)
    # Count the number of points within the bounding box
    num_points_within_bbox = np.sum(mask)
    bbvol = 0.05*0.05*0.05
    bbvol_per_sample = bbvol/num_points_within_bbox


    appx_vol = bbvol_per_sample*np_O.shape[0]
    print("appx vol", appx_vol)

    # 5. Create and save object
    np_object = {"O": np_O,
                 "uniform_sampling_points": None, 
                 "uniform_sampling_sdf_vals": None,
                 "P0": np_P0,
                 "P0YM": np_P0_YM,
                 "surfV": None,
                 "surfF": None,
                 "surfYM": None,
                 "YM": np_YM,
                 "poisson": 0.4,
                 "vol": appx_vol,
                 "name": "skull",
                 "P0_indices": None
                }
    
    plot_implicit(np_O, np_object["P0"], np_YM)
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+str(num_handles)+"-"+"object")

def generate_link(sim_obj_name, num_handles):
    np_object = generate_object_and_handles_from_sdf("Link", num_handles=num_handles, postprocess=True)
    
    fcn = sdLink

    uniform_points = np_object["uniform_sampling_points"]
    sdf_vals = np_object["uniform_sampling_sdf_vals"]

    # 3. Marching cubes to recreate the surface mesh from sdf
    voxel_grid, voxel_pts, voxel_sdf = voxelize_point_cloud(uniform_points, 0.05, fcn, sdf_vals)
    tempV, surfF, normals, other = skimage.measure.marching_cubes(voxel_grid, level=0)
    interior_voxel_pts = voxel_pts[np.nonzero(voxel_sdf<=0)[0], :]
    surfV = rescale_and_recenter(tempV, np.min(interior_voxel_pts, axis=0), np.max(interior_voxel_pts, axis=0))
    surfYMs = np_object["YM"][0]*np.ones(surfV.shape[0])


    np_object["surfV"] = surfV
    np_object["surfF"] = surfF
    np_object["surfYM"] = surfYMs

    plot_mesh(surfV, surfF)

    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+str(num_handles)+"-"+"object")

def generate_spike(sim_obj_name, num_handles):
    np_object = generate_object_and_handles_from_pointcloud("Spike", num_handles=num_handles, postprocess=True)

    uniform_points = np_object["uniform_sampling_points"]
    sdf_vals = np_object["uniform_sampling_sdf_vals"]

    # 3. Marching cubes to recreate the surface mesh from sdf
    random_batch_indices = np.random.randint(low=0, high= uniform_points.shape[0], size=(10000,))
    uniform_points_sample = uniform_points[random_batch_indices,:]
    sdf_vals_sample = sdf_vals[random_batch_indices]
    voxel_grid, voxel_pts, voxel_sdf = interpolate_point_cloud(uniform_points_sample, 0.1, sdf_vals_sample)
    tempV, surfF, normals, other = skimage.measure.marching_cubes(voxel_grid, level=0)
    interior_voxel_pts = voxel_pts[np.nonzero(voxel_sdf<=0)[0], :]
    surfV = rescale_and_recenter(tempV, np.min(interior_voxel_pts, axis=0), np.max(interior_voxel_pts, axis=0))
    surfYMs = np_object["YM"][0]*np.ones(surfV.shape[0])


    np_object["surfV"] = surfV
    np_object["surfF"] = surfF
    np_object["surfYM"] = surfYMs

    plot_mesh(surfV, surfF)
    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+str(num_handles)+"-"+"object")

def generate_heterogeneous_monkey(sim_obj_name, num_handles):
    np_object = generate_object_and_handles_from_surf_mesh("Monkey", num_handles=5, postprocess=True, yms = 1e4)
    O = np_object["O"]
    YMs = np_object["YM"]

    #stiffen the trunk and branches
    boolAll =  O[:,0]<0.0
    branchInds = np.nonzero(boolAll)[0]
    YMs[branchInds] *= 1000
    np_object["YM"] = YMs 
    plot_implicit(O, np_object["P0"], YMs)
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+str(num_handles)+"-"+"object")

def generate_simple_skull(sim_obj_name, num_handles, ym1=1e4, ym2=1e7):
    global SPHERERAD
    SPHERERAD = 0.3
   
    fcn = sdSphere
    bounds = [[-1,-1,-1], [1,1,1]]

    # 2. uniformly sample bounding box of mesh, throw away sample points outside the mesh
    uniform_points = np.random.uniform(bounds[0], bounds[1], size=(500000, 3))
    sdf_vals = np.apply_along_axis(fcn, 1, uniform_points)
    brain_segment = np.nonzero(sdf_vals <= 0)[0]
    skull_segment = np.nonzero((sdf_vals < 0.03) & (sdf_vals > 0))[0]

    np_O = uniform_points[np.concatenate((brain_segment, skull_segment), axis=0), :]
    np_O_sdfval = sdf_vals[np.concatenate((brain_segment, skull_segment), axis=0)]
    print(brain_segment.shape, skull_segment.shape)

    YMs = np.ones(np_O.shape[0])
    surfV = None
    surfF = None

    voxel_grid, voxel_pts, voxel_sdf = voxelize_point_cloud(uniform_points, 0.05, fcn, sdf_vals)
    tempV, surfF, normals, other = skimage.measure.marching_cubes(voxel_grid, level=0)
    interior_voxel_pts = voxel_pts[np.nonzero(voxel_sdf<=0)[0], :]
    surfV = rescale_and_recenter(tempV, np.min(interior_voxel_pts, axis=0), np.max(interior_voxel_pts, axis=0))
    plot_mesh(surfV, surfF)
    igl.write_triangle_mesh(sim_obj_name + "/" +sim_obj_name+".obj", surfV, surfF)

    YMs[:brain_segment.shape[0]] *= 1e4
    YMs[brain_segment.shape[0]:] *= 1e7
    plot_implicit(np_O, np.zeros((2,3)), YMs)

    # 3. Get approximate vol by summing uniformly sampled verts
    bb_vol = (np.max(uniform_points[:,0]) - np.min(uniform_points[:,0])) * (np.max(uniform_points[:,1]) - np.min(uniform_points[:,1])) * (np.max(uniform_points[:,2]) - np.min(uniform_points[:,2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample*np_O.shape[0]
    

    # 4. Generate handle points using furthest point sampling
    indices= np.arange(num_handles) #farthest_point_sampler(torch.tensor(np_O, device="cpu").unsqueeze(0), num_handles).flatten()
    np_P0_ = np_O[indices, :]
    np_P0_YM = YMs[indices]
    plot_implicit(uniform_points, np_P0_)

    # 5. Create and save object
    np_object = {"O": np_O,
                 "uniform_sampling_points": uniform_points, 
                 "uniform_sampling_sdf_vals": sdf_vals,
                 "P0": np_P0_,
                 "P0YM": np_P0_YM,
                 "surfV": surfV,
                 "surfF": surfF,
                 "surfYM": None,
                 "YM": YMs,
                 "poisson": 0.45,
                 "vol": appx_vol,
                 "name": sim_obj_name,
                 "indices": indices,
                 "MC_res": 0.02
                }
    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+str(num_handles)+"-"+"object")

def generate_from_subspacemfem(sim_obj_name, num_handles):
    ###########################
    np_X, np_T, np_F = igl.read_mesh(sim_obj_name + "/" +sim_obj_name+"-"+str(num_handles)+ "-dir/"+"MESH.mesh")
    np_YM = 1e7*np.ones((np_X.shape[0],1))
    np_PR = 0.45

    # Compute the Euclidean distance from each point to the origin
    # Find the indices of points within the specified radius
    distances = np.linalg.norm(np_X, axis=1)
    indices_within_radius = np.where(distances <= 0.2)[0]
    np_YM[indices_within_radius,:] /= 1000

    VOL = torch.tensor(np.sum(igl.volume(np_X, np_T)), dtype=torch.float32, device=device)
    MUS = torch.tensor(np_YM/(2*(1+np_PR)), dtype=torch.float32, device=device) 
    LAMS = torch.tensor(np_YM*np_PR/((1+np_PR)*(1-2*np_PR)), dtype=torch.float32, device=device) 

    ###########################

    # # 2. uniformly sample bounding box of mesh, throw away sample points outside the mesh
    # uniform_points = np.random.uniform(bounds[0], bounds[1], size=(500000, 3))
    # sdf_vals = np.apply_along_axis(fcn, 1, uniform_points)
    # brain_segment = np.nonzero(sdf_vals <= 0)[0]
    # skull_segment = np.nonzero((sdf_vals < 0.05) & (sdf_vals > 0))[0]

    np_O = np_X
    np_O_sdfval = None
    

    YMs = np_YM.flatten()
    surfV = np_X
    surfF = np_F

    plot_implicit(np_O, np.zeros((2,3)), YMs)

    # 3. Get approximate vol by summing uniformly sampled verts
    appx_vol = VOL
    

    # 4. Generate handle points using furthest point sampling
    indices= np.arange(num_handles) #farthest_point_sampler(torch.tensor(np_O, device="cpu").unsqueeze(0), num_handles).flatten()
    np_P0_ = np_O[indices, :]
    np_P0_YM = YMs[indices]

    # 5. Create and save object
    np_object = {"O": np_O,
                 "uniform_sampling_points": None, 
                 "uniform_sampling_sdf_vals": None,
                 "P0": np_P0_,
                 "P0YM": np_P0_YM,
                 "surfV": surfV,
                 "surfF": surfF,
                 "surfYM": None,
                 "YM": YMs,
                 "poisson": 0.45,
                 "vol": appx_vol,
                 "name": sim_obj_name,
                 "indices": indices,
                 "MC_res": 0.02
                }
    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+str(num_handles)+"-"+"object")

def generate_heterogeneous_beam(sim_obj_name, num_handles):
    np_object = generate_object_and_handles_from_surf_mesh("511BeamELU", num_handles=num_handles, postprocess=True, yms=1e4)
    O = np_object["O"]
    YMs = np_object["YM"]

    #update YMs
    selection =  O[:,0]<3.5
    selectionIdxs = np.nonzero(selection)[0]
    YMs[selectionIdxs] *= 10
    plot_implicit(O, np_object["P0"], YMs)

    selection2 =  O[:,0]<1.5
    selectionIdxs2 = np.nonzero(selection2)[0]
    YMs[selectionIdxs2] *= 10

    np_object["YM"] = YMs 
    plot_implicit(O, np_object["P0"], YMs)

    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+str(num_handles)+"-"+"object")

def generate_sphere(sim_obj_name, num_handles):
    global SPHERERAD
    SPHERERAD = 0.5
   
    fcn = sdSphere
    bounds = [[-1,-1,-1], [1,1,1]]

    # 2. uniformly sample bounding box of mesh, throw away sample points outside the mesh
    uniform_points = np.random.uniform(bounds[0], bounds[1], size=(500000, 3))
    sdf_vals = np.apply_along_axis(fcn, 1, uniform_points)
    brain_segment = np.nonzero(sdf_vals <= 0)[0]

    np_O = uniform_points[brain_segment, :]
    np_O_sdfval = sdf_vals[brain_segment]

    YMs = 1e7*np.ones(np_O.shape[0])
    surfV = None
    surfF = None

    voxel_grid, voxel_pts, voxel_sdf = voxelize_point_cloud(uniform_points, 0.05, fcn, sdf_vals)
    tempV, surfF, normals, other = skimage.measure.marching_cubes(voxel_grid, level=0)
    interior_voxel_pts = voxel_pts[np.nonzero(voxel_sdf<=0)[0], :]
    surfV = rescale_and_recenter(tempV, np.min(interior_voxel_pts, axis=0), np.max(interior_voxel_pts, axis=0))
    plot_mesh(surfV, surfF)
    igl.write_triangle_mesh(sim_obj_name + "/" +sim_obj_name+".obj", surfV, surfF)


    plot_implicit(np_O, np.zeros((2,3)), YMs)

    # 3. Get approximate vol by summing uniformly sampled verts
    bb_vol = (np.max(uniform_points[:,0]) - np.min(uniform_points[:,0])) * (np.max(uniform_points[:,1]) - np.min(uniform_points[:,1])) * (np.max(uniform_points[:,2]) - np.min(uniform_points[:,2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample*np_O.shape[0]
    

    # 4. Generate handle points using furthest point sampling
    indices= np.arange(num_handles) #farthest_point_sampler(torch.tensor(np_O, device="cpu").unsqueeze(0), num_handles).flatten()
    np_P0_ = np_O[indices, :]
    np_P0_YM = YMs[indices]
    plot_implicit(uniform_points, np_P0_)

    # 5. Create and save object
    np_object = {"O": np_O,
                 "uniform_sampling_points": uniform_points, 
                 "uniform_sampling_sdf_vals": sdf_vals,
                 "P0": np_P0_,
                 "P0YM": np_P0_YM,
                 "surfV": surfV,
                 "surfF": surfF,
                 "surfYM": None,
                 "YM": YMs,
                 "density": 1000,
                 "poisson": 0.45,
                 "vol": appx_vol,
                 "name": sim_obj_name,
                 "indices": indices,
                 "MC_res": 0.02
                }
    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+str(num_handles)+"-"+"object")

## NEW OBJECT FORMAT

def generate_from_sdf(name, yms = 1e4, prs = 0.45, rhos = 1000, surf = False, show_plot=True):
    global SDBOXSIZE
    global SPHERERAD
    # 1. set fcn to the correct sdf fcn, for now just sdLink
    fcn = None
    bounds = [[-1,-1,-1], [1,1,1]]
    if name == "Link":
        fcn = sdLink
        bounds = [[-0.61,-0.61,-0.61], [0.61,0.61,0.61]]
    elif name == "Box":
        fcn = sdBox
        SDBOXSIZE = [1,1,1]
        bounds = [[-2,-2,-2], [2,2,2]]
    elif name == "BigBox":
        fcn = sdBox
        SDBOXSIZE = [10,10,10]
        bounds = [[-20,-20,-20], [20,20,20]]
    elif name == "ThinSheet":
        SDBOXSIZE = [3,0.4, 0.04]
        # bounds = [[-3,-0.5,-0.05], [3,0.5,0.05]]
        # fcn = sdBox
    elif name == "PaperSheet":
        SDBOXSIZE = [3,0.4, 0.04]
        bounds = [[-3,-0.5,-0.5], [3,0.5,0.5]]
        fcn = sdBox
    elif name == "ThickerSheet":
        SDBOXSIZE = [3, 0.4, 0.2]
        bounds = [[-3,-0.5,-0.5], [3,0.5,0.5]]
        fcn = sdBox
    elif name == "Mandelbulb":
        fcn = DE
        bounds = [[-2,-2,-2], [2,2,2]]
    elif name == "Sphere":
        fcn=sdSphere 
        bounds = [[-2,-2,-2], [2,2,2]]
    else:
        print("Undefined SDF")
        return

    # 2. uniformly sample bounding box of mesh, throw away sample points outside the mesh
    uniform_points = np.random.uniform(bounds[0], bounds[1], size=(500000, 3))
    sdf_vals = np.apply_along_axis(fcn, 1, uniform_points)
    keep_points = np.nonzero(sdf_vals <= 0)[0] # keep points where sd is not positive
    np_O = uniform_points[keep_points, :]
    np_O_sdfval = sdf_vals[keep_points]
    if show_plot:
        plot_implicit(np_O, np_O_sdfval)

    YMs = yms*np.ones(np_O.shape[0])
    PRs = prs*np.ones_like(YMs)
    Rhos = rhos*np.ones_like(YMs)
    surfV = None
    surfF = None

    if surf:
        # 3. Marching cubes to recreate the surface mesh from sdf
        voxel_grid, voxel_pts, voxel_sdf = voxelize_point_cloud(uniform_points, 0.1, fcn, sdf_vals)
        tempV, surfF, normals, other = skimage.measure.marching_cubes(voxel_grid, level=0)
        interior_voxel_pts = voxel_pts[np.nonzero(voxel_sdf<=0)[0], :]
        surfV = rescale_and_recenter(tempV, np.min(interior_voxel_pts, axis=0), np.max(interior_voxel_pts, axis=0))
    
    # 3. Get approximate vol by summing uniformly sampled verts
    bb_vol = (np.max(uniform_points[:,0]) - np.min(uniform_points[:,0])) * (np.max(uniform_points[:,1]) - np.min(uniform_points[:,1])) * (np.max(uniform_points[:,2]) - np.min(uniform_points[:,2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample*np_O.shape[0]
    

    # 5. Create and save object
    np_object = {"Name":type, 
                 "Dim": 3, 
                 "BoundingBoxSamplePts": uniform_points, 
                 "BoundingBoxSignedDists": sdf_vals,
                 "ObjectSamplePts": np_O, 
                 "ObjectYMs": YMs, 
                 "ObjectPRs": PRs, 
                 "ObjectRho": Rhos, 
                 "ObjectColors": None,
                 "ObjectVol": appx_vol, 
                 "SurfV": surfV, 
                 "SurfF": surfF, 
                 "MarchingCubesRes": -1
                }
    return np_object
    
def generate_ct_bladder(sim_obj_name):
    fname = "ExampleSetupScripts/Bladder/BladderCloud.obj"
    bladder, _, _, _, _, _ = igl.read_obj(fname)
    
    # #downsample
    # random_batch_indices = np.random.randint(low=0, high= bladder.shape[0], size=(int(bladder.shape[0]),))
    # bladder = bladder[random_batch_indices,:]/1000
    print(bladder)
    bladder = bladder*10
    print(bladder)
    bladderYM = 10000*np.ones(bladder.shape[0])
    bladderPR = 0.45*np.ones(bladder.shape[0])
    bladderRho = 100*np.ones(bladder.shape[0])

    np_O = bladder

    plot_implicit(bladder, bladderYM)


    # 3. Get approximate vol by summing uniformly sampled verts
    # Create a boolean mask for points within the bounding box
    # Define the two corner points of the bounding box
    min_corner = np.array([0.234, 0.0787, -0.286])*10
    max_corner = np.array([0.257, 0.10875, -0.249])*10
    mask = np.all((np_O >= min_corner) & (np_O <= max_corner), axis=1)

    # Count the number of points within the bounding box
    num_points_within_bbox = np.sum(mask)

    bbvol = np.abs(max_corner[0] - min_corner[0])*np.abs(max_corner[1] - min_corner[1])*np.abs(max_corner[2] - min_corner[2])
    bbvol_per_sample = bbvol/num_points_within_bbox


    appx_vol = bbvol_per_sample*np_O.shape[0]
    print(bbvol, appx_vol)

    # 5. Create and save object
    np_object = {"Name":sim_obj_name, 
                 "Dim": 3, 
                 "BoundingBoxSamplePts": None, 
                 "BoundingBoxSignedDists": None,
                 "ObjectSamplePts": bladder, 
                 "ObjectYMs": bladderYM, 
                 "ObjectPRs": bladderPR, 
                 "ObjectRho": bladderRho, 
                 "ObjectColors": None,
                 "ObjectVol": appx_vol, 
                 "SurfV": None, 
                 "SurfF": None, 
                 "MarchingCubesRes": -1
                }
    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+"object")
    
    if not os.path.exists(sim_obj_name+"/"+sim_obj_name+"-training-settings.json"):
        training_dict = getDefaultTrainingSettings()
        training_dict["LRStart"] = 1e-3
        training_dict["LREnd"] = 1e-4
        json_object = json.dumps(training_dict, indent=4)
        # Writing to sample.json
        with open(sim_obj_name+"/"+sim_obj_name+"-training-settings.json", "w") as outfile:
            outfile.write(json_object)

def generate_clean_ct_skullbrain(sim_obj_name):
    fname = "ExampleSetupScripts/Brain/ParaviewOutput/allfacesamplesinverted"
    allfacepc = np.array(torch.load(fname, map_location=torch.device("cpu")))
    #remove shoulder verts
    allfacepc = allfacepc[allfacepc[:, 2]<140, :]
    # allfacepc = allfacepc[allfacepc[:, 1]>130, :]   
    random_batch_indices = np.random.randint(low=0, high= allfacepc.shape[0], size=(int(allfacepc.shape[0]/5),))
    allfacepc = allfacepc[random_batch_indices,:]

    # 1. Center points at origin and rescale
    allfacepc[:, 0:3] = allfacepc[:, 0:3]/500
    midpoint = np.mean(allfacepc[:,0:3], axis=0)
    centered_points = allfacepc[:,0:3] - midpoint
    allfacepc[:,0:3] = centered_points
    # 2. Remove points past the skull
    global SPHERERAD
    SPHERERAD = 0.14
    fcn = sdSphere
    # 2. uniformly sample bounding box of mesh, throw away sample points outside the mesh
    sdf_vals = np.apply_along_axis(fcn, 1,allfacepc[:,0:3])
    skull_segment1 = sdf_vals > 0
    brain_segment1 = allfacepc[:,3]<0.3
    skull_segment2 = np.logical_not(brain_segment1)
    
    skull_segment = np.logical_or(skull_segment1, skull_segment2)
    brain_segment = np.logical_not(skull_segment)
    bone = allfacepc[skull_segment, :]
    soft = allfacepc[brain_segment,:]

    boneYM = 1e7*np.ones(bone.shape[0])
    softYM = 1e3*np.ones(soft.shape[0])

    np_O = np.concatenate((bone[:,0:3], soft[:, 0:3]), axis=0)
    np_O = rotate_points_x(torch.tensor(np_O, device=device, dtype=torch.float32), -30, axis = 0).cpu().detach().numpy()

    np_YM = np.concatenate((boneYM, softYM)) 
    np_PRs = 0.45*np.ones_like(np_YM)
    np_Rhos = 1000*np.ones_like(np_YM)

    plot_implicit(np_O, np_YM)

    # 3. Get approximate vol by summing uniformly sampled verts
    # Create a boolean mask for points within the bounding box
    # Define the two corner points of the bounding box
    min_corner = np.array([-0.1, -0.1, -0.1]) 
    max_corner = np.array([0.1, 0.1, 0.1]) 
    mask = np.all((np_O >= min_corner) & (np_O <= max_corner), axis=1)
    # Count the number of points within the bounding box
    num_points_within_bbox = np.sum(mask)
    bbvol = np.abs(max_corner[0] - min_corner[0])*np.abs(max_corner[1] - min_corner[1])*np.abs(max_corner[2] - min_corner[2])
    bbvol_per_sample = bbvol/num_points_within_bbox
    appx_vol = bbvol_per_sample*np_O.shape[0]
    print(bbvol, num_points_within_bbox, appx_vol)

    # 5. Create and save object
    np_object = {"Name":sim_obj_name, 
                 "Dim": 3, 
                 "BoundingBoxSamplePts": None, 
                 "BoundingBoxSignedDists": None,
                 "ObjectSamplePts": np_O, 
                 "ObjectYMs": np_YM, 
                 "ObjectPRs": np_PRs, 
                 "ObjectRho": np_Rhos, 
                 "ObjectColors": None,
                 "ObjectVol": appx_vol, 
                 "SurfV": None, 
                 "SurfF": None, 
                 "MarchingCubesRes": -1
                }
    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+"object")
    
    if not os.path.exists(sim_obj_name+"/"+sim_obj_name+"-training-settings.json"):
        training_dict = getDefaultTrainingSettings()
        json_object = json.dumps(training_dict, indent=4)
        # Writing to sample.json
        with open(sim_obj_name+"/"+sim_obj_name+"-training-settings.json", "w") as outfile:
            outfile.write(json_object)

def generate_ct_skullstripped_brain(sim_obj_name):
    fname = "CTSkullPC.obj"
    skull, _, _, _, _, _ = igl.read_obj(fname)
    bone = skull[0::20,:]
    # bone = bone[bone[:, 2]<65, :]

    #resize to a reasonable head size
    bone = bone/500

    # 1. read surface mesh obj file
    fname = "CTBrainSurface.obj"
    surfV, _, _, surfF, _, _ = igl.read_obj(fname)
    surfV = surfV/550

    # 2. uniformly sample bounding box of mesh, throw away sample points outside the mesh
    uniform_points = np.random.uniform([np.min(surfV[:,0]), np.min(surfV[:,1]), np.min(surfV[:,2])], [np.max(surfV[:,0]), np.max(surfV[:,1]), np.max(surfV[:,2])], size=(50000, 3))
    sdf_vals, _, closest = igl.signed_distance(uniform_points, surfV, surfF)
    keep_points = np.nonzero(sdf_vals <= 0)[0] # keep points where sd is not positive

    soft = uniform_points[keep_points, :]

    #rearrance brain a bit
    soft[:,1] -= 0.015
    soft[:,2] -= 0.015

    boneYM = 1e7*np.ones(bone.shape[0])
    softYM = 1e4*np.ones(soft.shape[0])


    np_O = np.concatenate((bone[:,0:3], soft[:, 0:3]), axis=0)
    # np_O = rotate_points_x(torch.tensor(np_O, device=device, dtype=torch.float32), -30, axis = 0).cpu().detach().numpy()

    np_YM = np.concatenate((boneYM, softYM)) 
    np_PRs = 0.45*np.ones_like(np_YM)
    np_Rhos = 1000*np.ones_like(np_YM)

    plot_implicit(np_O, np_YM)

    # 3. Get approximate vol by summing uniformly sampled verts
    bbvol = (np.max(surfV[:,0]) - np.min(surfV[:,0])) * (np.max(surfV[:,1]) - np.min(surfV[:,1])) * (np.max(surfV[:,2]) - np.min(surfV[:,2]))
    vol_per_sample = bbvol / uniform_points.shape[0]
    appx_vol = vol_per_sample*np_O.shape[0]
    print(bbvol, appx_vol)

    # 5. Create and save object
    np_object = {"Name":sim_obj_name, 
                 "Dim": 3, 
                 "BoundingBoxSamplePts": None, 
                 "BoundingBoxSignedDists": None,
                 "ObjectSamplePts": np_O, 
                 "ObjectYMs": np_YM, 
                 "ObjectPRs": np_PRs, 
                 "ObjectRho": np_Rhos, 
                 "ObjectColors": None,
                 "ObjectVol": appx_vol, 
                 "SurfV": None, 
                 "SurfF": None, 
                 "MarchingCubesRes": -1
                }
    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+"object")
    
    if not os.path.exists(sim_obj_name+"/"+sim_obj_name+"-training-settings.json"):
        training_dict = getDefaultTrainingSettings()
        json_object = json.dumps(training_dict, indent=4)
        # Writing to sample.json
        with open(sim_obj_name+"/"+sim_obj_name+"-training-settings.json", "w") as outfile:
            outfile.write(json_object)

def generate_ct_abdomen(sim_obj_name):
    fname = "ExampleSetupScripts/Abdomen/Abdomen.obj"
    fullabdomen, _, _, _, _, _ = igl.read_obj(fname)
    
    
    abdomen = fullabdomen[0::100,:]/100/5.0

    # #downsample
    # random_batch_indices = np.random.randint(low=0, high= abdomen.shape[0], size=(int(abdomen.shape[0]),))
    # abdomen = abdomen[random_batch_indices,:]/1000

    abdomenYM = 1000*np.ones(abdomen.shape[0])
    abdomenPR = 0.45*np.ones(abdomen.shape[0])
    abdomenRho = 1000*np.ones(abdomen.shape[0])

    np_O = abdomen

    plot_implicit(abdomen, abdomenYM)


    # 3. Get approximate vol by summing uniformly sampled verts
    # Create a boolean mask for points within the bounding box
    # Define the two corner points of the bounding box
    min_corner = np.array([0.87, 1.66, 0])/5.0
    max_corner = np.array([1.76, 3.25, 1.48])/5.0
    mask = np.all((np_O >= min_corner) & (np_O <= max_corner), axis=1)

    # Count the number of points within the bounding box
    num_points_within_bbox = np.sum(mask)

    bbvol = np.abs(max_corner[0] - min_corner[0])*np.abs(max_corner[1] - min_corner[1])*np.abs(max_corner[2] - min_corner[2])
    bbvol_per_sample = bbvol/num_points_within_bbox


    appx_vol = bbvol_per_sample*abdomen.shape[0]
    print(bbvol, bbvol_per_sample, appx_vol)

    # 5. Create and save object
    np_object = {"Name":sim_obj_name, 
                 "Dim": 3, 
                 "BoundingBoxSamplePts": None, 
                 "BoundingBoxSignedDists": None,
                 "ObjectSamplePts": abdomen, 
                 "ObjectYMs": abdomenYM, 
                 "ObjectPRs": abdomenPR, 
                 "ObjectRho": abdomenRho, 
                 "ObjectColors": None,
                 "ObjectVol": appx_vol, 
                 "SurfV": None, 
                 "SurfF": None, 
                 "MarchingCubesRes": -1
                }
    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+"object")
    
    if not os.path.exists(sim_obj_name+"/"+sim_obj_name+"-training-settings.json"):
        training_dict = getDefaultTrainingSettings()
        training_dict["LRStart"] = 1e-3
        training_dict["LREnd"] = 1e-4
        json_object = json.dumps(training_dict, indent=4)
        # Writing to sample.json
        with open(sim_obj_name+"/"+sim_obj_name+"-training-settings.json", "w") as outfile:
            outfile.write(json_object)

def generate_simple_skullbrain(sim_obj_name):
    global SPHERERAD
    SPHERERAD = 0.3
   
    fcn = sdSphere
    bounds = [[-1,-1,-1], [1,1,1]]

    # 2. uniformly sample bounding box of mesh, throw away sample points outside the mesh
    uniform_points = np.random.uniform(bounds[0], bounds[1], size=(500000, 3))
    sdf_vals = np.apply_along_axis(fcn, 1, uniform_points)
    brain_segment = np.nonzero(sdf_vals <= 0)[0]
    skull_segment = np.nonzero((sdf_vals < 0.02) & (sdf_vals > 0))[0]

    np_O = uniform_points[np.concatenate((brain_segment, skull_segment), axis=0), :]
    print(brain_segment.shape, skull_segment.shape)

    YMs = np.ones(np_O.shape[0])
    surfV = None
    surfF = None


    YMs[:brain_segment.shape[0]] *= 1e3
    YMs[brain_segment.shape[0]:] *= 1e8
    PRs = 0.45*np.ones_like(YMs)
    Rhos = 1000*np.ones_like(YMs)
    plot_implicit(np_O, YMs)

    # 3. Get approximate vol by summing uniformly sampled verts
    bb_vol = (np.max(uniform_points[:,0]) - np.min(uniform_points[:,0])) * (np.max(uniform_points[:,1]) - np.min(uniform_points[:,1])) * (np.max(uniform_points[:,2]) - np.min(uniform_points[:,2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample*np_O.shape[0]
    

    plot_implicit(uniform_points)

    # 5. Create and save object
    np_object = {"Name":sim_obj_name, 
                 "Dim": 3, 
                 "BoundingBoxSamplePts": None, 
                 "BoundingBoxSignedDists": None,
                 "ObjectSamplePts": np_O, 
                 "ObjectYMs": YMs, 
                 "ObjectPRs": PRs, 
                 "ObjectRho": Rhos, 
                 "ObjectColors": None,
                 "ObjectVol": appx_vol, 
                 "SurfV": None, 
                 "SurfF": None, 
                 "MarchingCubesRes": -1
                }
    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+"object")
    
    if not os.path.exists(sim_obj_name+"/"+sim_obj_name+"-training-settings.json"):
        training_dict = getDefaultTrainingSettings()
        json_object = json.dumps(training_dict, indent=4)
        # Writing to sample.json
        with open(sim_obj_name+"/"+sim_obj_name+"-training-settings.json", "w") as outfile:
            outfile.write(json_object)

def generate_layered_sphere(sim_obj_name):
    global SPHERERAD
    SPHERERAD = 0.3
   
    fcn = sdSphere
    bounds = [[-1,-1,-1], [1,1,1]]

    # 2. uniformly sample bounding box of mesh, throw away sample points outside the mesh
    uniform_points = np.random.uniform(bounds[0], bounds[1], size=(500000, 3))
    sdf_vals = np.apply_along_axis(fcn, 1, uniform_points)
    brain_segment = np.nonzero(((sdf_vals>0.04) & (sdf_vals<0.08)) | (sdf_vals < 0))[0]
    skull_segment = np.nonzero((sdf_vals < 0.04) & (sdf_vals > 0))[0]

    np_O = uniform_points[np.concatenate((brain_segment, skull_segment), axis=0), :]
    print(brain_segment.shape, skull_segment.shape)

    YMs = np.ones(np_O.shape[0])
    surfV = None
    surfF = None


    YMs[:brain_segment.shape[0]] *= 1e3
    YMs[brain_segment.shape[0]:] *= 1e8
    PRs = 0.45*np.ones_like(YMs)
    Rhos = 1000*np.ones_like(YMs)
    plot_implicit(np_O, YMs)

    # 3. Get approximate vol by summing uniformly sampled verts
    bb_vol = (np.max(uniform_points[:,0]) - np.min(uniform_points[:,0])) * (np.max(uniform_points[:,1]) - np.min(uniform_points[:,1])) * (np.max(uniform_points[:,2]) - np.min(uniform_points[:,2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample*np_O.shape[0]
    

    plot_implicit(uniform_points)

    # 5. Create and save object
    np_object = {"Name":sim_obj_name, 
                 "Dim": 3, 
                 "BoundingBoxSamplePts": None, 
                 "BoundingBoxSignedDists": None,
                 "ObjectSamplePts": np_O, 
                 "ObjectYMs": YMs, 
                 "ObjectPRs": PRs, 
                 "ObjectRho": Rhos, 
                 "ObjectColors": None,
                 "ObjectVol": appx_vol, 
                 "SurfV": None, 
                 "SurfF": None, 
                 "MarchingCubesRes": -1
                }
    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+"object")
    
    if not os.path.exists(sim_obj_name+"/"+sim_obj_name+"-training-settings.json"):
        training_dict = getDefaultTrainingSettings()
        json_object = json.dumps(training_dict, indent=4)
        # Writing to sample.json
        with open(sim_obj_name+"/"+sim_obj_name+"-training-settings.json", "w") as outfile:
            outfile.write(json_object)

def generate_small_box(sim_obj_name):
    np_object = generate_from_sdf("Box", yms=1e5)


    training_dict = getDefaultTrainingSettings()

    training_dict["LRStart"] = 1e-3
    training_dict["LREnd"] = 1e-4
    training_dict["TSamplingStdev"] = 1

    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+"object")
    
    if not os.path.exists(sim_obj_name+"/"+sim_obj_name+"-training-settings.json"):
        
        json_object = json.dumps(training_dict, indent=4)
        # Writing to sample.json
        with open(sim_obj_name+"/"+sim_obj_name+"-training-settings.json", "w") as outfile:
            outfile.write(json_object)
    pass

def generate_large_box(sim_obj_name):
    np_object = generate_from_sdf("BigBox", yms=1e3, prs = 0.45, rhos = 100, surf = True)

    training_dict = getDefaultTrainingSettings()

    training_dict["LRStart"] = 1e-3
    training_dict["LREnd"] = 1e-4
    training_dict["TSamplingStdev"] = 1
    training_dict["NumTrainingSteps"] = 10000

    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+"object")
    
    if not os.path.exists(sim_obj_name+"/"+sim_obj_name+"-training-settings.json"):
        
        json_object = json.dumps(training_dict, indent=4)
        # Writing to sample.json
        with open(sim_obj_name+"/"+sim_obj_name+"-training-settings.json", "w") as outfile:
            outfile.write(json_object)
    
def generate_ribbon(sim_obj_name):
    np_object = generate_from_sdf("PaperSheet", yms=1e7, prs=0.45, rhos = 10, surf = True)

    training_dict = getDefaultTrainingSettings()

    training_dict["LRStart"] = 1e-4
    training_dict["LREnd"] = 1e-5
    training_dict["TSamplingStdev"] = 0.1
    training_dict["NumTrainingSteps"] = 20000
    training_dict["NumSamplePts"] = 2000

    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+"object")
    
    if not os.path.exists(sim_obj_name+"/"+sim_obj_name+"-training-settings.json"):
        
        json_object = json.dumps(training_dict, indent=4)
        # Writing to sample.json
        with open(sim_obj_name+"/"+sim_obj_name+"-training-settings.json", "w") as outfile:
            outfile.write(json_object)

def generate_nerf_tree(sim_obj_name):
    nerf_pts, nerf_colors = read_ply("nerftree.ply")

    # nerf rearrangements
    nerf_pts = rotate_points_x(torch.tensor(nerf_pts, device=device, dtype=torch.float32), -90, axis = 0).cpu().detach().numpy()
    nerf_pts = nerf_pts*5
    nerf_pts[:,1] -= np.min(nerf_pts[:,1])

    print(nerf_pts.shape, nerf_colors.shape)
    # normalize colors
    nerf_colors_normed = nerf_colors/255.0

    #find the trunk and branch (appx) and stiffen them
    selected_rows = np.where((nerf_colors_normed[:, 1] < 1*nerf_colors_normed[:, 0]) & (nerf_colors_normed[:, 1] < 1*nerf_colors_normed[:, 2]))[0]
    nerf_YMs = 1e4*np.ones(nerf_pts.shape[0])
    nerf_YMs[selected_rows] *= 100

    nerf_PRs = 0.45*np.ones_like(nerf_YMs)
    nerf_Rho = 100*np.ones_like(nerf_YMs)


    plot_implicit(nerf_pts, scalars=nerf_YMs, colors=nerf_colors_normed)


    # 5. Create and save object
    np_object = {"Name":sim_obj_name, 
                 "Dim": 3, 
                 "BoundingBoxSamplePts": None, 
                 "BoundingBoxSignedDists": None,
                 "ObjectSamplePts": nerf_pts, 
                 "ObjectSampleColors" : nerf_colors,
                 "ObjectYMs": nerf_YMs, 
                 "ObjectPRs": nerf_PRs, 
                 "ObjectRho": nerf_Rho, 
                 "ObjectColors": None,
                 "ObjectVol": 1, 
                 "SurfV": None, 
                 "SurfF": None, 
                 "MarchingCubesRes": -1
                }
    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+"object")
    
    if not os.path.exists(sim_obj_name+"/"+sim_obj_name+"-training-settings.json"):
        training_dict = getDefaultTrainingSettings()
        json_object = json.dumps(training_dict, indent=4)
        # Writing to sample.json
        with open(sim_obj_name+"/"+sim_obj_name+"-training-settings.json", "w") as outfile:
            outfile.write(json_object)

def generate_nerf_iron(sim_obj_name):
    nerf_pts, nerf_colors = read_ply("nerfiron.ply")

    # nerf rearrangements
    nerf_pts = rotate_points_x(torch.tensor(nerf_pts, device=device, dtype=torch.float32), -90, axis = 0).cpu().detach().numpy()
    nerf_pts = nerf_pts
    nerf_pts[:,1] -= np.min(nerf_pts[:,1])

    print(nerf_pts.shape, nerf_colors.shape)
    # normalize colors
    nerf_colors_normed = nerf_colors/255.0

    #find the trunk and branch (appx) and stiffen them
    nerf_YMs = 1e4*np.ones(nerf_pts.shape[0])

    nerf_PRs = 0.45*np.ones_like(nerf_YMs)
    nerf_Rho = 100*np.ones_like(nerf_YMs)


    plot_implicit(nerf_pts, scalars=nerf_YMs, colors=nerf_colors_normed)


    # 5. Create and save object
    np_object = {"Name":sim_obj_name, 
                 "Dim": 3, 
                 "BoundingBoxSamplePts": None, 
                 "BoundingBoxSignedDists": None,
                 "ObjectSamplePts": nerf_pts, 
                 "ObjectSampleColors" : nerf_colors,
                 "ObjectYMs": nerf_YMs, 
                 "ObjectPRs": nerf_PRs, 
                 "ObjectRho": nerf_Rho, 
                 "ObjectColors": None,
                 "ObjectVol": 0.01, 
                 "SurfV": None, 
                 "SurfF": None, 
                 "MarchingCubesRes": -1
                }
    
    if not os.path.exists(sim_obj_name):
        os.makedirs(sim_obj_name)
    torch.save(np_object, sim_obj_name + "/" +sim_obj_name+"-"+"object")
    
    if not os.path.exists(sim_obj_name+"/"+sim_obj_name+"-training-settings.json"):
        training_dict = getDefaultTrainingSettings()
        json_object = json.dumps(training_dict, indent=4)
        # Writing to sample.json
        with open(sim_obj_name+"/"+sim_obj_name+"-training-settings.json", "w") as outfile:
            outfile.write(json_object)

# NEW OBJECT FORMAT
# generate_ct_abdomen("CTAbdomen")
# generate_ct_bladder("CTBladder")
# generate_clean_ct_skullbrain("CTBrain")
# generate_ct_skullstripped_brain("CTSkullStripped")
# generate_simple_skullbrain("SimpleBrain2")
# generate_layered_sphere("LayeredSphere")
# generate_small_box("OrsBox")
# generate_large_box("LargeBox")
# generate_ribbon("Ribbon")
# generate_nerf_tree("NerfTree")
# generate_nerf_iron("NerfIron")