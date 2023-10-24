import sys, os
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


print(training_name)

# Opening JSON file with training settings
with open(fname+"-training-settings.json", 'r') as openfile:
    training_settings = json.load(openfile)

if os.path.exists(object_name+"/"+training_name+"-training/timings"):
    timings = torch.load(object_name+"/"+training_name+"-training" + "/timings")
else:
    timings = None

losses = np.array(torch.load(object_name+"/"+training_name+"-training" + "/losses"))
print(losses)
try:
    losses = losses[:,0]
except IndexError:
    losses = np.expand_dims(losses, axis=-1)
    losses = losses[:,0]

Handles_post = torch.load(object_name+"/"+training_name+"-training" + "/Handles_post")
Handles_pre = torch.load(object_name+"/"+training_name+"-training" + "/Handles_pre")

for nnnn, pppp in Handles_post.model.named_parameters():
    print(nnnn, pppp.size())

# Get a list of all files in the folder
all_files = os.listdir(object_name+"/"+training_name+"-training" + "/")
# Filter files with the "training" prefix
training_files = [filename for filename in all_files if filename.startswith('Handles_post-its-')]
epoch_list = sorted(list(set([int(f.split("Handles_post-its-")[1].split("-")[0]) for f in training_files])))

Handles_post.to_device(device)
Handles_pre.to_device(device)
Handles_pre.eval()
t_O = torch.tensor(np_object["ObjectSamplePts"]).to(device)

# Compute the moving average
window_size = 100
ma = moving_average(losses[500:], window_size)

# Plot the loss curve
plt.plot(ma)
plt.title('Loss curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

if timings is not None:
    # Compute the moving average
    window_size = 100
    ma = moving_average(timings[500:], window_size)

    # Plot the loss curve
    plt.plot(ma)
    plt.title('Timings')
    plt.xlabel('Iterations')
    plt.ylabel('Times in ms')
    plt.show()


np_W0, np_X0, np_G0 = test(Handles_pre, t_O, int(t_O.shape[0]))
np_W, np_X, np_G = test(Handles_post, t_O, int(t_O.shape[0]))

X0 = torch.tensor(np_X)
W  = torch.tensor(np_W)

num_handles = W.shape[1]
num_samples = X0.shape[0]
X03 = torch.cat((X0, torch.ones(X0.shape[0]).unsqueeze(-1)), dim=1)
X03reps = X03.repeat_interleave(3, dim=0).repeat((1, 3*num_handles))
Wreps = W.repeat_interleave(12, dim=1).repeat_interleave(3, dim=0)
WX03reps = torch.mul(Wreps, X03reps)
Bsetup = torch.kron(torch.ones(num_samples).unsqueeze(-1), torch.eye(3)).repeat((1,num_handles))
Bmask = torch.repeat_interleave(Bsetup, 4, dim=1)

B = torch.mul(Bmask, WX03reps)
print(B.T @ B)
print("ALSO WTW")
print(np_W.T@np_W)


####################################
global points 
global weights
points = np_X 
weights = np_W 

global handle_num
global epoch_num
handle_num = 0
epoch_num = len(epoch_list)

max_all_w = np.max(np.max(weights))
min_all_w = np.min(np.min(weights))

# Combo box to choose from options
# There, the options are a list of strings in `ui_options`,
# and the currently selected element is stored in `ui_options_selected`.
def callback():
    global handle_num, points, weights, epoch_num, epoch_list
    changed2, epoch_num = psim.SliderInt("Epoch:", epoch_num, v_min=0, v_max=len(epoch_list))
    changed1, handle_num = psim.SliderInt("Handles:", handle_num, v_min=0, v_max=weights.shape[1]-1)
    if changed1 or changed2:
        ii = int(handle_num)
        if epoch_num == len(epoch_list):        
            weights, points, np_G = test(Handles_post, t_O, int(t_O.shape[0]))
        else:
            ei = epoch_list[epoch_num]
            Handles_its = torch.load(object_name+"/"+training_name+"-training" + "/Handles_post-its-"+str(ei))
            Handles_its.to_device(device)
            weights, points,  np_G = test(Handles_its, t_O, int(t_O.shape[0]))

        w = weights[:, ii]
        # ps_cloud.add_scalar_quantity("Weight: "+str(ii), w, enabled=True)
        # Normalize the weights to the range [0, 1]
        # w_norm = (w - min_all_w) / (max_all_w - min_all_w)
        ps_cloud.update_point_positions(points)
        ps_cloud.add_scalar_quantity("Weight: "+str(ii), w, enabled=True, datatype="symmetric")
        # color_mat = np.column_stack((w_norm, w_norm, w_norm))
        # color_mat[:,1:2] = 1
        # # print(color_mat)
        # ps_cloud.add_color_quantity("weights: "+str(ii), color_mat, enabled=True)




# Show the plot
ps.init()
ps.remove_all_structures()
ps_cloud = ps.register_point_cloud("samples", points, enabled=True)
ps_cloud.add_scalar_quantity("Weight: "+str(0), weights[:,0], enabled=True, datatype="symmetric")
ps_unit_cloud = ps.register_point_cloud("UnitSphere",  np.array([[0,0,0],[1,0,0]]), enabled=True, radius=1)
ps.set_user_callback(callback)
ps.show()