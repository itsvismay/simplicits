import torch 
import math

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def rotate_points_x(points, d, axis = 0):
    n = points.shape[0]

    # Find the midpoint of the points
    midpoint = torch.mean(points, dim=0)

    # Convert the rotation angle from degrees to radians
    theta = math.pi * d / 180.0

    # Create the translation matrix to move the points to the origin
    translation_to_origin = torch.eye(4).to(device)
    translation_to_origin[:3, 3] = -midpoint

    
    # Create the rotation matrix
    if axis == 0:
        # Create the rotation matrix
        rotation_matrix = torch.tensor([[1, 0, 0, 0],
                                    [0, math.cos(theta), -math.sin(theta), 0],
                                    [0, math.sin(theta), math.cos(theta), 0],
                                    [0, 0, 0, 1]], dtype=points.dtype).to(device)
    if axis == 1:
        rotation_matrix = torch.tensor([[math.cos(theta), 0, math.sin(theta),0],
                                        [0, 1, 0,0],
                                        [-math.sin(theta), 0, math.cos(theta),0],
                                        [0,0,0,1]],
                                   dtype=points.dtype, device=device)
    if axis == 2:
        rotation_matrix = torch.tensor([[math.cos(theta), -math.sin(theta), 0,0],
                                        [math.sin(theta), math.cos(theta), 0,0],
                                        [0, 0, 1,0],
                                        [0,0,0,1]],
                                   dtype=points.dtype, device=device)

    # Create the translation matrix to move the points back to their original position
    translation_to_original_position = torch.eye(4).to(device)
    translation_to_original_position[:3, 3] = midpoint

    # Apply the transformations to each point
    ones_column = torch.ones(n, 1).to(device)
    transformed_points = torch.cat((points, ones_column), dim=1)
    transformed_points = torch.matmul(torch.matmul(torch.matmul(translation_to_original_position, rotation_matrix), translation_to_origin), transformed_points.t()).t()

    # Extract the rotated points without the homogeneous coordinates
    rotated_points = transformed_points[:, :3]
    return rotated_points
