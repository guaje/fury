"""
Tiny NeRF

Paper:
NeRF: Representing scenes as neural radiance fields for view synthesis:
https://arxiv.org/abs/2003.08934

Authors' TensorFlow implementation: https://github.com/bmild/nerf
"""

# NOTE: All functions that have a `#TESTED` under the docstring imply that they
# have been tested against their corresponding tensorflow implementations.
from math import pi

import matplotlib.pyplot as plt
import numpy as np

from fury import actor, window
from fury.data import read_dataset
from fury.lib import FloatArray, Texture
from fury.nn.utils import (
    get_minibatches,
    positional_encoding,
    run_one_iter_of_tinynerf,
)
from fury.optpkg import optional_package
from fury.utils import (
    apply_affine,
    rgb_to_vtk,
    set_polydata_tcoords,
    update_surface_actor_colors,
)

torch, has_torch, _ = optional_package("torch")

if has_torch:
    import torch
    import torch.nn as nn


# TinyNeRF: Network architecture
class VeryTinyNerfModel(nn.Module):
    """
    Define a "very tiny" NeRF model comprising three fully connected layers.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6):
        super(VeryTinyNerfModel, self).__init__()
        # Input layer (default: 39 -> 128)
        self.layer1 = nn.Linear(
            3 + 3 * 2 * num_encoding_functions, filter_size
        )
        # Layer 2 (default: 128 -> 128)
        self.layer2 = nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = nn.Linear(filter_size, 4)
        # Short hand for nn.functional.relu
        self.relu = nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def billboard_texture_actor(center, scale):
    texture_actor = actor.billboard(center, colors=(1, 1, 1), scales=scale)
    actor_pd = texture_actor.GetMapper().GetInput()
    uv_vals = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    tex_coords = FloatArray()
    tex_coords.SetNumberOfComponents(2)
    tex_coords.SetNumberOfTuples(4)
    [tex_coords.SetTuple(i, uv_vals[i]) for i in range(4)]
    set_polydata_tcoords(actor_pd, tex_coords)
    return texture_actor


def left_click_pose_callback(obj, event):
    global data_dict, images, img_actor, picked_pose, picked_pose_actor
    clicked_pose_actor = obj
    update_surface_actor_colors(clicked_pose_actor, np.array([[0, 1, 0]]))
    update_surface_actor_colors(picked_pose_actor, np.array([[1, 1, 1]]))
    picked_pose = data_dict[clicked_pose_actor]["idx"]
    picked_pose_actor = clicked_pose_actor
    img_arr = images[picked_pose] * 255
    texture_to_billboard(img_actor, img_arr.astype(np.uint8))


def pose_explorator(scene, pose_affines):
    global data_dict, img_actor, picked_pose, picked_pose_actor
    data_dict = {}
    centers = np.zeros((pose_affines.shape[0], 3))
    picked_pose = 0
    picked_pose_actor = None
    for i in range(pose_affines.shape[0]):
        centers[i] = apply_affine(pose_affines[i], centers[i])
        center = np.array([centers[i]])
        color = (0, 1, 0) if i == picked_pose else (1, 1, 1)
        pose_actor = actor.dot(center, color)
        pose_actor.AddObserver(
            "LeftButtonPressEvent", left_click_pose_callback, 1
        )
        if i == picked_pose:
            picked_pose_actor = pose_actor
        scene.add(pose_actor)
        data_dict[pose_actor] = {"idx": i}
    centers_mins = np.min(centers, axis=0)
    centers_maxs = np.max(centers, axis=0)
    centers_avgs = np.mean(np.vstack((centers_mins, centers_maxs)), axis=0)
    img_arr = images[picked_pose] * 255
    scale = 5
    bb_center = np.array([[centers_avgs[0], centers_maxs[1] + scale + 1, 0]])
    img_actor = billboard_texture_actor(bb_center, scale)
    texture_to_billboard(img_actor, img_arr.astype(np.uint8))
    scene.add(img_actor)


def texture_to_billboard(billboard_actor, img):
    grid = rgb_to_vtk(img)
    texture = Texture()
    texture.SetInputDataObject(grid)
    texture.Update()
    # billboard_actor.SetTexture(texture)
    billboard_actor.GetProperty().SetTexture("texture0", texture)


if __name__ == "__main__":
    global images

    scene = window.Scene()

    show_m = window.ShowManager(scene, size=(1280, 720))

    # Get data
    data_fname = read_dataset("tiny_nerf_data.npz")

    # Determine device to run on (GPU vs CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load up input images, poses, intrinsics, etc.
    # Load input images, poses, and intrinsics
    data = np.load(data_fname)

    # Images
    images = data["images"]

    # Camera extrinsics (poses)
    poses = data["poses"]

    # pose_explorator(scene, poses)
    # window.show(scene)
    # show_m.initialize()
    # show_m.start()

    tform_cam2world = torch.from_numpy(poses).to(device)

    # Focal length (intrinsics)
    focal_length = data["focal"]
    tform_focal_length = torch.from_numpy(focal_length).to(device)

    # Height and width of each image
    height, width = images.shape[1:3]

    # Near and far clipping thresholds for depth values.
    near_thresh = 2.0
    far_thresh = 6.0

    # Hold one image out (for test).
    testimg, testpose = images[101], tform_cam2world[101]
    testimg = torch.from_numpy(testimg).to(device)

    # Map images to device
    images = torch.from_numpy(images[:100, ..., :3]).to(device)

    # Display the image used for testing
    plt.imshow(testimg.detach().cpu().numpy())
    plt.show()

    # Parameters for TinyNeRF training

    # Number of functions used in the positional encoding (Be sure to update
    # the model if this number changes).
    num_encoding_functions = 6

    # Specify encoding function.
    encode = lambda x: positional_encoding(
        x, num_encoding_functions=num_encoding_functions
    )

    # Number of depth samples along each ray.
    depth_samples_per_ray = 32

    # Chunksize (NOTE: this isn't batchsize in the conventional sense. This
    # only specifies the number of rays to be queried in one go. Backprop still
    # happens only after all rays from the current "bundle" are queried and
    # rendered).
    # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory.
    chunksize = 16384

    # Optimizer parameters
    lr = 5e-3
    num_iters = 1000

    # Misc parameters
    display_every = 100  # Number of iters after which stats are displayed

    # Model
    model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train-Eval-Repeat!

    # Seed RNG, for repeatability
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    for i in range(num_iters):
        # Randomly pick an image as the target.
        target_img_idx = np.random.randint(images.shape[0])
        target_img = images[target_img_idx].to(device)
        target_tform_cam2world = tform_cam2world[target_img_idx].to(device)

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        rgb_predicted = run_one_iter_of_tinynerf(
            height,
            width,
            tform_focal_length,
            target_tform_cam2world,
            near_thresh,
            far_thresh,
            depth_samples_per_ray,
            model,
            encode,
            get_minibatches,
            chunksize=chunksize,
        )

        # Compute mean-squared error between the predicted and target images.
        # Backprop!
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Display images/plots/stats
        if i % display_every == 0:
            # Render the held-out view
            rgb_predicted = run_one_iter_of_tinynerf(
                height,
                width,
                tform_focal_length,
                testpose,
                near_thresh,
                far_thresh,
                depth_samples_per_ray,
                model,
                encode,
                get_minibatches,
                chunksize=chunksize,
            )
            loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
            print("Loss:", loss.item())
            psnr = -10.0 * torch.log10(loss)

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(rgb_predicted.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

    print("Done!")
