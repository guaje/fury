"""
Tiny NeRF

Paper:
NeRF: Representing scenes as neural radiance fields for view synthesis:
https://arxiv.org/abs/2003.08934

Authors' TensorFlow implementation: https://github.com/bmild/nerf
"""

# NOTE: All functions that have a `#TESTED` under the docstring imply that they
# have been tested against their corresponding tensorflow implementations.
import matplotlib.pyplot as plt
import numpy as np

from fury.data import read_dataset
from fury.nn.utils import (
    get_minibatches,
    positional_encoding,
    run_one_iter_of_tinynerf,
)
from fury.optpkg import optional_package

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


if __name__ == "__main__":
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
    tform_cam2world = data["poses"]
    tform_cam2world = torch.from_numpy(tform_cam2world).to(device)

    # Focal length (intrinsics)
    focal_length = data["focal"]
    focal_length = torch.from_numpy(focal_length).to(device)

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
            focal_length,
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
                focal_length,
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
