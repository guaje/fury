import os

import numpy as np
from dipy.data.fetcher import dipy_home
from dipy.io.image import load_nifti

from fury import actor, window

if __name__ == "__main__":
    show_man = window.ShowManager(size=(1280, 720))

    dataset_dir = os.path.join(dipy_home, "stanford_hardi")

    coeffs, affine = load_nifti(
        os.path.join(dataset_dir, "odf_debug_sh_coeffs_9x11x15(4).nii.gz")
    )

    valid_mask = np.abs(coeffs).max(axis=(-1)) > 0
    indices = np.nonzero(valid_mask)

    centers = np.asarray(indices).T

    x, y, z, s = coeffs.shape
    coeffs = coeffs[:, :, :].reshape((x * y * z, s))

    # max_val = coeffs.min(axis=1)
    # total = np.sum(abs(coeffs), axis=1)
    # coeffs = np.dot(np.diag(1 / total), coeffs)  # * 1.7

    odf_actor = actor.odf(centers, coeffs, scales=0.7)

    show_man.scene.add(odf_actor)

    show_man.start()
