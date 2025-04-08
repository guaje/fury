import time

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere as sphere
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti

# from dipy.reconst.csdeconv import auto_response_ssst
from dipy.reconst.shm import CsaOdfModel, sf_to_sh

# from dipy.viz.plotting import image_mosaic
from fury import actor, window

if __name__ == "__main__":
    show_man = window.ShowManager(size=(1280, 720))

    hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(
        name="stanford_hardi"
    )
    data, affine = load_nifti(hardi_fname)

    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    gtab = gradient_table(bvals, bvecs=bvecs)

    # sphere = unit_icosahedron.subdivide(n=5)

    nd = sphere.vertices.shape[0]
    print("The number of directions on the sphere is {}".format(nd))

    # response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)

    # Let us now compute the ODFs using this response function:
    csa_model = CsaOdfModel(gtab, 4)

    # data = data[:, :, data.shape[2] // 2 : data.shape[2] // 2 + 4]
    csa_odf = csa_model.fit(data).odf(sphere)

    print(csa_odf.shape)

    coeffs = sf_to_sh(csa_odf, sphere, sh_order_max=4)
    coeffs_shape = coeffs.shape
    mid = coeffs_shape[2] // 2
    coeffs = coeffs[:, :, mid : mid + 1, :]

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
