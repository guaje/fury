from dipy.io.image import load_nifti
from fury import actor, window
from fury.lib import (Actor, AdaptiveSubdivisionFilter,
                      ButterflySubdivisionFilter, LinearSubdivisionFilter,
                      LoopSubdivisionFilter, PolyDataMapper)

import numpy as np
import os


if __name__ == '__main__':
    fdir = '/run/media/guaje/Data/GDrive/Data/repo/UCSF_tumor_grant/input/'
    fname = os.path.join(fdir, 'SPGR_tumor.nii.gz')

    img_data, img_affine = load_nifti(fname)
    dim = np.unique(img_data).shape[0]

    if dim == 2:
        og_roi_actor = actor.contour_from_roi(img_data, affine=img_affine,
                                           color=(0, 1, 0))

        print(og_roi_actor.GetMapper().GetInput().GetNumberOfPoints())
        print(og_roi_actor.GetMapper().GetInput().GetNumberOfPolys())

        #subdiv_filter = AdaptiveSubdivisionFilter
        subdiv_filter = ButterflySubdivisionFilter()
        #subdiv_filter = LinearSubdivisionFilter()
        #subdiv_filter = LoopSubdivisionFilter()
        subdiv_filter.SetNumberOfSubdivisions(2)
        subdiv_filter.SetInputData(og_roi_actor.GetMapper().GetInput())
        subdiv_filter.Update()

        sd_roi_mapper = PolyDataMapper()
        sd_roi_mapper.SetInputConnection(subdiv_filter.GetOutputPort())
        #sd_roi_mapper.ScalarVisibilityOff()

        print(sd_roi_mapper.GetInput().GetNumberOfPoints())
        print(sd_roi_mapper.GetInput().GetNumberOfPolys())

        sd_roi_actor = Actor()
        sd_roi_actor.SetMapper(sd_roi_mapper)

        sd_roi_actor.GetProperty().SetRepresentationToWireframe()

        scene = window.Scene()
        scene.add(sd_roi_actor)
        window.show(scene)
