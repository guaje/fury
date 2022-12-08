from dipy.io.image import load_nifti
from fury import actor, window
from fury.lib import (Actor, AdaptiveSubdivisionFilter,
                      ButterflySubdivisionFilter, LinearSubdivisionFilter,
                      LoopSubdivisionFilter, PolyDataMapper)

import numpy as np
import os


def subdivide_polydata(polydata, n_subdivs=1, subdiv_method='linear'):
    """Subdivide an actor's mesh.
    https://www.theobjects.com/dragonfly/dfhelp/2022-1/Content/3D%20Modeling/Meshes/Subdividing%20Meshes.htm

    Parameters
    ----------
    polydata : vtkPolyData
        Polydata to be subdivided.
    n_subdivs : int, optional
        Number of subdivisions to perform. The default is 1.
    subdiv_method : str, optional
        The subdivision method to use. The default is 'linear'.

    Returns
    -------
    subdiv_polydata : vtkPolyData
        The subdivided polydata.
    """
    subdiv_method = subdiv_method.lower()
    if subdiv_method == 'linear':
        subdiv_filter = LinearSubdivisionFilter()
    elif subdiv_method == 'loop':
        subdiv_filter = LoopSubdivisionFilter()
    elif subdiv_method == 'butterfly':
        subdiv_filter = ButterflySubdivisionFilter()
    subdiv_filter.SetNumberOfSubdivisions(n_subdivs)
    subdiv_filter.SetInputData(polydata)
    subdiv_filter.Update()
    return subdiv_filter.GetOutput()


if __name__ == '__main__':
    fdir = '/run/media/guaje/Data/GDrive/Data/repo/UCSF_tumor_grant/input/'
    fname = os.path.join(fdir, 'SPGR_tumor.nii.gz')

    img_data, img_affine = load_nifti(fname)
    dim = np.unique(img_data).shape[0]

    if dim == 2:
        og_roi_actor = actor.contour_from_roi(img_data, affine=img_affine,
                                           color=(0, 1, 0))

        og_roi_mapper = og_roi_actor.GetMapper()
        og_roi_mapper.Update()

        og_roi_polydata = og_roi_mapper.GetInput()

        print(og_roi_polydata.GetNumberOfPoints())
        print(og_roi_polydata.GetNumberOfPolys())

        subdiv_polydata = subdivide_polydata(og_roi_polydata, n_subdivs=2,
                                             subdiv_method='butterfly')

        print(subdiv_polydata.GetNumberOfPoints())
        print(subdiv_polydata.GetNumberOfPolys())

        subdiv_mapper = PolyDataMapper()
        subdiv_mapper.SetInputData(subdiv_polydata)

        subdiv_actor = Actor()
        subdiv_actor.SetMapper(subdiv_mapper)

        subdiv_actor.GetProperty().SetRepresentationToWireframe()

        scene = window.Scene()
        scene.add(subdiv_actor)
        window.show(scene)
