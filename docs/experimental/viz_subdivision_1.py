from dipy.io.image import load_nifti
from fury import actor, window
from fury.lib import (Actor, AdaptiveSubdivisionFilter,
                      ButterflySubdivisionFilter, LinearSubdivisionFilter,
                      LoopSubdivisionFilter, PolyDataMapper)

import numpy as np
import os


def adaptive_subdivide_polydata(polydata, max_edge_len=None, max_tri_area=None,
                                max_n_tris=None, max_n_passes=None):
    """Adaptively subdivide an actor's mesh.

    Parameters
    ----------
    polydata : vtkPolyData
        Polydata to be subdivided.
    max_edge_len : float, optional
        The maximum edge length that a triangle may have. Edges longer than
        this value are split in half and the associated triangles are modified
        accordingly. The default is None.
    max_tri_area : float, optional
        The maximum area that a triangle may have. Triangles larger than this
        value are subdivided to meet this threshold. Note that if this
        criterion is used it may produce non-watertight meshes as a result.
        The default is None.
    max_n_tris : int, optional
        The maximum number of triangles that can be created. If the limit is
        hit, it may result in premature termination of the algorithm and the
        results may be less than satisfactory (for example non-watertight
        meshes may be created). By default, the limit is set to a very large
        number (i.e., no effective limit). The default is None.
    max_n_passes : int, optional
        The maximum number of passes (i.e., levels of subdivision). If the
        limit is hit, then the subdivision process stops and additional passes
        (needed to meet other criteria) are aborted. The default limit is set
        to a very large number (i.e., no effective limit). The default is None.

    Returns
    -------
    subdiv_polydata : vtkPolyData
        The subdivided polydata.
    """
    subdiv_filter = AdaptiveSubdivisionFilter()
    if max_edge_len:
        subdiv_filter.SetMaximumEdgeLength(max_edge_len)
    if max_tri_area:
        subdiv_filter.SetMaximumTriangleArea(max_tri_area)
    if max_n_tris:
        subdiv_filter.SetMaximumNumberOfTriangles(max_n_tris)
    if max_n_passes:
        subdiv_filter.SetMaximumNumberOfPasses(max_n_passes)
    subdiv_filter.SetInputData(polydata)
    subdiv_filter.Update()
    return subdiv_filter.GetOutput()


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
        color = (0, 1, 0)
        og_roi_actor = actor.contour_from_roi(img_data, affine=img_affine,
                                           color=color)

        og_roi_mapper = og_roi_actor.GetMapper()
        og_roi_mapper.Update()

        og_roi_polydata = og_roi_mapper.GetInput()

        print(og_roi_polydata.GetNumberOfPoints())
        print(og_roi_polydata.GetNumberOfPolys())

        """
        subdiv_polydata = subdivide_polydata(og_roi_polydata, n_subdivs=2,
                                             subdiv_method='butterfly')
        """

        subdiv_polydata = adaptive_subdivide_polydata(
            og_roi_polydata, max_edge_len=.2)

        print(subdiv_polydata.GetNumberOfPoints())
        print(subdiv_polydata.GetNumberOfPolys())

        subdiv_mapper = PolyDataMapper()
        subdiv_mapper.SetInputData(subdiv_polydata)
        subdiv_mapper.ScalarVisibilityOff()

        subdiv_actor = Actor()
        subdiv_actor.SetMapper(subdiv_mapper)

        subdiv_actor.GetProperty().SetRepresentationToWireframe()
        subdiv_actor.GetProperty().SetColor(color)

        scene = window.Scene()
        #scene.add(og_roi_actor)
        scene.add(subdiv_actor)
        window.show(scene)
