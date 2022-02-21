from fury import window
from fury.lib import CellArray, Points, PolyData, PolyDataMapper
from fury.utils import numpy_to_vtk_points
from fury.shaders import (attribute_to_actor, load, replace_shader_in_actor,
                          shader_to_actor)


from fury.lib import Actor
from fury.utils import numpy_to_vtk_colors, set_polydata_colors


import numpy as np


def generate_peaks():
    dirs01 = np.array([[-.4, .4, .8], [.7, .6, .1], [.4, -.3, .2],
                       [0, 0, 0], [0, 0, 0]])
    dirs10 = np.array([[.6, -.6, -.2], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0]])
    dirs11 = np.array([[0., .3, .3], [-.8, .4, -.5], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0]])
    dirs12 = np.array([[0, 0, 0], [.7, .6, .1], [0, 0, 0], [0, 0, 0],
                       [0, 0, 0]])

    peaks_dirs = np.zeros((1, 2, 3, 5, 3))

    peaks_dirs[0, 0, 1, :, :] = dirs01
    peaks_dirs[0, 1, 0, :, :] = dirs10
    peaks_dirs[0, 1, 1, :, :] = dirs11
    peaks_dirs[0, 1, 2, :, :] = dirs12

    peaks_vals = np.zeros((1, 2, 3, 5))

    peaks_vals[0, 0, 1, :] = np.array([.3, .2, .6, 0, 0])
    peaks_vals[0, 1, 0, :] = np.array([.5, 0, 0, 0, 0])
    peaks_vals[0, 1, 1, :] = np.array([.2, .5, 0, 0, 0])
    peaks_vals[0, 1, 2, :] = np.array([0, .7, 0, 0, 0])
    return peaks_dirs, peaks_vals, np.eye(4)


if __name__ == '__main__':
    peak_dirs, peak_vals, peak_affine = generate_peaks()

    valid_mask = np.abs(peak_dirs).max(axis=(-2, -1)) > 0

    indices = np.nonzero(valid_mask)

    valid_idx_dirs = peak_dirs[indices]

    valid_idx_vals = peak_vals[indices]

    valid_dirs = valid_idx_dirs[
                 :, ~np.all(np.abs(valid_idx_dirs).max(axis=-1) == 0, axis=0),
                 :]

    centers = np.asarray(indices).T

    centers_shape = centers.shape

    vtk_points = numpy_to_vtk_points(centers)

    colors = (1, 1, 1)
    colors = np.asarray(colors)
    colors = np.tile(255 * colors, (centers_shape[0], 1))
    vtk_colors = numpy_to_vtk_colors(colors)
    vtk_colors.SetName('colors')

    vtk_vertices = Points()
    # Create the topology of the point (a vertex)
    vtk_faces = CellArray()
    # Add points
    for i in range(len(centers)):
        p = centers[i]
        id = vtk_vertices.InsertNextPoint(p)
        vtk_faces.InsertNextCell(1)
        vtk_faces.InsertCellPoint(id)
    # Create a polydata object
    poly_data = PolyData()
    # Set the vertices and faces we created as the geometry and topology of the
    # polydata
    poly_data.SetPoints(vtk_vertices)
    poly_data.SetVerts(vtk_faces)

    set_polydata_colors(poly_data, colors)

    mapper = PolyDataMapper()
    mapper.SetInputData(poly_data)
    mapper.SetVBOShiftScaleMethod(False)
    #mapper.ScalarVisibilityOn()
    ##mapper.SetScalarModeToUsePointFieldData()
    #mapper.SelectColorArray('colors')
    #mapper.Update()

    test_dir = valid_idx_dirs[:, 0, :]

    peaks_shape = valid_dirs.shape

    peaks_data = np.empty((peaks_shape[:2] + (4,)))
    peaks_data[:, :, :3] = valid_dirs
    peaks_data[:, :peaks_shape[1], 3] = valid_idx_vals[:, :peaks_shape[1]]
    peaks_data = peaks_data.reshape((peaks_shape[0], peaks_shape[1] * 4))

    peak_actor = Actor()
    peak_actor.SetMapper(mapper)

    attribute_to_actor(peak_actor, centers, 'centers')

    attribute_to_actor(peak_actor, test_dir, 'dir')
    attribute_to_actor(peak_actor, peaks_data, 'peaks')

    vs_dec_code = \
    """
    in vec3 centers;
    in vec3 dir;
    in vec4 peaks[];
    
    out vec3 centerVertexMCVSOutput;
    out vec3 dirVertexMCVSOutput;
    out vec4 peaksVertexMCVSOutput[];
    out vec4 vertexMCVSOutput;
    """

    vs_impl_code = \
    """
    centerVertexMCVSOutput = centers;
    dirVertexMCVSOutput = dir;
    peaksVertexMCVSOutput = peaks;
    vertexMCVSOutput = vertexMC;
    """

    shader_to_actor(peak_actor, 'vertex', decl_code=vs_dec_code,
                    impl_code=vs_impl_code)
    replace_shader_in_actor(peak_actor, 'geometry', load('peak.geom'))

    scene = window.Scene()

    scene.add(peak_actor)

    scene.azimuth(30)
    scene.reset_camera()
    scene.reset_clipping_range()

    window.show(scene)
