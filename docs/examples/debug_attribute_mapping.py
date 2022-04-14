import numpy as np


from fury import window
from fury.shaders import shader_to_actor
from fury.utils import set_polydata_colors
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (vtkDataObject, vtkCellArray,
                                           vtkPolyData)
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper


if __name__ == '__main__':
    centers = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    data = np.array([[1, 0, 0, 1, 0, 0, 1, 1],
                     [0, 1, 0, 1, 0, 1, 0, 1],
                     [0, 0, 1, 1, 1, 0, 0, 1]])

    colors = np.array([1, 1, 1])
    colors = np.tile(255 * colors, (3, 1))

    vtk_vertices = vtkPoints()
    # Create the topology of the point (a vertex)
    vtk_faces = vtkCellArray()
    # Add points
    for i in range(len(centers)):
        p = centers[i]
        id = vtk_vertices.InsertNextPoint(p)
        vtk_faces.InsertNextCell(1)
        vtk_faces.InsertCellPoint(id)
    # Create a polydata object
    poly_data = vtkPolyData()
    # Set the vertices and faces we created as the geometry and topology of the
    # polydata
    poly_data.SetPoints(vtk_vertices)
    poly_data.SetVerts(vtk_faces)

    set_polydata_colors(poly_data, colors)

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(poly_data)

    actor = vtkActor()
    actor.SetMapper(mapper)

    num_components = data.shape[1]
    vtk_array = numpy_support.numpy_to_vtk(data)
    vtk_array.SetNumberOfComponents(num_components)
    vtk_array.SetName('data')

    mapper.GetInput().GetPointData().AddArray(vtk_array)
    mapper.MapDataArrayToVertexAttribute(
        'Data', 'Data', vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)

    vs_dec_code = \
        """
        in float data[8];
        """

    vs_impl_code = \
        """
        vec4 dataVecs[2];
        for(int i = 0; i < 2; i++)
        {
            dataVecs[i] = vec4(data[i * 4], data[i * 4 + 1], data[i * 4 + 2], 
            data[i * 4 + 3]);
        }
        vertexColorVSOutput = vec4(dataVecs[0].rg, 1, 1); 
        """

    shader_to_actor(actor, 'vertex', decl_code=vs_dec_code,
                    impl_code=vs_impl_code)

    scene = window.Scene()
    scene.add(actor)
    window.show(scene)
