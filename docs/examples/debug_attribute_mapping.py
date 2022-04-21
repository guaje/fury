import numpy as np


from fury import window
from fury.shaders import replace_shader_in_actor, shader_to_actor
from fury.utils import set_polydata_colors
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (vtkDataObject, vtkCellArray,
                                           vtkPolyData)
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

GEO_CODE = \
"""
//VTK::System::Dec
//VTK::PositionVC::Dec
uniform mat4 MCDCMatrix;

//VTK::PrimID::Dec

// declarations below aren't necessary because they are already injected 
// by PrimID template this comment is just to justify the passthrough below
//in vec4 vertexColorVSOutput[];
//out vec4 vertexColorGSOutput;

//VTK::Color::Dec
//VTK::Normal::Dec
//VTK::Light::Dec
//VTK::TCoord::Dec
//VTK::Picking::Dec
//VTK::DepthPeeling::Dec
//VTK::Clip::Dec
//VTK::Output::Dec

// Convert points to line strips
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

void build_square(vec4 position)
{
    gl_Position = position + vec4(-.5, -.5, 0, 0);  // 1: Bottom left
    EmitVertex();
    gl_Position = position + vec4(.5, -.5, 0, 0);  // 2: Bottom right
    EmitVertex();
    gl_Position = position + vec4(-.5, .5, 0, 0);  // 3: Top left
    EmitVertex();
    gl_Position = position + vec4(.5, .5, 0, 0);  // 4: Top right
    EmitVertex();
    EndPrimitive();
}

void main()
{
vertexColorGSOutput = vertexColorVSOutput[0];
build_square(gl_in[0].gl_Position);
}
"""


if __name__ == '__main__':
    centers = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    data = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                     [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                     [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]])

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

    point_data = mapper.GetInput().GetPointData()
    point_data.AddArray(vtk_array)

    mapper.MapDataArrayToVertexAttribute(
        'data', 'data', vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)

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
        vertexColorVSOutput = vec4(dataVecs[0].rgb, 1);
        """

    #TODO: Try different blocks
    shader_to_actor(actor, 'vertex', decl_code=vs_dec_code,
                    impl_code=vs_impl_code)

    replace_shader_in_actor(actor, 'geometry', GEO_CODE)

    scene = window.Scene()
    scene.add(actor)
    window.show(scene)
