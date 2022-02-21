from dipy.data import get_sphere
from fury import actor, window
from fury.lib import (Actor, CellArray, Command, Points, PolyData,
                      PolyDataMapper, VTK_OBJECT, calldata_type)
from fury.utils import (numpy_to_vtk_colors, numpy_to_vtk_points,
                        set_polydata_colors)
from fury.shaders import (attribute_to_actor, load, replace_shader_in_actor,
                          shader_to_actor)


import numpy as np


def find_instance(faces, first, second):
    global final_vertices_list
    found_flag = 0
    for i in range(196):
        if faces[i, 0] == first and faces[i, 1] == second and found_flag == 0:
            found_flag = 1
            temp = faces[i, :]
            x1 = temp[1]
            x2 = temp[2]
            final_vertices_list.append(x2)
            faces[i, :] = np.array([-1, -1, -1])
    if found_flag == 1:
        return find_instance(faces, x1, x2)
    else:
        return 0


@calldata_type(VTK_OBJECT)
def shader_callback(caller, event, calldata=None):
    global final_vertices_list, vertices
    if calldata is not None:
        for i in range(100):
            calldata.SetUniform3f('vertices[{}]'.format(i),
                                  vertices[i].tolist())
        for i in range(330):
            calldata.SetUniformi('order[{}]'.format(i), final_vertices_list[i])


if __name__ == '__main__':
    global final_vertices_list, num_vertices, vertices

    num_spheres = 5

    # Define N random spheres' centers
    centers = np.random.rand(num_spheres, 3) * 50 - 25

    # Create the geometry of a point (the coordinate)
    points = Points()

    # Create the topology of the point (a vertex)
    polys = CellArray()

    # We need an array of point id's for InsertNextCell.
    pnt_ids = np.zeros(num_spheres, dtype=np.int32)

    for i in range(num_spheres):
        pnt_ids[i] = points.InsertNextPoint(centers[i, :].tolist())

    polys.InsertNextCell(num_spheres, pnt_ids)

    # Create a polydata object
    polydata = PolyData()

    # Set the points and polys we created as the geometry and
    # topology of the polydata
    polydata.SetPoints(points)
    #polydata.SetPolys(polys)
    polydata.SetVerts(polys)

    # Create array of vertex colors
    colors = np.random.rand(num_spheres, 3) * 255
    vtk_colors = numpy_to_vtk_colors(colors)
    vtk_colors.SetName('colors')

    set_polydata_colors(polydata, colors)

    # Get vertices and faces of a sphere
    sphere = get_sphere('repulsion100')

    vertices = sphere.vertices
    faces = sphere.faces.astype('i4')

    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]

    # TODO: Move radius calculation to GS
    vertices *= 2

    final_vertices_list = []

    for i in range(num_faces):
        x = faces[i, :] == np.array([-1, -1, -1])
        if x.sum() == 0:
            first = faces[i, 1]
            second = faces[i, 2]
            final_vertices_list.append(faces[i, 0])
            final_vertices_list.append(faces[i, 1])
            final_vertices_list.append(faces[i, 2])
            find_instance(faces, first, second)

    mapper = PolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetVBOShiftScaleMethod(False)
    mapper.AddObserver(Command.UpdateShaderEvent, shader_callback)

    sphere_actor = Actor()
    sphere_actor.SetMapper(mapper)
    #sphere_actor.GetProperty().SetRepresentationToWireframe()
    #sphere_actor.GetProperty().SetPointSize(2)

    # TODO: Load shader from file
    gs_code = \
    """
    //VTK::System::Dec
    //VTK::PositionVC::Dec
    uniform mat4 MCDCMatrix;
    uniform mat4 MCVCMatrix;
    uniform vec3 vertices[{num_vertices}];
    uniform int order[330];
    //VTK::PrimID::Dec
    
    // declarations below aren't necessary because
    // they are already injected by PrimID template
    //in vec4 vertexColorVSOutput[];
    //out vec4 vertexColorGSOutput;
    //in vec4 vertexVCVSOutput[];
    //out vec4 vertexVCVSOutput;
    
    //VTK::Color::Dec
    //VTK::Normal::Dec
    //VTK::Light::Dec
    //VTK::TCoord::Dec
    //VTK::Picking::Dec
    //VTK::DepthPeeling::Dec
    //VTK::Clip::Dec
    //VTK::Output::Dec
    
    layout(points) in;
    
    layout(triangle_strip, max_vertices = 200) out;
    
    void build_sphere(vec4 position)
    {{
        for(int i = 0; i < 100; i++)
        {{
            vec4 newPoint = vec4(vertices[order[i]], 0);
            gl_Position = position + MCDCMatrix * newPoint;
            //vertexVCGSOutput = vertexVCVSOutput[0] + MCVCMatrix * newPoint;
            EmitVertex();
        }}
        EndPrimitive();
    }}
    
    void main()
    {{
        vertexColorGSOutput = vertexColorVSOutput[0];
        build_sphere(gl_in[0].gl_Position);
    }}
    """.format(num_vertices=num_vertices)

    replace_shader_in_actor(sphere_actor, 'geometry', gs_code)

    scene = window.Scene()

    scene.add(sphere_actor)
    scene.add(actor.axes())

    window.show(scene)
