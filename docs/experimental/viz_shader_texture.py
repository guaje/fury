import numpy as np
from viz_shader_canvas import cube

from fury import window
from fury.lib import (
    Actor,
    ImageFlip,
    JPEGReader,
    OpenGLPolyDataMapper,
    PolyDataNormals,
    SphereSource,
    Texture,
)
from fury.shaders import replace_shader_in_actor, shader_to_actor
from fury.utils import numpy_to_vtk_image_data

scene = window.Scene()

selected_actor = "sphere"

if selected_actor == "cube":
    canvas_actor = cube()

if selected_actor == "sphere":
    # Generate an sphere polydata
    sphere = SphereSource()
    sphere.SetThetaResolution(300)
    sphere.SetPhiResolution(300)

    norms = PolyDataNormals()
    norms.SetInputConnection(sphere.GetOutputPort())

    mapper = OpenGLPolyDataMapper()
    mapper.SetInputConnection(norms.GetOutputPort())

    canvas_actor = Actor()
    canvas_actor.SetMapper(mapper)

texture = Texture()
texture.CubeMapOn()

selected_texture = "numpy"

if selected_texture == "file":
    file = "sugar.jpg"
    imgReader = JPEGReader()
    imgReader.SetFileName(file)
    for i in range(6):
        flip = ImageFlip()
        flip.SetInputConnection(imgReader.GetOutputPort())
        flip.SetFilteredAxis(1)
        texture.SetInputConnection(i, flip.GetOutputPort())

if selected_texture == "numpy":
    arr = 255 * np.random.randn(512, 512, 3)
    arr[:256] = np.array([255, 0, 0])
    grid = numpy_to_vtk_image_data(arr.astype(np.uint8))
    for i in range(6):
        texture.SetInputDataObject(i, grid)

canvas_actor.SetTexture(texture)

vs_dec = "out vec3 TexCoords;"
vs_impl = """
vec3 camPos = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
TexCoords = reflect(vertexMC.xyz - camPos, normalize(normalMC));
//TexCoords = normalMC;
//TexCoords = vertexMC.xyz;
"""
shader_to_actor(canvas_actor, "vertex", decl_code=vs_dec, impl_code=vs_impl)

fs_code = """
//VTK::System::Dec  // Always start with this line
//VTK::Output::Dec  // Always have this line in your shader
in vec3 TexCoords;
uniform samplerCube texture_0;

void main()
{
    gl_FragData[0] = texture(texture_0, TexCoords);
}
"""
replace_shader_in_actor(canvas_actor, "fragment", fs_code)

scene.add(canvas_actor)

window.show(scene)
