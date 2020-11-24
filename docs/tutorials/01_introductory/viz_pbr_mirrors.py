from fury import actor, window
from fury.data import read_viz_textures
from fury.shaders import shader_to_actor, attribute_to_actor
from fury.utils import vertices_from_actor


import os
import numpy as np
import vtk


def get_cubemap(files_names):
    texture = vtk.vtkTexture()
    texture.CubeMapOn()
    for idx, fn in enumerate(files_names):
        if not os.path.isfile(fn):
            print('Nonexistent texture file:', fn)
            return texture
        else:
            # Read the images
            reader_factory = vtk.vtkImageReader2Factory()
            img_reader = reader_factory.CreateImageReader2(fn)
            img_reader.SetFileName(fn)

            flip = vtk.vtkImageFlip()
            flip.SetInputConnection(img_reader.GetOutputPort())
            flip.SetFilteredAxis(1)  # flip y axis
            texture.SetInputConnection(idx, flip.GetOutputPort(0))
    return texture


if __name__ == '__main__':
    num_actors = 5
    translate = 10
    centers = translate * np.random.rand(num_actors, 3) - translate / 2
    directions = np.random.rand(num_actors, 3)
    colors = np.random.rand(num_actors, 3)
    scales = np.random.rand(num_actors)
    #mirror_actor = actor.cone(centers, directions, colors, heights=scales,
    #                          resolution=40)
    #mirror_actor = actor.cube(centers, directions=directions, colors=colors,
    #                          scales=scales)
    mirror_actor = actor.sphere(centers, colors, radii=scales, theta=32,
                                phi=32)

    cubemap_fns = [read_viz_textures('skybox-px.jpg'),
                   read_viz_textures('skybox-nx.jpg'),
                   read_viz_textures('skybox-py.jpg'),
                   read_viz_textures('skybox-ny.jpg'),
                   read_viz_textures('skybox-pz.jpg'),
                   read_viz_textures('skybox-nz.jpg')]

    # Load the cube map
    cubemap = get_cubemap(cubemap_fns)

    # Load the skybox
    skybox = get_cubemap(cubemap_fns)
    skybox.InterpolateOn()
    skybox.RepeatOff()
    skybox.EdgeClampOn()

    skybox_actor = vtk.vtkSkybox()
    skybox_actor.SetTexture(skybox)

    mirror_actor.GetProperty().SetInterpolationToPBR()

    # Lets use a rough metallic surface
    metallic_coef = 1.
    roughness_coef = 0.

    #mirror_actor.GetProperty().SetMetallic(metallic_coef)
    #mirror_actor.GetProperty().SetRoughness(roughness_coef)

    # TODO: Try metallicity/roughness shader implementation
    # TODO: Create big_metal and big_rough arrays and pass them to VS
    mirror_vertices = vertices_from_actor(mirror_actor)
    # TODO: Forward metal and rough variables to FS attributeVSOutput
    # TODO: Update metallicUniform and roughnessUniform in ValuePass::Impl

    #shader_to_actor(mirror_actor, 'fragment', debug=True)

    scene = window.Scene()

    scene.UseImageBasedLightingOn()
    if vtk.VTK_VERSION_NUMBER >= 90000000000:
        scene.SetEnvironmentTexture(cubemap)
    else:
        scene.SetEnvironmentCubeMap(cubemap)

    scene.add(mirror_actor)
    scene.add(actor.axes())
    scene.add(skybox_actor)

    window.show(scene)
