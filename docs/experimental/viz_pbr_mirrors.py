import os

import numpy as np

from fury import actor, window
from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.io import load_cubemap_texture
from fury.shaders import attribute_to_actor, shader_to_actor
from fury.utils import vertices_from_actor

if __name__ == '__main__':

    fetch_viz_cubemaps()

    texture_name = 'skybox'
    #texture_name = 'brudslojan'
    textures = read_viz_cubemap(texture_name)

    cubemap = load_cubemap_texture(textures)

    #cubemap.RepeatOff()
    #cuebmap.EdgeClampOn()

    scene = window.Scene(skybox=cubemap)
    #scene.skybox(visible=False)
    scene.skybox(gamma_correct=False)

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

    mirror_actor.GetProperty().SetInterpolationToPBR()

    # Lets use a rough metallic surface
    metallic_coef = 1.
    roughness_coef = 0.

    mirror_actor.GetProperty().SetMetallic(metallic_coef)
    mirror_actor.GetProperty().SetRoughness(roughness_coef)

    # TODO: Try metallicity/roughness shader implementation
    # TODO: Create big_metal and big_rough arrays and pass them to VS
    #mirror_vertices = vertices_from_actor(mirror_actor)
    # TODO: Forward metal and rough variables to FS attributeVSOutput
    # TODO: Update metallicUniform and roughnessUniform in ValuePass::Impl

    #shader_to_actor(mirror_actor, 'fragment', debug=True)

    scene.add(mirror_actor)
    scene.add(actor.axes())

    window.show(scene)
