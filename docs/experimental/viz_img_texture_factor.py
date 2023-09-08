import os

import numpy as np

from fury import actor, window
from fury.data import read_viz_textures
from fury.lib import FloatArray, ImageReader2Factory, Texture
from fury.primitive import prim_sphere
from fury.utils import (
    get_polydata_normals,
    normals_from_v_f,
    normals_to_actor,
    set_polydata_tcoords,
    update_polydata_normals,
)


def get_texture(file_name):
    texture = Texture()
    if not os.path.isfile(file_name):
        print('Nonexistent texture file:', file_name)
        return texture
    # Read the images
    reader_factory = ImageReader2Factory()
    img_reader = reader_factory.CreateImageReader2(file_name)
    img_reader.SetFileName(file_name)

    texture.SetInputConnection(img_reader.GetOutputPort())
    texture.Update()

    return texture


if __name__ == '__main__':
    scene = window.Scene()
    scene.background((1, 1, 1))

    center = np.array([[0, 0, 0]])

    vertices, faces = prim_sphere(name='repulsion100')
    texture_actor = actor.sphere(
        center, (1, 1, 1), vertices=vertices, faces=faces, use_primitive=True)

    actor_pd = texture_actor.GetMapper().GetInput()

    # NOTE: Calculate normals
    # NOTE: Method 1. Using VTK's functions
    update_polydata_normals(actor_pd)
    normals = get_polydata_normals(actor_pd)

    # NOTE: Method 2. From vertices and faces
    #normals = normals_from_v_f(vertices, faces)
    #normals_to_actor(texture_actor, normals)

    # NOTE: Cylindrical projection on a sphere
    # TODO: Study and understand UV texture coordinates
    u_vals = np.arctan2(normals[:, 0], normals[:, 2]) / (2 * np.pi) + .5
    v_vals = normals[:, 1] * .5 + .5

    num_pnts = normals.shape[0]

    t_coords = FloatArray()
    t_coords.SetNumberOfComponents(2)
    t_coords.SetNumberOfTuples(num_pnts)

    for i in range(num_pnts):
        u = u_vals[i]
        v = v_vals[i]
        tc = [u, v]
        t_coords.SetTuple(i, tc)

    set_polydata_tcoords(actor_pd, t_coords)

    albedo_fn = read_viz_textures('watermelon_basecolor.png')

    albedo = get_texture(albedo_fn)
    albedo.UseSRGBColorSpaceOn()

    # NOTE: Needed to work
    texture_actor.GetProperty().SetInterpolationToPBR()

    # NOTE: Configure textures (needs TCoords on the mesh)
    texture_actor.GetProperty().SetBaseColorTexture(albedo)

    scene.add(texture_actor)

    window.show(scene)
