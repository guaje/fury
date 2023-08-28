import os

import numpy as np

from fury import actor, window
from fury.data import fetch_viz_cubemaps, read_viz_cubemap, read_viz_textures
from fury.io import load_cubemap_texture
from fury.lib import (
    Actor,
    FloatArray,
    ImageReader2Factory,
    NamedColors,
    ParametricFunctionSource,
    ParametricSuperEllipsoid,
    PNGReader,
    Points,
    PolyData,
    PolyDataMapper,
    ProbeFilter,
    Texture,
    WarpScalar,
    numpy_support,
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


def uvt_coords(u_res, v_res, pd):
    """
    Generate u, v texture coordinates on a parametric surface.
    :param u_res: u resolution
    :param v_res: v resolution
    :param pd: The polydata representing the surface.
    :return: The polydata with the texture coordinates added.
    """
    u0 = 1.0
    v0 = 0.0
    du = 1.0 / (u_res - 1)
    dv = 1.0 / (v_res - 1)
    num_pts = pd.GetNumberOfPoints()
    t_coords = FloatArray()
    t_coords.SetNumberOfComponents(2)
    t_coords.SetNumberOfTuples(num_pts)
    t_coords.SetName('Texture Coordinates')
    pt_id = 0
    u = u0
    for i in range(0, u_res):
        v = v0
        for j in range(0, v_res):
            tc = [u, v]
            t_coords.SetTuple(pt_id, tc)
            v += dv
            pt_id += 1
        u -= du
    pd.GetPointData().SetTCoords(t_coords)
    return pd


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

    albedo_fn = read_viz_textures('alien_terrain_basecolor.png')
    material_fn = read_viz_textures('alien_terrain_material.png')
    emissive_fn = read_viz_textures('alien_terrain_emissive.png')
    normal_fn = read_viz_textures('alien_terrain_normal.png')
    height_fn = read_viz_textures('alien_terrain_height.png')

    #albedo_fn = read_viz_textures('metal_panel_dirty_basecolor.png')
    #material_fn = read_viz_textures('metal_panel_dirty_material.png')
    #normal_fn = read_viz_textures('metal_panel_dirty_normal.png')
    #height_fn = read_viz_textures('metal_panel_dirty_height.png')

    #albedo_fn = read_viz_textures('basic_rock_basecolor.png')
    #material_fn = read_viz_textures('basic_rock_material.png')
    #normal_fn = read_viz_textures('basic_rock_normal.png')
    #height_fn = read_viz_textures('basic_rock_height.png')

    #albedo_fn = read_viz_textures('stylized_ground_basecolor.png')
    #material_fn = read_viz_textures('stylized_ground_material.png')
    #normal_fn = read_viz_textures('stylized_ground_normal.png')
    #height_fn = read_viz_textures('stylized_ground_height.png')

    # Get the textures
    material = get_texture(material_fn)
    albedo = get_texture(albedo_fn)
    albedo.UseSRGBColorSpaceOn()
    normal = get_texture(normal_fn)
    emissive = get_texture(emissive_fn)
    emissive.UseSRGBColorSpaceOn()

    reader = PNGReader()
    reader.SetFileName(height_fn)
    reader.Update()

    image_extents = np.asarray(reader.GetDataExtent())
    image_size = np.asarray([
        image_extents[1] - image_extents[0],
        image_extents[3] - image_extents[2],
        image_extents[5] - image_extents[4]])

    u_res = 1000
    v_res = 1000
    surface = ParametricSuperEllipsoid()
    #surface.SetN1(.8)
    #surface.SetN2(.8)
    #surface.SetXRadius(.1)
    #surface.SetYRadius(.1)
    #surface.SetZRadius(.1)

    source = ParametricFunctionSource()
    source.SetUResolution(u_res)
    source.SetVResolution(v_res)
    source.SetParametricFunction(surface)
    source.Update()

    # Build the tcoords
    pd = uvt_coords(u_res, v_res, source.GetOutput())

    t_coords = pd.GetPointData().GetTCoords()

    probe_points = Points()
    probe_points.SetNumberOfPoints(t_coords.GetNumberOfValues())

    np_coords = numpy_support.vtk_to_numpy(t_coords)
    n_0 = np.zeros([t_coords.GetNumberOfTuples(), 1])

    probe_points.SetData(
        numpy_support.numpy_to_vtk(np.hstack([np_coords, n_0]) * image_size))

    probe_poly = PolyData()
    probe_poly.SetPoints(probe_points)

    probes = ProbeFilter()
    probes.SetSourceData(reader.GetOutput())
    probes.SetInputData(probe_poly)
    probes.Update()

    source.GetOutput().GetPointData().SetScalars(
        probes.GetOutput().GetPointData().GetScalars())

    #height_scale = .00075  # Snitch
    height_scale = .00125  # Alien-Rock
    #height_scale = .0005  # Ground
    warp = WarpScalar()
    #warp.SetInputData(tangents.GetOutput())
    warp.SetInputData(source.GetOutput())
    warp.SetScaleFactor(height_scale)
    warp.Update()

    # Now the tangents
    #tangents = vtk.vtkPolyDataTangents()
    #tangents.SetInputData(pd)
    #tangents.SetInputData(warp.GetOutput())
    #tangents.Update()

    # Build the pipeline
    mapper = PolyDataMapper()
    mapper.SetInputConnection(warp.GetOutputPort())
    #mapper.SetInputData(tangents.GetOutput())
    mapper.GetInput().GetPointData().SetScalars(None)

    actor = Actor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetInterpolationToPBR()

    colors = NamedColors()

    # Lets use a rough metallic surface
    metallic_coef = 1.
    roughness_coef = 1.

    # Other parameters
    occlusion_str = 10.
    normal_scale = 10.
    emissive_factor = [1., 1., 1.]

    # configure the basic properties
    actor.GetProperty().SetColor(colors.GetColor3d('White'))
    actor.GetProperty().SetMetallic(metallic_coef)
    actor.GetProperty().SetRoughness(roughness_coef)

    # configure textures (needs tcoords on the mesh)
    actor.GetProperty().SetBaseColorTexture(albedo)

    actor.GetProperty().SetORMTexture(material)
    actor.GetProperty().SetOcclusionStrength(occlusion_str)

    actor.GetProperty().SetEmissiveTexture(emissive)
    actor.GetProperty().SetEmissiveFactor(emissive_factor)

    # needs tcoords, normals and tangents on the mesh
    actor.GetProperty().SetNormalTexture(normal)
    actor.GetProperty().SetNormalScale(normal_scale)

    scene.add(actor)

    window.show(scene)
