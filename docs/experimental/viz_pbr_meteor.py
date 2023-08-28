from fury import actor, window
from fury.data import read_viz_textures
from vtk.util import numpy_support


import numpy as np
import os
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


def get_texture(file_name):
    texture = vtk.vtkTexture()
    if not os.path.isfile(file_name):
        print('Nonexistent texture file:', file_name)
        return texture
    # Read the images
    reader_factory = vtk.vtkImageReader2Factory()
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
    t_coords = vtk.vtkFloatArray()
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
    # NOTE: Need to be loaded in that order
    cubemap_fns = [read_viz_textures('skybox-px.jpg'),
                   read_viz_textures('skybox-nx.jpg'),
                   read_viz_textures('skybox-py.jpg'),
                   read_viz_textures('skybox-ny.jpg'),
                   read_viz_textures('skybox-pz.jpg'),
                   read_viz_textures('skybox-nz.jpg')]

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

    # Load the cube map
    cubemap = get_cubemap(cubemap_fns)

    # Load the skybox
    skybox = get_cubemap(cubemap_fns)
    skybox.InterpolateOn()
    skybox.RepeatOff()
    skybox.EdgeClampOn()

    # Get the textures
    material = get_texture(material_fn)
    albedo = get_texture(albedo_fn)
    albedo.UseSRGBColorSpaceOn()
    normal = get_texture(normal_fn)
    emissive = get_texture(emissive_fn)
    emissive.UseSRGBColorSpaceOn()

    scene = window.Scene()

    reader = vtk.vtkPNGReader()
    reader.SetFileName(height_fn)
    reader.Update()

    image_extents = np.asarray(reader.GetDataExtent())
    image_size = np.asarray([image_extents[1] - image_extents[0],
                             image_extents[3] - image_extents[2],
                             image_extents[5] - image_extents[4]])

    u_res = 1000
    v_res = 1000
    surface = vtk.vtkParametricSuperEllipsoid()
    #surface.SetN1(.8)
    #surface.SetN2(.8)
    #surface.SetXRadius(.1)
    #surface.SetYRadius(.1)
    #surface.SetZRadius(.1)

    source = vtk.vtkParametricFunctionSource()
    source.SetUResolution(u_res)
    source.SetVResolution(v_res)
    source.SetParametricFunction(surface)
    source.Update()

    # Build the tcoords
    pd = uvt_coords(u_res, v_res, source.GetOutput())

    t_coords = pd.GetPointData().GetTCoords()

    probe_points = vtk.vtkPoints()
    probe_points.SetNumberOfPoints(t_coords.GetNumberOfValues())

    np_coords = numpy_support.vtk_to_numpy(t_coords)
    n_0 = np.zeros([t_coords.GetNumberOfTuples(), 1])

    probe_points.SetData(
        numpy_support.numpy_to_vtk(np.hstack([np_coords, n_0]) * image_size))

    probe_poly = vtk.vtkPolyData()
    probe_poly.SetPoints(probe_points)

    probes = vtk.vtkProbeFilter()
    probes.SetSourceData(reader.GetOutput())
    probes.SetInputData(probe_poly)
    probes.Update()

    source.GetOutput().GetPointData().SetScalars(
        probes.GetOutput().GetPointData().GetScalars())

    #height_scale = .00075  # Snitch
    height_scale = .00125  # Alien-Rock
    #height_scale = .0005  # Ground
    warp = vtk.vtkWarpScalar()
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
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(warp.GetOutputPort())
    #mapper.SetInputData(tangents.GetOutput())
    mapper.GetInput().GetPointData().SetScalars(None)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetInterpolationToPBR()

    colors = vtk.vtkNamedColors()

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

    scene.UseImageBasedLightingOn()
    """
    if vtk.VTK_VERSION_NUMBER >= 90000000000:
        scene.SetEnvironmentTexture(cubemap)
    else:
        scene.SetEnvironmentCubeMap(cubemap)
    """
    scene.background(colors.GetColor3d('White'))

    scene.add(actor)

    # Comment out if you don't want a skybox
    #skybox_actor = vtk.vtkSkybox()
    #skybox_actor.SetTexture(skybox)
    #scene.add(skybox_actor)

    window.show(scene)
