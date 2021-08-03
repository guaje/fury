from dipy.data import get_fnames
from fury import actor, ui, window
from fury.data import read_viz_textures
from fury.io import load_polydata
from fury.utils import get_actor_from_polydata
from fury.shaders import add_shader_callback, load, shader_to_actor
from scipy.spatial import Delaunay


import math
import numpy as np
import os
import random
import vtk


def build_label(text, font_size=16, color=(1, 1, 1), bold=False, italic=False,
                shadow=False):
    label = ui.TextBlock2D()
    label.message = text
    label.font_size = font_size
    label.font_family = 'Arial'
    label.justification = 'left'
    label.bold = bold
    label.italic = italic
    label.shadow = shadow
    label.actor.GetTextProperty().SetBackgroundColor(0, 0, 0)
    label.actor.GetTextProperty().SetBackgroundOpacity(0.0)
    label.color = color
    return label


def change_slice_ao_strength(slider):
    global obj_actor
    obj_actor.GetProperty().SetOcclusionStrength(slider._value)


def change_slice_ior_1(slider):
    global ior_1
    ior_1 = slider._value


def change_slice_ior_2(slider):
    global ior_2
    ior_2 = slider._value


def change_slice_metallic(slider):
    global obj_actor
    obj_actor.GetProperty().SetMetallic(slider._value)


def change_slice_roughness(slider):
    global obj_actor
    obj_actor.GetProperty().SetRoughness(slider._value)


def change_slice_opacity(slider):
    global obj_actor
    obj_actor.GetProperty().SetOpacity(slider._value)


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


def obj_brain():
    brain_lh = get_fnames(name='fury_surface')
    polydata = load_polydata(brain_lh)
    return get_actor_from_polydata(polydata)


def obj_spheres(radii=2, theta=32, phi=32):
    centers = [[-5, 5, 0], [0, 5, 0], [5, 5, 0], [-5, 0, 0], [0, 0, 0],
               [5, 0, 0], [-5, -5, 0], [0, -5, 0], [5, -5, 0]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0],
              [0, 0, 0], [.5, .5, .5], [1, 1, 1]]
    return actor.sphere(centers, colors, radii=radii, theta=theta, phi=phi)


def obj_surface():
    size = 11
    vertices = list()
    for i in range(-size, size):
        for j in range(-size, size):
            fact1 = - math.sin(i) * math.cos(j)
            fact2 = - math.exp(abs(1 - math.sqrt(i ** 2 + j ** 2) / math.pi))
            z_coord = -abs(fact1 * fact2)
            vertices.append([i, j, z_coord])
    c_arr = np.random.rand(len(vertices), 3)
    random.shuffle(vertices)
    vertices = np.array(vertices)
    tri = Delaunay(vertices[:, [0, 1]])
    faces = tri.simplices
    c_loop = [None, c_arr]
    f_loop = [None, faces]
    s_loop = [None, "butterfly", "loop"]
    for smooth_type in s_loop:
        for face in f_loop:
            for color in c_loop:
                surface_actor = actor.surface(vertices, faces=face,
                                              colors=color, smooth=smooth_type)
    return surface_actor


def uniforms_callback(_caller, _event, calldata=None):
    global ior_1, ior_2
    if calldata is not None:
        calldata.SetUniformf('IOR1', ior_1)
        calldata.SetUniformf('IOR2', ior_2)


def win_callback(obj, event):
    global pbr_panel, control_panel, size
    if size != obj.GetSize():
        size_old = size
        size = obj.GetSize()
        size_change = [size[0] - size_old[0], 0]
        pbr_panel.re_align(size_change)
        control_panel.re_align(size_change)


if __name__ == '__main__':
    global control_panel, ior_1, ior_2, obj_actor, pbr_panel, size

    #obj_actor = obj_brain()
    #obj_actor = obj_surface()
    obj_actor = obj_spheres()

    #ambient = obj_actor.GetProperty().GetAmbient()
    #ambient_color = obj_actor.GetProperty().GetAmbientColor()
    #diffuse = obj_actor.GetProperty().GetDiffuse()
    #diffuseColor = obj_actor.GetProperty().GetDiffuseColor()
    #specular = obj_actor.GetProperty().GetSpecular()
    #specular_power = obj_actor.GetProperty().GetSpecularPower()
    #specular_color = obj_actor.GetProperty().GetSpecularColor()
    #specular_color = vtk.vtkNamedColors().GetColor3d('White')

    ior_1 = 1.  # Air
    #ior_1 = 1.333  # Water(20 °C)
    ior_2 = 1.5  # Glass
    #ior_2 = .18  # Silver
    #ior_2 = .47  # Gold
    #ior_2 = 1.  # Air
    #ior_2 = 2.33  # Platinum

    obj_actor.GetProperty().SetInterpolationToPBR()
    #metallic = .0
    metallic = obj_actor.GetProperty().GetMetallic()
    roughness = .0
    #roughness = obj_actor.GetProperty().GetRoughness()
    emissive_factor = obj_actor.GetProperty().GetEmissiveFactor()
    ao_strength = obj_actor.GetProperty().GetOcclusionStrength()

    obj_actor.GetProperty().SetMetallic(metallic)
    obj_actor.GetProperty().SetRoughness(roughness)

    opacity = 1.
    obj_actor.GetProperty().SetOpacity(opacity)

    add_shader_callback(obj_actor, uniforms_callback)

    fs_dec_code = load('refractive_dec.frag')
    fs_impl_code = load('refractive_impl.frag')

    #shader_to_actor(obj_actor, 'vertex', debug=True)
    #shader_to_actor(obj_actor, 'fragment', debug=True)
    shader_to_actor(obj_actor, 'fragment', decl_code=fs_dec_code)
    shader_to_actor(obj_actor, 'fragment', impl_code=fs_impl_code,
                    block='light')

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

    scene = window.Scene()

    scene.UseImageBasedLightingOn()
    if vtk.vtkVersion.GetVTKMajorVersion() >= 9:
        scene.SetEnvironmentTexture(cubemap)
    else:
        scene.SetEnvironmentCubeMap(cubemap)

    scene.add(obj_actor)
    scene.add(skybox_actor)

    #window.show(scene)

    show_m = window.ShowManager(scene=scene, reset_camera=False,
                                order_transparent=True)
    show_m.initialize()

    pbr_panel = ui.Panel2D((320, 500), position=(-25, 5),
                           color=(.25, .25, .25), opacity=.75, align='right')

    panel_label_refractive_pbr = build_label('Refractive PBR', font_size=18,
                                             bold=True)
    slider_label_ior_1 = build_label('IOR1')
    slider_label_ior_2 = build_label('IOR2')
    slider_label_metallic = build_label('Metallic')
    slider_label_roughness = build_label('Roughness')
    slider_label_ao_strength = build_label('AO Strength')

    label_pad_x = .06

    pbr_panel.add_element(panel_label_refractive_pbr, (.02, .95))
    pbr_panel.add_element(slider_label_ior_1, (label_pad_x, .86))
    pbr_panel.add_element(slider_label_ior_2, (label_pad_x, .77))
    pbr_panel.add_element(slider_label_metallic, (label_pad_x, .68))
    pbr_panel.add_element(slider_label_roughness, (label_pad_x, .59))
    pbr_panel.add_element(slider_label_ao_strength, (label_pad_x, .5))

    length = 150
    text_template = '{value:.1f}'

    slider_slice_ior_1 = ui.LineSlider2D(
        initial_value=ior_1, min_value=.1, max_value=5, length=length,
        text_template=text_template)
    slider_slice_ior_2 = ui.LineSlider2D(
        initial_value=ior_2, min_value=.1, max_value=5, length=length,
        text_template=text_template)
    slider_slice_metallic = ui.LineSlider2D(
        initial_value=metallic, max_value=1, length=length,
        text_template=text_template)
    slider_slice_roughness = ui.LineSlider2D(
        initial_value=roughness, max_value=1, length=length,
        text_template=text_template)
    slider_slice_ao_strength = ui.LineSlider2D(
        initial_value=ao_strength, max_value=1, length=length,
        text_template=text_template)

    slider_slice_ior_1.on_change = change_slice_ior_1
    slider_slice_ior_2.on_change = change_slice_ior_2
    slider_slice_metallic.on_change = change_slice_metallic
    slider_slice_roughness.on_change = change_slice_roughness
    slider_slice_ao_strength.on_change = change_slice_ao_strength

    slice_pad_x = .46

    pbr_panel.add_element(slider_slice_ior_1, (slice_pad_x, .86))
    pbr_panel.add_element(slider_slice_ior_2, (slice_pad_x, .77))
    pbr_panel.add_element(slider_slice_metallic, (slice_pad_x, .68))
    pbr_panel.add_element(slider_slice_roughness, (slice_pad_x, .59))
    pbr_panel.add_element(slider_slice_ao_strength, (slice_pad_x, .5))

    scene.add(pbr_panel)

    control_panel = ui.Panel2D((320, 80), position=(-25, 510),
                               color=(.25, .25, .25), opacity=.75,
                               align='right')

    panel_label_control = build_label('Control', font_size=18,
                                      bold=True)
    slider_label_opacity = build_label('Opacity')

    control_panel.add_element(panel_label_control, (.02, .7))
    control_panel.add_element(slider_label_opacity, (label_pad_x, .3))

    slider_slice_opacity = ui.LineSlider2D(
        initial_value=opacity, max_value=1, length=length,
        text_template=text_template)

    slider_slice_opacity.on_change = change_slice_opacity

    control_panel.add_element(slider_slice_opacity, (slice_pad_x, .3))

    scene.add(control_panel)

    size = scene.GetSize()

    show_m.add_window_callback(win_callback)

    show_m.start()
