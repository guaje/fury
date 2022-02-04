from dipy.data import get_fnames
from fury import actor, ui, window
from fury.data import (fetch_viz_cubemaps, fetch_viz_models, read_viz_cubemap,
                       read_viz_models)
from fury.io import load_cubemap_texture, load_polydata
from fury.lib import ImageData, Texture, numpy_support
from fury.utils import (get_actor_from_polydata, get_polydata_colors,
                        get_polydata_vertices, rotate, set_polydata_colors)
from fury.shaders import add_shader_callback, load, shader_to_actor
from scipy.spatial import Delaunay


import math
import numpy as np
import random


def change_slice_ior_1(slider):
    global ior_1
    ior_1 = slider.value


def change_slice_ior_2(slider):
    global ior_2
    ior_2 = slider.value


def change_slice_metallic(slider):
    global obj_actor
    obj_actor.GetProperty().SetMetallic(slider.value)


def change_slice_roughness(slider):
    global obj_actor
    obj_actor.GetProperty().SetRoughness(slider.value)


def change_slice_opacity(slider):
    global obj_actor
    obj_actor.GetProperty().SetOpacity(slider.value)


def get_cubemap_from_ndarrays(array, flip=True):
    texture = Texture()
    texture.CubeMapOn()
    for idx, img in enumerate(array):
        vtk_img = ImageData()
        vtk_img.SetDimensions(img.shape[1], img.shape[0], 1)
        if flip:
            # Flip horizontally
            vtk_arr = numpy_support.numpy_to_vtk(np.flip(
                img.swapaxes(0, 1), axis=1).reshape((-1, 3), order='F'))
        else:
            vtk_arr = numpy_support.numpy_to_vtk(img.reshape((-1, 3),
                                                             order='F'))
        vtk_arr.SetName('Image')
        vtk_img.GetPointData().AddArray(vtk_arr)
        vtk_img.GetPointData().SetActiveScalars('Image')
        texture.SetInputDataObject(idx, vtk_img)
    return texture


def obj_brain():
    brain_lh = get_fnames(name='fury_surface')
    polydata = load_polydata(brain_lh)
    return get_actor_from_polydata(polydata)


def obj_model(model='glyptotek.vtk', color=None):
    if model != 'glyptotek.vtk':
        fetch_viz_models()
    model = read_viz_models(model)
    polydata = load_polydata(model)
    if color is not None:
        color = np.asarray([color]) * 255
        colors = get_polydata_colors(polydata)
        if colors is not None:
            num_vertices = colors.shape[0]
            new_colors = np.repeat(color, num_vertices, axis=0)
            colors[:, :] = new_colors
        else:
            vertices = get_polydata_vertices(polydata)
            num_vertices = vertices.shape[0]
            new_colors = np.repeat(color, num_vertices, axis=0)
            set_polydata_colors(polydata, new_colors)
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

    fetch_viz_cubemaps()

    # texture_name = 'skybox'
    texture_name = 'brudslojan'
    textures = read_viz_cubemap(texture_name)

    cubemap = load_cubemap_texture(textures)

    """
    img_shape = (1024, 1024)

    # Flip horizontally
    img_grad = np.flip(np.tile(np.linspace(0, 255, num=img_shape[0]),
                               (img_shape[1], 1)).astype(np.uint8), axis=1)
    cubemap_side_img = np.stack((img_grad,) * 3, axis=-1)

    cubemap_top_img = np.ones((img_shape[0], img_shape[1], 3)).astype(
        np.uint8) * 255

    cubemap_bottom_img = np.zeros((img_shape[0], img_shape[1], 3)).astype(
        np.uint8)

    cubemap_imgs = [cubemap_side_img, cubemap_side_img, cubemap_top_img,
                    cubemap_bottom_img, cubemap_side_img, cubemap_side_img]

    cubemap = get_cubemap_from_ndarrays(cubemap_imgs, flip=False)
    """

    #cubemap.RepeatOff()
    #cubemap.EdgeClampOn()

    scene = window.Scene(skybox=cubemap)
    scene.skybox(gamma_correct=False)

    #scene.background((1, 1, 1))
    #scene.background((0, 0, 0))

    #scene.roll(-145)
    #scene.pitch(70)

    # Scene rotation only. For specific skybox only.
    scene.yaw(-110)

    #obj_actor = obj_brain()
    #obj_actor = obj_surface()
    #obj_actor = obj_model(model='suzanne.obj', color=(0, 1, 1))
    #obj_actor = obj_model(model='glyptotek.vtk', color=(0, 1, 1))
    obj_actor = obj_model(model='glyptotek.vtk')
    #obj_actor = obj_spheres()

    rotate(obj_actor, rotation=(-145, 0, 0, 1))
    rotate(obj_actor, rotation=(-70, 1, 0, 0))

    rotate(obj_actor, rotation=(-110, 0, 1, 0))

    scene.add(obj_actor)

    scene.reset_camera()
    scene.zoom(1.9)

    ior_1 = 1.  # Air
    #ior_1 = 1.333  # Water(20 °C)
    ior_2 = 1.5  # Glass
    #ior_2 = .18  # Silver
    #ior_2 = .47  # Gold
    #ior_2 = 1.  # Air
    #ior_2 = 2.33  # Platinum

    obj_actor.GetProperty().SetInterpolationToPBR()
    metallic = 0
    obj_actor.GetProperty().SetMetallic(metallic)
    roughness = 0
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

    #window.show(scene)

    show_m = window.ShowManager(scene=scene, size=(1920, 1080),
                                reset_camera=False, order_transparent=True)
    show_m.initialize()

    pbr_panel = ui.Panel2D((400, 230), position=(5, 5), color=(.25, .25, .25),
                           opacity=.75, align='right')

    panel_label_refractive_pbr = ui.TextBlock2D(
        text='Refractive PBR', font_size=18, bold=True)
    slider_label_ior_1 = ui.TextBlock2D(text='IoR1', font_size=16)
    slider_label_ior_2 = ui.TextBlock2D(text='IoR2', font_size=16)
    slider_label_metallic = ui.TextBlock2D(text='Metallic', font_size=16)
    slider_label_roughness = ui.TextBlock2D(text='Roughness', font_size=16)

    label_pad_x = .06

    pbr_panel.add_element(panel_label_refractive_pbr, (.02, .90))
    pbr_panel.add_element(slider_label_ior_1, (label_pad_x, .70))
    pbr_panel.add_element(slider_label_ior_2, (label_pad_x, .50))
    pbr_panel.add_element(slider_label_metallic, (label_pad_x, .30))
    pbr_panel.add_element(slider_label_roughness, (label_pad_x, .10))

    length = 260
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

    slider_slice_ior_1.on_change = change_slice_ior_1
    slider_slice_ior_2.on_change = change_slice_ior_2
    slider_slice_metallic.on_change = change_slice_metallic
    slider_slice_roughness.on_change = change_slice_roughness

    slice_pad_x = .28

    pbr_panel.add_element(slider_slice_ior_1, (slice_pad_x, .70))
    pbr_panel.add_element(slider_slice_ior_2, (slice_pad_x, .50))
    pbr_panel.add_element(slider_slice_metallic, (slice_pad_x, .30))
    pbr_panel.add_element(slider_slice_roughness, (slice_pad_x, .10))

    scene.add(pbr_panel)

    control_panel = ui.Panel2D((400, 90), position=(5, 240),
                               color=(.25, .25, .25), opacity=.75,
                               align='right')

    panel_label_control = ui.TextBlock2D(
        text='Control', font_size=18, bold=True)
    slider_label_opacity = ui.TextBlock2D(text='Opacity', font_size=16)

    control_panel.add_element(panel_label_control, (.02, .70))
    control_panel.add_element(slider_label_opacity, (label_pad_x, .30))

    slider_slice_opacity = ui.LineSlider2D(
        initial_value=opacity, max_value=1, length=length,
        text_template=text_template)

    slider_slice_opacity.on_change = change_slice_opacity

    control_panel.add_element(slider_slice_opacity, (slice_pad_x, .3))

    scene.add(control_panel)

    size = scene.GetSize()

    show_m.add_window_callback(win_callback)

    show_m.start()
