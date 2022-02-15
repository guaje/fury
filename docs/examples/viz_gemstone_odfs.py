from dipy.data import get_sphere
from dipy.io.image import load_nifti
from dipy.reconst.shm import sh_to_sf_matrix
from fury import actor, ui, window
from fury.data import (fetch_viz_cubemaps, fetch_viz_dmri, read_viz_cubemap,
                       read_viz_dmri)
from fury.io import load_cubemap_texture
from fury.lib import ImageData, Texture, numpy_support
from fury.shaders import add_shader_callback, load, shader_to_actor
from fury.utils import rotate, update_polydata_normals


import numpy as np


def change_slice_absorption(slider):
    global absorption
    absorption = slider.value


def change_slice_ior_1(slider):
    global ior_1
    ior_1 = slider.value


def change_slice_ior_2(slider):
    global ior_2
    ior_2 = slider.value


def change_slice_roughness(slider):
    global odf_actor_x, odf_actor_y, odf_actor_z
    odf_actor_x.GetProperty().SetRoughness(slider.value)
    odf_actor_y.GetProperty().SetRoughness(slider.value)
    odf_actor_z.GetProperty().SetRoughness(slider.value)


def change_slice_opacity(slider):
    global odf_actor_x, odf_actor_y, odf_actor_z
    odf_actor_x.set_opacity(slider.value)
    odf_actor_y.set_opacity(slider.value)
    odf_actor_z.set_opacity(slider.value)


def change_slice_x(slider):
    global odf_actor_x
    value = int(np.rint(slider.value))
    odf_actor_x.slice_along_axis(value, 'xaxis')


def change_slice_y(slider):
    global odf_actor_y
    value = int(np.rint(slider.value))
    odf_actor_y.slice_along_axis(value, 'yaxis')


def change_slice_z(slider):
    global odf_actor_z
    value = int(np.rint(slider.value))
    odf_actor_z.slice_along_axis(value, 'zaxis')


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


def uniforms_callback(_caller, _event, calldata=None):
    global absorption, ior_1, ior_2
    if calldata is not None:
        calldata.SetUniformf('absorption', absorption)
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
    global absorption, control_panel, ior_1, ior_2, odf_actor_x, odf_actor_y, \
        odf_actor_z, pbr_panel, size

    fetch_viz_cubemaps()

    texture_name = 'skybox'
    #texture_name = 'brudslojan'
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
    #scene.skybox(gamma_correct=False)

    #scene.background((1, 1, 1))
    #scene.background((0, 0, 0))

    # Scene rotation for brudslojan texture
    #scene.yaw(-110)

    fetch_viz_dmri()

    fname = read_viz_dmri('fodf.nii.gz')

    sh, affine = load_nifti(fname)
    grid_shape = sh.shape[:-1]

    sphere = get_sphere('repulsion724')
    b_mat = sh_to_sf_matrix(sphere, 8, return_inv=False)

    scale = .75
    norm = False
    colormap = None
    radial_scale = True
    opacity = 1.
    global_cm = False

    # ODF slicer for axial slice
    odf_actor_z = actor.odf_slicer(sh, affine=affine, sphere=sphere,
                                   scale=scale, norm=norm,
                                   radial_scale=radial_scale, opacity=opacity,
                                   colormap=colormap, global_cm=global_cm,
                                   B_matrix=b_mat)
    odf_polydata_z = odf_actor_z.GetMapper().GetInput()
    update_polydata_normals(odf_polydata_z)

    # ODF slicer for coronal slice
    odf_actor_y = actor.odf_slicer(sh, affine=affine, sphere=sphere,
                                   scale=scale, norm=norm,
                                   radial_scale=radial_scale, opacity=opacity,
                                   colormap=colormap, global_cm=global_cm,
                                   B_matrix=b_mat)
    odf_actor_y.display_extent(0, grid_shape[0] - 1, grid_shape[1] // 2,
                               grid_shape[1] // 2, 0, grid_shape[2] - 1)
    odf_polydata_y = odf_actor_y.GetMapper().GetInput()
    update_polydata_normals(odf_polydata_y)

    # ODF slicer for sagittal slice
    odf_actor_x = actor.odf_slicer(sh, affine=affine, sphere=sphere,
                                   scale=scale, norm=norm,
                                   radial_scale=radial_scale, opacity=opacity,
                                   colormap=colormap, global_cm=global_cm,
                                   B_matrix=b_mat)
    odf_actor_x.display_extent(grid_shape[0] // 2, grid_shape[0] // 2, 0,
                               grid_shape[1] - 1, 0, grid_shape[2] - 1)
    odf_polydata_x = odf_actor_x.GetMapper().GetInput()
    update_polydata_normals(odf_polydata_x)

    # Actor rotation for brudslojan texture
    #rotate(odf_actor_z, rotation=(-110, 0, 1, 0))

    scene.add(odf_actor_z)
    scene.add(odf_actor_y)
    scene.add(odf_actor_x)

    scene.reset_camera()
    # scene.zoom(1.9)  # Glyptotek's zoom

    ior_1 = 1.  # Air
    # ior_1 = 1.333  # Water(20 °C)
    ior_2 = 1.5  # Glass
    # ior_2 = .18  # Silver
    # ior_2 = .47  # Gold
    # ior_2 = 1.  # Air
    # ior_2 = 2.33  # Platinum

    absorption = 2

    odf_actor_z.GetProperty().SetInterpolationToPBR()
    odf_actor_y.GetProperty().SetInterpolationToPBR()
    odf_actor_x.GetProperty().SetInterpolationToPBR()

    roughness = 0
    odf_actor_z.GetProperty().SetRoughness(roughness)
    odf_actor_y.GetProperty().SetRoughness(roughness)
    odf_actor_x.GetProperty().SetRoughness(roughness)

    add_shader_callback(odf_actor_z, uniforms_callback)
    add_shader_callback(odf_actor_y, uniforms_callback)
    add_shader_callback(odf_actor_x, uniforms_callback)

    fs_dec_code = load('refractive_dec.frag')
    fs_impl_code = load('refractive_impl.frag')

    # shader_to_actor(odf_actor_z, 'vertex', debug=True)
    # shader_to_actor(odf_actor_z, 'fragment', debug=True)

    shader_to_actor(odf_actor_z, 'fragment', decl_code=fs_dec_code)
    shader_to_actor(odf_actor_y, 'fragment', decl_code=fs_dec_code)
    shader_to_actor(odf_actor_x, 'fragment', decl_code=fs_dec_code)

    shader_to_actor(odf_actor_z, 'fragment', impl_code=fs_impl_code,
                    block='light')
    shader_to_actor(odf_actor_y, 'fragment', impl_code=fs_impl_code,
                    block='light')
    shader_to_actor(odf_actor_x, 'fragment', impl_code=fs_impl_code,
                    block='light')

    # window.show(scene)

    show_m = window.ShowManager(scene=scene, size=(1920, 1080),
                                reset_camera=False, order_transparent=True)
    show_m.initialize()

    pbr_panel = ui.Panel2D((400, 230), position=(5, 5), color=(.25, .25, .25),
                           opacity=.75, align='right')

    panel_label_refractive_pbr = ui.TextBlock2D(
        text='Refractive PBR', font_size=18, bold=True)
    slider_label_ior_1 = ui.TextBlock2D(text='IoR1', font_size=16)
    slider_label_ior_2 = ui.TextBlock2D(text='IoR2', font_size=16)
    slider_label_absorption = ui.TextBlock2D(text='Absorption', font_size=16)
    slider_label_roughness = ui.TextBlock2D(text='Roughness', font_size=16)

    label_pad_x = .06

    pbr_panel.add_element(panel_label_refractive_pbr, (.02, .90))
    pbr_panel.add_element(slider_label_ior_1, (label_pad_x, .70))
    pbr_panel.add_element(slider_label_ior_2, (label_pad_x, .50))
    pbr_panel.add_element(slider_label_absorption, (label_pad_x, .30))
    pbr_panel.add_element(slider_label_roughness, (label_pad_x, .10))

    length = 260
    text_template = '{value:.1f}'

    slider_slice_ior_1 = ui.LineSlider2D(
        initial_value=ior_1, min_value=.1, max_value=5, length=length,
        text_template=text_template)
    slider_slice_ior_2 = ui.LineSlider2D(
        initial_value=ior_2, min_value=.1, max_value=5, length=length,
        text_template=text_template)
    slider_slice_absorption = ui.LineSlider2D(
        initial_value=absorption, max_value=5, length=length,
        text_template=text_template)
    slider_slice_roughness = ui.LineSlider2D(
        initial_value=roughness, max_value=1, length=length,
        text_template=text_template)

    slider_slice_ior_1.on_change = change_slice_ior_1
    slider_slice_ior_2.on_change = change_slice_ior_2
    slider_slice_absorption.on_change = change_slice_absorption
    slider_slice_roughness.on_change = change_slice_roughness

    slice_pad_x = .28

    pbr_panel.add_element(slider_slice_ior_1, (slice_pad_x, .70))
    pbr_panel.add_element(slider_slice_ior_2, (slice_pad_x, .50))
    pbr_panel.add_element(slider_slice_absorption, (slice_pad_x, .30))
    pbr_panel.add_element(slider_slice_roughness, (slice_pad_x, .10))

    scene.add(pbr_panel)

    control_panel = ui.Panel2D((400, 230), position=(5, 240),
                               color=(.25, .25, .25), opacity=.75,
                               align='right')

    panel_label_control = ui.TextBlock2D(
        text='Control', font_size=18, bold=True)
    slider_label_opacity = ui.TextBlock2D(text='Opacity', font_size=16)
    slider_label_x = ui.TextBlock2D(text='Sagittal', font_size=16)
    slider_label_y = ui.TextBlock2D(text='Coronal', font_size=16)
    slider_label_z = ui.TextBlock2D(text='Axial', font_size=16)

    control_panel.add_element(panel_label_control, (.02, .90))
    control_panel.add_element(slider_label_x, (label_pad_x, .70))
    control_panel.add_element(slider_label_y, (label_pad_x, .50))
    control_panel.add_element(slider_label_z, (label_pad_x, .30))
    control_panel.add_element(slider_label_opacity, (label_pad_x, .10))

    slider_slice_x = ui.LineSlider2D(
        initial_value=grid_shape[0] // 2, max_value=grid_shape[0] - 1,
        length=length, text_template='{value:.0f}')
    slider_slice_y = ui.LineSlider2D(
        initial_value=grid_shape[1] // 2, max_value=grid_shape[1] - 1,
        length=length, text_template='{value:.0f}')
    slider_slice_z = ui.LineSlider2D(
        initial_value=grid_shape[2] // 2, max_value=grid_shape[2] - 1,
        length=length, text_template='{value:.0f}')
    slider_slice_opacity = ui.LineSlider2D(
        initial_value=opacity, max_value=1, length=length,
        text_template=text_template)

    slider_slice_x.on_change = change_slice_x
    slider_slice_y.on_change = change_slice_y
    slider_slice_z.on_change = change_slice_z
    slider_slice_opacity.on_change = change_slice_opacity

    control_panel.add_element(slider_slice_x, (slice_pad_x, .70))
    control_panel.add_element(slider_slice_y, (slice_pad_x, .50))
    control_panel.add_element(slider_slice_z, (slice_pad_x, .30))
    control_panel.add_element(slider_slice_opacity, (slice_pad_x, .10))

    scene.add(control_panel)

    size = scene.GetSize()

    show_m.add_window_callback(win_callback)

    show_m.start()
