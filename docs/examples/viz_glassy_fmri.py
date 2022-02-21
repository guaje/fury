from datetime import timedelta
from dipy.io.image import load_nifti
from fury import ui, window
from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.io import load_cubemap_texture
from fury.lib import ImageData, PolyData, Texture, numpy_support
from fury.utils import (get_actor_from_polydata, rotate, set_polydata_colors,
                        set_polydata_vertices, set_polydata_triangles)
from fury.shaders import add_shader_callback, load, shader_to_actor
from matplotlib import cm
from nibabel import gifti
from nibabel.nifti1 import Nifti1Image
from nilearn import datasets, surface
from nilearn.image import index_img
import pandas as pd
from time import time


import gzip
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
    global right_hemi_actor
    right_hemi_actor.GetProperty().SetRoughness(slider.value)


def change_slice_opacity(slider):
    global right_hemi_actor
    right_hemi_actor.GetProperty().SetOpacity(slider.value)


def change_slice_volume(slider):
    global right_colors, right_hemi_actor, right_max_val, volume
    val = int(np.round(slider.value))
    if volume != val:
        volume = val
        right_vtk_colors = right_hemi_actor.GetMapper().GetInput().\
            GetPointData().GetArray('colors')
        actor_right_colors = numpy_support.vtk_to_numpy(right_vtk_colors)
        actor_right_colors[:] = right_colors[:, volume, :]
        right_vtk_colors.Modified()


def compute_background_colors(bg_data, bg_cmap='bone_r'):
    bg_data_shape = bg_data.shape
    bg_cmap = cm.get_cmap(bg_cmap)
    bg_min = np.min(bg_data)
    bg_max = np.max(bg_data)
    bg_diff = bg_max - bg_min
    bg_colors = np.empty((bg_data_shape[0], 3))
    for i in range(bg_data_shape[0]):
        # Normalize background data between [0, 1]
        val = (bg_data[i] - bg_min) / bg_diff
        bg_colors[i] = np.array(bg_cmap(val))[:3]
    bg_colors *= 255
    return bg_colors


def compute_textures(img, affine, mesh, volumes=1, radius=3):
    if type(volumes) == int:
        if volumes == 1:
            nifti = Nifti1Image(img, affine)
            return surface.vol_to_surf(nifti, mesh, radius=radius)[:, None]
        else:
            volumes = np.arange(volumes)
    num_vols = len(volumes)
    textures = np.empty((mesh[0].shape[0], len(volumes)))
    for idx, vol in enumerate(volumes):
        print('Computing texture for volume ({:02d}/{}): {:4d}'.format(
            idx + 1, num_vols, vol + 1))
        nifti = Nifti1Image(img[..., vol], affine)
        textures[:, idx] = surface.vol_to_surf(nifti, mesh, radius=radius)
    return textures


def compute_texture_colors(textures, max_val, cmap='bwr'):
    color_cmap = cm.get_cmap(cmap)
    textures_shape = textures.shape
    colors = np.empty(textures_shape + (3,))
    if textures_shape[1] == 1:
        print('Computing colors from texture...')
        for i in range(textures_shape[0]):
            # Normalize values between [0, 1]
            val = (textures[i] + max_val) / (2 * max_val)
            colors[i] = np.array(color_cmap(val))[0, :3]
    else:
        for j in range(textures_shape[1]):
            print('Computing colors for texture {:02d}/{}'.format(
                j + 1, textures_shape[1]))
            for i in range(textures_shape[0]):
                # Normalize values between [0, 1]
                val = (textures[i, j] + max_val) / (2 * max_val)
                colors[i, j] = np.array(color_cmap(val))[:3]
    colors *= 255
    return colors


def data_from_neurovault_motor_task():
    motor_imgs = datasets.fetch_neurovault_motor_task()
    fmri_img, fmri_affine = load_nifti(motor_imgs.images[0])
    return fmri_img, fmri_affine


def data_from_haxby():
    haxby_dataset = datasets.fetch_haxby()
    labels = pd.read_csv(haxby_dataset.session_target[0], sep=' ')
    condition_mask_1 = labels['labels'] == 'face'
    condition_mask_2 = labels['labels'] == 'cat'
    nifti_img_1 = index_img(haxby_dataset.func[0], condition_mask_1)
    nifti_img_2 = index_img(haxby_dataset.func[0], condition_mask_2)
    fmri_affine = nifti_img_1.affine
    fmri_img_1 = np.asanyarray(nifti_img_1.dataobj)
    fmri_img_2 = np.asanyarray(nifti_img_2.dataobj)
    fmri_img_max_1 = np.max(fmri_img_1)
    fmri_img_max_2 = np.max(fmri_img_2)
    fmri_img_max = np.max([fmri_img_max_1, fmri_img_max_2])
    fmri_img_1 = fmri_img_1 / fmri_img_max
    fmri_img_2 = fmri_img_2 / fmri_img_max
    fmri_img = fmri_img_1 - fmri_img_2
    return fmri_img, fmri_affine


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


def get_hemisphere_actor(fname, colors=None):
    points, triangles = surface.load_surf_mesh(fname)
    polydata = PolyData()
    set_polydata_vertices(polydata, points)
    set_polydata_triangles(polydata, triangles)
    set_polydata_colors(polydata, colors)
    return get_actor_from_polydata(polydata)


def points_from_gzipped_gifti(fname):
    with gzip.open(fname) as f:
        as_bytes = f.read()
    parser = gifti.GiftiImage.parser()
    parser.parse(as_bytes)
    gifti_img = parser.img
    return gifti_img.darrays[0].data


def threshold_colors(textures, texture_colors, background_colors=None,
                     threshold=0):
    if threshold == 0:
        return texture_colors
    else:
        colors = np.ones(texture_colors.shape) * 255
        for j in range(textures.shape[1]):
            for i in range(textures.shape[0]):
                if -thr < textures[i, j] < thr:
                    if background_colors is not None:
                        colors[i, j] = background_colors[i, j]
                    continue
                colors[i, j] = texture_colors[i, j]
        return colors


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
    global absorption, control_panel, ior_1, ior_2, pbr_panel, \
        right_colors, right_hemi_actor, right_max_val, size, thr, volume

    fetch_viz_cubemaps()

    texture_name = 'skybox'
    #texture_name = 'brudslojan'
    textures = read_viz_cubemap(texture_name)

    cubemap = load_cubemap_texture(textures)

    """
    # NOTE: Test ndarray texture to cubemap
    cubemap_img = 255 * np.random.randn(512, 512, 3)
    cubemap_img[:256] = np.array([255, 0, 0])
    cubemap_img = cubemap_img.astype(np.uint8)
    """

    # NOTE: Test px, nx, py, ny, pz, nz
    #img_shape = (1024, 1024)
    #img_255s = 255 * np.ones(img_shape).astype(np.uint8)

    """
    img_zeros = np.zeros(img_shape).astype(np.uint8)
    cubemap_red_img = np.stack((img_255s, img_zeros, img_zeros), axis=-1)
    cubemap_green_img = np.stack((img_zeros, img_255s, img_zeros), axis=-1)
    cubemap_blue_img = np.stack((img_zeros, img_zeros, img_255s), axis=-1)
    """

    """
    # Flip horizontally
    img_grad = np.flip(np.tile(np.linspace(0, 255, num=img_shape[0]),
                               (img_shape[1], 1)).astype(np.uint8), axis=1)
    img_grad = np.tile(np.linspace(0, 200, num=img_shape[0]),
                       (img_shape[1], 1)).astype(np.uint8)
    """

    """
    cubemap_red_img = np.stack((img_255s, img_grad, img_grad), axis=-1)
    cubemap_green_img = np.stack((img_grad, img_255s, img_grad), axis=-1)
    cubemap_blue_img = np.stack((img_grad, img_grad, img_255s), axis=-1)
    """

    """
    cubemap_side_img = np.stack((img_grad,) * 3, axis=-1)

    cubemap_bottom_img = np.ones((img_shape[0], img_shape[1], 3)).astype(
        np.uint8) * 200

    cubemap_top_img = np.zeros((img_shape[0], img_shape[1], 3)).astype(
        np.uint8)

    cubemap_imgs = [cubemap_side_img, cubemap_side_img, cubemap_top_img,
                    cubemap_bottom_img, cubemap_side_img, cubemap_side_img]
    """

    """
    cubemap_imgs = [cubemap_red_img, cubemap_red_img, cubemap_green_img,
                    cubemap_green_img, cubemap_blue_img, cubemap_blue_img]
    """

    """
    tile_size = 16
    num_tiles = 64
    num_tiles = int(num_tiles // 2)
    checkerboard = np.kron(
        [[1, 0] * num_tiles, [0, 1] * num_tiles] * num_tiles,
        np.ones((tile_size, tile_size)))
    checkerboard *= 255
    checkerboard_img = checkerboard_img.astype(np.uint8)
    checkerboard_img = np.stack((checkerboard,) * 3, axis=-1)
    cubemap_imgs = [checkerboard_img, checkerboard_img, checkerboard_img,
                    checkerboard_img, checkerboard_img, checkerboard_img]
    """

    #cubemap = get_cubemap_from_ndarrays(cubemap_imgs, flip=False)

    #cubemap.RepeatOff()
    #cubemap.EdgeClampOn()

    scene = window.Scene(skybox=cubemap)
    scene.skybox(visible=False)
    #scene.skybox(gamma_correct=False)

    scene.background((1, 1, 1))

    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')

    right_pial_mesh = surface.load_surf_mesh(fsaverage.pial_right)
    right_sulc_points = points_from_gzipped_gifti(fsaverage.sulc_right)

    fmri_img, fmri_affine = data_from_neurovault_motor_task()
    right_textures = compute_textures(fmri_img, fmri_affine, right_pial_mesh)
    num_volumes = 1

    """
    fmri_img, fmri_affine = data_from_haxby()
    img_shape = fmri_img.shape
    num_volumes = 10
    # NOTE: Evenly spaced N volumes
    volumes = np.rint(np.linspace(0, img_shape[3] - 1,
                                  num=num_volumes)).astype(int)
    t = time()
    right_textures = compute_textures(fmri_img, fmri_affine, right_pial_mesh,
                                      volumes)
    print('Time: {}'.format(timedelta(seconds=time() - t)))
    """

    """
    from nilearn.plotting import plot_surf_stat_map
    import matplotlib.pyplot as plt
    #plot_surf_stat_map(fsaverage.infl_right, right_textures,
    plot_surf_stat_map(fsaverage.infl_right, right_textures[:, 9],
                       #hemi='right', colorbar=True, threshold=1,
                       hemi='right', colorbar=True, threshold=.01,
                       cmap='seismic', bg_map=fsaverage.sulc_right)
    plt.show()
    """

    volume = 0

    print('Computing background colors...')
    t = time()
    right_bg_colors = compute_background_colors(right_sulc_points)
    print('Time: {}'.format(timedelta(seconds=time() - t)))

    right_max_val = np.max(np.abs(right_textures))

    t = time()
    right_tex_colors = compute_texture_colors(right_textures, right_max_val)
    print('Time: {}'.format(timedelta(seconds=time() - t)))

    #thr = right_max_val * .1
    #thr = .01
    thr = 1

    print('Thresholding colors...')
    t = time()
    right_colors = threshold_colors(
        right_textures, right_tex_colors, #threshold=thr)
        background_colors=right_bg_colors, threshold=thr)
    print('Time: {}'.format(timedelta(seconds=time() - t)))

    right_hemi_actor = get_hemisphere_actor(fsaverage.infl_right,
                                            colors=right_colors[:, volume, :])

    # Scene rotation for skybox texture
    #scene.pitch(-25)  # Look down for a darker IBL background

    # Scene rotation for brudslojan texture
    #scene.yaw(-110)

    view = 'right lateral'
    if view == 'right lateral':
        rotate(right_hemi_actor, rotation=(-90, 0, 0, 1))
        rotate(right_hemi_actor, rotation=(-80, 1, 0, 0))

    # Actor rotation for skybox texture
    #rotate(right_hemi_actor, rotation=(-25, 1, 0, 0))

    # Actor rotation for brudslojan texture
    #rotate(right_hemi_actor, rotation=(-110, 0, 1, 0))

    ior_1 = 1.  # Air
    # ior_1 = 1.333  # Water(20 °C)
    ior_2 = 1.5  # Glass
    # ior_2 = .18  # Silver
    # ior_2 = .47  # Gold
    # ior_2 = 1.  # Air
    # ior_2 = 2.33  # Platinum

    absorption = 2

    right_hemi_actor.GetProperty().SetInterpolationToPBR()
    roughness = .0
    right_hemi_actor.GetProperty().SetRoughness(roughness)

    opacity = 1.
    right_hemi_actor.GetProperty().SetOpacity(opacity)

    add_shader_callback(right_hemi_actor, uniforms_callback)

    fs_dec_code = load('refractive_dec.frag')
    fs_impl_code = load('refractive_impl.frag')

    #shader_to_actor(right_surf_actor, 'vertex', debug=True)
    #shader_to_actor(right_surf_actor, 'fragment', debug=True)
    shader_to_actor(right_hemi_actor, 'fragment', decl_code=fs_dec_code)
    shader_to_actor(right_hemi_actor, 'fragment', impl_code=fs_impl_code,
                    block='light')

    scene.add(right_hemi_actor)

    """
    if view == 'right lateral':
        scene.roll(-90)
        scene.pitch(87)
    """

    scene.reset_camera()
    scene.reset_clipping_range()

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

    control_panel = ui.Panel2D((400, 130), position=(5, 240),
                               color=(.25, .25, .25), opacity=.75,
                               align='right')

    panel_label_control = ui.TextBlock2D(
        text='Control', font_size=18, bold=True)
    slider_label_volume = ui.TextBlock2D(text='Volume', font_size=16)
    slider_label_opacity = ui.TextBlock2D(text='Opacity', font_size=16)

    control_panel.add_element(panel_label_control, (.02, .8))
    if right_colors.shape[1] > 1:
        control_panel.add_element(slider_label_volume, (label_pad_x, .55))
    control_panel.add_element(slider_label_opacity, (label_pad_x, .2))

    slider_slice_volume = ui.LineSlider2D(
        initial_value=volume,
        max_value=num_volumes if num_volumes == 1 else num_volumes - 1,
        length=length,
        text_template='{value:.0f}')

    slider_slice_opacity = ui.LineSlider2D(
        initial_value=opacity, max_value=1, length=length,
        text_template=text_template)

    slider_slice_volume.on_change = change_slice_volume
    slider_slice_opacity.on_change = change_slice_opacity

    if right_colors.shape[1] > 1:
        control_panel.add_element(slider_slice_volume, (slice_pad_x, .55))
    control_panel.add_element(slider_slice_opacity, (slice_pad_x, .2))

    scene.add(control_panel)

    size = scene.GetSize()

    show_m.add_window_callback(win_callback)

    show_m.start()
