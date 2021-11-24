from dipy.io.image import load_nifti
from fury import actor, ui, window
from fury.data import read_viz_textures
from fury.utils import (get_actor_from_polydata, rotate, set_polydata_colors,
                        set_polydata_vertices, set_polydata_triangles)
from fury.shaders import add_shader_callback, load, shader_to_actor
from matplotlib import cm
from nibabel import gifti
from nibabel.nifti1 import Nifti1Image
from nilearn import datasets, surface
from nilearn.image import index_img
import pandas as pd
from vtk.util import numpy_support


import gzip
import numpy as np
import os
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
    global right_hemi_actor
    right_hemi_actor.GetProperty().SetOcclusionStrength(slider.value)


def change_slice_ior_1(slider):
    global ior_1
    ior_1 = slider.value


def change_slice_ior_2(slider):
    global ior_2
    ior_2 = slider.value


def change_slice_metallic(slider):
    global right_hemi_actor
    right_hemi_actor.GetProperty().SetMetallic(slider.value)


def change_slice_roughness(slider):
    global right_hemi_actor
    right_hemi_actor.GetProperty().SetRoughness(slider.value)


def change_slice_opacity(slider):
    global right_hemi_actor
    right_hemi_actor.GetProperty().SetOpacity(slider.value)


def change_slice_volume(slider):
    global right_hemi_actor, right_max_val, right_textures, volume
    val = int(np.round(slider.value))
    if volume != val:
        volume = val
        right_vtk_colors = right_hemi_actor.GetMapper().GetInput().\
            GetPointData().GetArray('colors')
        right_colors = numpy_support.vtk_to_numpy(right_vtk_colors)
        texture = right_textures[:, volume]
        right_colors[:] = colors_from_texture(texture, right_max_val)[:]
        right_vtk_colors.Modified()


def colors_from_texture(texture, max_val, cmap='seismic', thr=None,
                        bg_data=None, bg_cmap='gray_r'):
    color_cmap = cm.get_cmap(cmap)
    colors = np.empty((texture.shape[0], 3))
    if thr is not None and bg_data is not None:
        bg_cmap = cm.get_cmap(bg_cmap)
        bg_min = np.min(bg_data)
        bg_max = np.max(bg_data)
        bg_diff = bg_max - bg_min
    for i in range(texture.shape[0]):
        if thr is not None and bg_data is not None:
            if -thr <= texture[i] <= thr:
                # Normalize background data between [0, 1]
                val = (bg_data[i] - bg_min) / bg_diff
                colors[i] = np.array(bg_cmap(val))[:3]
                continue
        # Normalize values between [0, 1]
        val = (texture[i] + max_val) / (2 * max_val)
        colors[i] = np.array(color_cmap(val))[:3]
    colors *= 255
    return colors


def compute_textures(img, affine, mesh, volumes, radius=3):
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


def get_cubemap_from_ndarrays(array, flip=True):
    texture = vtk.vtkTexture()
    texture.CubeMapOn()
    for idx, img in enumerate(array):
        vtk_img = vtk.vtkImageData()
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


def get_hemisphere_actor(fname, colors=None):
    points, triangles = surface.load_surf_mesh(fname)
    polydata = vtk.vtkPolyData()
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
    global control_panel, ior_1, ior_2, pbr_panel, right_hemi_actor, \
        right_max_val, right_textures, size, volume

    scene = window.Scene()

    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
    #motor_imgs = datasets.fetch_neurovault_motor_task()
    haxby_dataset = datasets.fetch_haxby()

    right_pial_mesh = surface.load_surf_mesh(fsaverage.pial_right)
    #right_infl_mesh = surface.load_surf_mesh(fsaverage.infl_right)
    right_sulc_points = points_from_gzipped_gifti(fsaverage.sulc_right)

    #fmri_img, fmri_affine = load_nifti(motor_imgs.images[0])

    labels = pd.read_csv(haxby_dataset.session_target[0], sep=' ')
    condition_mask_face = labels['labels'] == 'face'
    condition_mask_house = labels['labels'] == 'house'
    nifti_img_face = index_img(haxby_dataset.func[0], condition_mask_face)
    nifti_img_house = index_img(haxby_dataset.func[0], condition_mask_house)
    fmri_affine = nifti_img_face.affine
    fmri_img_face = np.asanyarray(nifti_img_face.dataobj)
    fmri_img_house = np.asanyarray(nifti_img_house.dataobj)
    fmri_img_face_max = np.max(fmri_img_face)
    fmri_img_house_max = np.max(fmri_img_house)
    fmri_img_max = np.max([fmri_img_face_max, fmri_img_house_max])
    fmri_img_face = fmri_img_face / fmri_img_max
    fmri_img_house = fmri_img_house / fmri_img_max
    fmri_img = fmri_img_face - fmri_img_house

    img_shape = fmri_img.shape
    volume = 0
    num_volumes = img_shape[3] if len(img_shape) == 4 else 1
    num_volumes = 10
    #num_volumes = 1

    # NOTE: Evenly spaced N volumes
    volumes = np.rint(np.linspace(0, img_shape[3] - 1,
                                  num=num_volumes)).astype(int)

    right_textures = compute_textures(fmri_img, fmri_affine, right_pial_mesh,
                                      #num_volumes)
                                      volumes)

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

    right_max_val = np.max(np.abs(right_textures))
    #right_max_val = np.max(np.abs(scores))

    # TODO: Pre-compute colors for each texture
    right_colors = colors_from_texture(
        #right_textures[:, volume], right_max_val, thr=1,
        right_textures[:, volume], right_max_val, thr=.01,
        bg_data=right_sulc_points)
        #scores - chance, right_max_val, thr=.1, bg_data=right_sulc_points)

    right_hemi_actor = get_hemisphere_actor(fsaverage.infl_right,
                                            colors=right_colors)

    view = 'right lateral'
    if view == 'right lateral':
        rotate(right_hemi_actor, rotation=(-90, 0, 0, 1))
        rotate(right_hemi_actor, rotation=(-80, 1, 0, 0))

    ior_1 = 1.  # Air
    # ior_1 = 1.333  # Water(20 °C)
    ior_2 = 1.5  # Glass
    # ior_2 = .18  # Silver
    # ior_2 = .47  # Gold
    # ior_2 = 1.  # Air
    # ior_2 = 2.33  # Platinum

    right_hemi_actor.GetProperty().SetInterpolationToPBR()
    metallic = .0
    roughness = .0
    emissive_factor = right_hemi_actor.GetProperty().GetEmissiveFactor()
    ao_strength = right_hemi_actor.GetProperty().GetOcclusionStrength()

    right_hemi_actor.GetProperty().SetMetallic(metallic)
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

    #texture_name = 'skybox'
    texture_name = 'brudslojan'
    cubemap_fns = [read_viz_textures(texture_name + '-px.jpg'),
                   read_viz_textures(texture_name + '-nx.jpg'),
                   read_viz_textures(texture_name + '-py.jpg'),
                   read_viz_textures(texture_name + '-ny.jpg'),
                   read_viz_textures(texture_name + '-pz.jpg'),
                   read_viz_textures(texture_name + '-nz.jpg')]
    # Load the cube map
    cubemap = get_cubemap(cubemap_fns)

    """
    # NOTE: Test ndarray texture to cubemap
    cubemap_img = 255 * np.random.randn(512, 512, 3)
    cubemap_img[:256] = np.array([255, 0, 0])
    cubemap_img = cubemap_img.astype(np.uint8)
    """

    # NOTE: Test px, nx, py, ny, pz, nz
    img_shape = (1024, 1024)
    #img_255s = 255 * np.ones(img_shape).astype(np.uint8)

    """
    img_zeros = np.zeros(img_shape).astype(np.uint8)
    cubemap_red_img = np.stack((img_255s, img_zeros, img_zeros), axis=-1)
    cubemap_green_img = np.stack((img_zeros, img_255s, img_zeros), axis=-1)
    cubemap_blue_img = np.stack((img_zeros, img_zeros, img_255s), axis=-1)
    """

    # Flip horizontally
    #img_grad = np.flip(np.tile(np.linspace(0, 255, num=img_shape[0]),
    #                           (img_shape[1], 1)).astype(np.uint8), axis=1)
    img_grad = np.tile(np.linspace(0, 50, num=img_shape[0]),
                       (img_shape[1], 1)).astype(np.uint8)

    """
    cubemap_red_img = np.stack((img_255s, img_grad, img_grad), axis=-1)
    cubemap_green_img = np.stack((img_grad, img_255s, img_grad), axis=-1)
    cubemap_blue_img = np.stack((img_grad, img_grad, img_255s), axis=-1)
    """

    """
    cubemap_side_img = np.stack((img_grad,) * 3, axis=-1)

    cubemap_bottom_img = np.ones((img_shape[0], img_shape[1], 3)).astype(
        np.uint8) * 50

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

    # Load the skybox
    skybox = cubemap
    skybox.InterpolateOn()
    skybox.RepeatOff()
    skybox.EdgeClampOn()

    skybox_actor = vtk.vtkSkybox()
    skybox_actor.SetTexture(skybox)

    scene.UseImageBasedLightingOn()
    if vtk.vtkVersion.GetVTKMajorVersion() >= 9:
        scene.SetEnvironmentTexture(cubemap)
    else:
        scene.SetEnvironmentCubeMap(cubemap)

    scene.add(right_hemi_actor)

    scene.add(skybox_actor)
    #scene.background((1, 1, 1))

    """
    if view == 'right lateral':
        scene.roll(-90)
        scene.pitch(87)
    """

    scene.reset_camera()
    scene.reset_clipping_range()

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

    control_panel = ui.Panel2D((320, 130), position=(-25, 510),
                               color=(.25, .25, .25), opacity=.75,
                               align='right')

    panel_label_control = build_label('Control', font_size=18,
                                      bold=True)
    slider_label_volume = build_label('Volume')
    slider_label_opacity = build_label('Opacity')

    control_panel.add_element(panel_label_control, (.02, .8))
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

    control_panel.add_element(slider_slice_volume, (slice_pad_x, .55))
    control_panel.add_element(slider_slice_opacity, (slice_pad_x, .2))

    scene.add(control_panel)

    size = scene.GetSize()

    show_m.add_window_callback(win_callback)

    show_m.start()
