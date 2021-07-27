from dipy.io.image import load_nifti
from fury import actor, colormap, ui, window
from fury.data import read_viz_textures
from fury.utils import (colors_from_actor, get_actor_from_polydata,
                        set_polydata_colors, set_polydata_vertices,
                        set_polydata_triangles, update_actor)
from fury.shaders import add_shader_callback, load, shader_to_actor
from nibabel.nifti1 import Nifti1Image
from nilearn import surface
from vtk.util import numpy_support


import numpy as np
import os
import vtk


_FSAVG_DIR = '/run/media/guaje/Data/Data/repo_files/fsaverage/fsaverage/'
_NEUROVAULT_DIR = '/run/media/guaje/Data/Data/repo_files/pipelines/fMRI/' \
                  'NeuroVault-10426/'
_HAXBY_DIR = '/run/media/guaje/Data/Data/repo_files/Haxby_2001/subj2/'

_MOTOR_FNAME = os.path.join(_NEUROVAULT_DIR,
                            'task001_left_vs_right_motor.nii.gz')

_BOLD_FNAME = os.path.join(_HAXBY_DIR, 'bold.nii.gz')

_HEMI_DICT = {
    'left': {
        'infl': os.path.join(_FSAVG_DIR, 'infl_left.gii.gz'),
        'pial': os.path.join(_FSAVG_DIR, 'pial_left.gii.gz'),
        'sulc': os.path.join(_FSAVG_DIR, 'sulc_left.gii.gz')
    },
    'right': {
        'infl': os.path.join(_FSAVG_DIR, 'infl_right.gii.gz'),
        'pial': os.path.join(_FSAVG_DIR, 'pial_right.gii.gz'),
        'sulc': os.path.join(_FSAVG_DIR, 'sulc_right.gii.gz')
    }}


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
    global right_hemi_actor, right_textures, volume
    volume = int(np.round(slider.value))
    fmri_nii = Nifti1Image(fmri_img[..., volume], fmri_affine)
    right_vtk_colors = right_hemi_actor.GetMapper().GetInput().GetPointData().\
        GetArray('colors')
    right_colors = numpy_support.vtk_to_numpy(right_vtk_colors)
    #right_colors[:] = calculate_colors(fmri_nii, right_pial_mesh)[:]
    #right_vtk_colors.Modified()


def colors_from_texture(texture, cmap='viridis'):
    return colormap.create_colormap(texture, name=cmap) * 255


def compute_textures(img, affine, mesh, num_volumes):
    if num_volumes == 1:
        nifti = Nifti1Image(img, affine)
        return surface.vol_to_surf(nifti, mesh)[:, None]
    else:
        textures = np.empty((mesh[0].shape[0], num_volumes))
        for i in range(num_volumes):
            print('Computing texture N°{:2d}'.format(i + 1))
            nifti = Nifti1Image(img[..., i], affine)
            textures[:, i] = surface.vol_to_surf(nifti, mesh)
        return textures


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
        right_textures, size, volume

    right_pial_mesh = surface.load_surf_mesh(_HEMI_DICT['right']['pial'])

    #fmri_img, fmri_affine = load_nifti(_MOTOR_FNAME)
    fmri_img, fmri_affine = load_nifti(_BOLD_FNAME)
    img_shape = fmri_img.shape
    volume = 0
    #num_volumes = img_shape[3] if len(img_shape) == 4 else 1
    num_volumes = 10

    right_textures = compute_textures(fmri_img, fmri_affine, right_pial_mesh,
                                      num_volumes)

    right_colors = colors_from_texture(right_textures[:, volume])

    right_hemi_actor = get_hemisphere_actor(_HEMI_DICT['right']['infl'],
                                            colors=right_colors)

    ior_1 = 1.  # Air
    # ior_1 = 1.333  # Water(20 °C)
    ior_2 = 1.5  # Glass
    # ior_2 = .18  # Silver
    # ior_2 = .47  # Gold
    # ior_2 = 1.  # Air
    # ior_2 = 2.33  # Platinum

    right_hemi_actor.GetProperty().SetInterpolationToPBR()
    #metallic = .0
    metallic = right_hemi_actor.GetProperty().GetMetallic()
    roughness = .0
    #roughness = right_surf_actor.GetProperty().GetRoughness()
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

    cubemap_fns = [read_viz_textures('waterfall-skybox-px.jpg'),
                   read_viz_textures('waterfall-skybox-nx.jpg'),
                   read_viz_textures('waterfall-skybox-py.jpg'),
                   read_viz_textures('waterfall-skybox-ny.jpg'),
                   read_viz_textures('waterfall-skybox-pz.jpg'),
                   read_viz_textures('waterfall-skybox-nz.jpg')]

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

    scene.add(right_hemi_actor)
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
