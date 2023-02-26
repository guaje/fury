from fury import actor, material, ui, window
from fury.io import load_cubemap_texture
from fury.utils import (
    normals_from_actor,
    tangents_from_direction_of_anisotropy,
    tangents_to_actor,
)


def change_slice_metallic(slider):
    global pbr_params
    pbr_params.metallic = slider.value


def change_slice_roughness(slider):
    global pbr_params
    pbr_params.roughness = slider.value


def change_slice_anisotropy(slider):
    global pbr_params
    pbr_params.anisotropy = slider.value


def change_slice_anisotropy_direction_x(slider):
    global doa, normals, sphere
    doa[0] = slider.value
    tangents = tangents_from_direction_of_anisotropy(normals, doa)
    tangents_to_actor(sphere, tangents)


def change_slice_anisotropy_direction_y(slider):
    global doa, normals, sphere
    doa[1] = slider.value
    tangents = tangents_from_direction_of_anisotropy(normals, doa)
    tangents_to_actor(sphere, tangents)


def change_slice_anisotropy_direction_z(slider):
    global doa, normals, sphere
    doa[2] = slider.value
    tangents = tangents_from_direction_of_anisotropy(normals, doa)
    tangents_to_actor(sphere, tangents)


def change_slice_anisotropy_rotation(slider):
    global pbr_params
    pbr_params.anisotropy_rotation = slider.value


def change_slice_coat_strength(slider):
    global pbr_params
    pbr_params.coat_strength = slider.value


def change_slice_coat_roughness(slider):
    global pbr_params
    pbr_params.coat_roughness = slider.value


def change_slice_base_ior(slider):
    global pbr_params
    pbr_params.base_ior = slider.value


def change_slice_coat_ior(slider):
    global pbr_params
    pbr_params.coat_ior = slider.value


def key_pressed(obj, event):
    global show_m
    key = obj.GetKeySym()
    if key == 's' or key == 'S':
        print('Saving image...')
        show_m.save_screenshot('screenshot.png', magnification=4)
        print('Image saved.')


def win_callback(obj, event):
    global control_panel, size
    if size != obj.GetSize():
        size_old = size
        size = obj.GetSize()
        size_change = [size[0] - size_old[0], 0]
        control_panel.re_align(size_change)


scene = window.Scene()
scene.background((1, 1, 1))

sphere = actor.sphere([[0, 0, 0]], (0, 0, 0), radii=2, theta=64, phi=64)

doa = [0, 1, 0.5]

normals = normals_from_actor(sphere)
tangents = tangents_from_direction_of_anisotropy(normals, doa)
tangents_to_actor(sphere, tangents)

pbr_params = material.manifest_pbr(sphere, roughness=.5)

scene.add(sphere)

show_m = window.ShowManager(
    scene=scene, size=(1920, 1080), reset_camera=False, order_transparent=True)

control_panel = ui.Panel2D(
    (400, 500), position=(5, 5), color=(0.25, 0.25, 0.25), opacity=0.75,
    align='right')

slider_label_metallic = ui.TextBlock2D(text='Metallic', font_size=16)
slider_label_roughness = ui.TextBlock2D(text='Roughness', font_size=16)
slider_label_anisotropy = ui.TextBlock2D(text='Anisotropy', font_size=16)
slider_label_anisotropy_rotation = ui.TextBlock2D(
    text='Anisotropy Rotation', font_size=16)
slider_label_anisotropy_direction_x = ui.TextBlock2D(
    text='Anisotropy Direction X', font_size=16)
slider_label_anisotropy_direction_y = ui.TextBlock2D(
    text='Anisotropy Direction Y', font_size=16)
slider_label_anisotropy_direction_z = ui.TextBlock2D(
    text='Anisotropy Direction Z', font_size=16)
slider_label_coat_strength = ui.TextBlock2D(
    text='Coat Strength', font_size=16)
slider_label_coat_roughness = ui.TextBlock2D(
    text='Coat Roughness', font_size=16)
slider_label_base_ior = ui.TextBlock2D(text='Base IoR', font_size=16)
slider_label_coat_ior = ui.TextBlock2D(text='Coat IoR', font_size=16)

control_panel.add_element(slider_label_metallic, (0.01, 0.95))
control_panel.add_element(slider_label_roughness, (0.01, 0.86))
control_panel.add_element(slider_label_anisotropy, (0.01, 0.77))
control_panel.add_element(slider_label_anisotropy_rotation, (0.01, 0.68))
control_panel.add_element(slider_label_anisotropy_direction_x, (0.01, 0.59))
control_panel.add_element(slider_label_anisotropy_direction_y, (0.01, 0.5))
control_panel.add_element(slider_label_anisotropy_direction_z, (0.01, 0.41))
control_panel.add_element(slider_label_coat_strength, (0.01, 0.32))
control_panel.add_element(slider_label_coat_roughness, (0.01, 0.23))
control_panel.add_element(slider_label_base_ior, (0.01, 0.14))
control_panel.add_element(slider_label_coat_ior, (0.01, 0.05))

slider_slice_metallic = ui.LineSlider2D(
    initial_value=pbr_params.metallic, max_value=1, length=195,
    text_template='{value:.1f}')
slider_slice_roughness = ui.LineSlider2D(
    initial_value=pbr_params.roughness, max_value=1, length=195,
    text_template='{value:.1f}')
slider_slice_anisotropy = ui.LineSlider2D(
    initial_value=pbr_params.anisotropy, max_value=1, length=195,
    text_template='{value:.1f}')
slider_slice_anisotropy_rotation = ui.LineSlider2D(
    initial_value=pbr_params.anisotropy_rotation, max_value=1, length=195,
    text_template='{value:.1f}')
slider_slice_coat_strength = ui.LineSlider2D(
    initial_value=pbr_params.coat_strength, max_value=1, length=195,
    text_template='{value:.1f}')
slider_slice_coat_roughness = ui.LineSlider2D(
    initial_value=pbr_params.coat_roughness, max_value=1, length=195,
    text_template='{value:.1f}')

slider_slice_anisotropy_direction_x = ui.LineSlider2D(
    initial_value=doa[0], min_value=-1, max_value=1, length=195,
    text_template='{value:.1f}')
slider_slice_anisotropy_direction_y = ui.LineSlider2D(
    initial_value=doa[1], min_value=-1, max_value=1, length=195,
    text_template='{value:.1f}')
slider_slice_anisotropy_direction_z = ui.LineSlider2D(
    initial_value=doa[2], min_value=-1, max_value=1, length=195,
    text_template='{value:.1f}')

slider_slice_base_ior = ui.LineSlider2D(
    initial_value=pbr_params.base_ior, min_value=1, max_value=2.3, length=195,
    text_template='{value:.02f}')
slider_slice_coat_ior = ui.LineSlider2D(
    initial_value=pbr_params.coat_ior, min_value=1, max_value=2.3, length=195,
    text_template='{value:.02f}')

slider_slice_metallic.on_change = change_slice_metallic
slider_slice_roughness.on_change = change_slice_roughness
slider_slice_anisotropy.on_change = change_slice_anisotropy
slider_slice_anisotropy_rotation.on_change = change_slice_anisotropy_rotation
slider_slice_anisotropy_direction_x.on_change = change_slice_anisotropy_direction_x
slider_slice_anisotropy_direction_y.on_change = change_slice_anisotropy_direction_y
slider_slice_anisotropy_direction_z.on_change = change_slice_anisotropy_direction_z
slider_slice_coat_strength.on_change = change_slice_coat_strength
slider_slice_coat_roughness.on_change = change_slice_coat_roughness
slider_slice_base_ior.on_change = change_slice_base_ior
slider_slice_coat_ior.on_change = change_slice_coat_ior

control_panel.add_element(slider_slice_metallic, (0.44, 0.95))
control_panel.add_element(slider_slice_roughness, (0.44, 0.86))
control_panel.add_element(slider_slice_anisotropy, (0.44, 0.77))
control_panel.add_element(slider_slice_anisotropy_rotation, (0.44, 0.68))
control_panel.add_element(slider_slice_anisotropy_direction_x, (0.44, 0.59))
control_panel.add_element(slider_slice_anisotropy_direction_y, (0.44, 0.5))
control_panel.add_element(slider_slice_anisotropy_direction_z, (0.44, 0.41))
control_panel.add_element(slider_slice_coat_strength, (0.44, 0.32))
control_panel.add_element(slider_slice_coat_roughness, (0.44, 0.23))
control_panel.add_element(slider_slice_base_ior, (0.44, 0.14))
control_panel.add_element(slider_slice_coat_ior, (0.44, 0.05))

scene.add(control_panel)

show_m.iren.AddObserver('KeyPressEvent', key_pressed)

size = scene.GetSize()

show_m.add_window_callback(win_callback)

show_m.start()
