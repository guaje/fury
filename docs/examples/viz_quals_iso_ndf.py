import os

from fury import actor, ui, window
from fury.lib import VTK_OBJECT, calldata_type
from fury.shaders import (
    add_shader_callback,
    compose_shader,
    import_fury_shader,
    shader_to_actor,
)


def change_slice_roughness(slider):
    global roughness
    roughness = slider.value


def key_pressed(obj, event):
    global show_m
    key = obj.GetKeySym()
    if key == 's' or key == 'S':
        print('Saving image...')
        show_m.save_screenshot('screenshot.png', magnification=4)
        print('Image saved.')


@calldata_type(VTK_OBJECT)
def uniforms_callback(_caller, _event, calldata=None):
    global roughness
    if calldata is not None:
        calldata.SetUniformf('roughness', roughness)


def win_callback(obj, event):
    global control_panel, size
    if size != obj.GetSize():
        size_old = size
        size = obj.GetSize()
        size_change = [size[0] - size_old[0], 0]


if __name__ == '__main__':
    global roughness
    roughness = 0
    
    scene = window.Scene()
    scene.background((1, 1, 1))
    
    obj_actor = actor.sphere([[0, 0, 0]], (0, 0, 0), radii=2, theta=64, phi=64)
    
    add_shader_callback(obj_actor, uniforms_callback)
    
    pi = '#define PI 3.14159265359'
    
    roughness_uniform = 'uniform float roughness;'
    
    square = import_fury_shader(os.path.join('utils', 'square.glsl'))
    
    gtr2 = import_fury_shader(
        os.path.join('lighting', 'ndf', 'gtr2.frag')
    )
    
    fs_decl = compose_shader([pi, roughness_uniform, square, gtr2])
    
    shader_to_actor(obj_actor, 'fragment', decl_code=fs_decl)
    
    normal = 'vec3 normal = normalVCVSOutput;'
    view = 'vec3 view = normalize(-vertexVC.xyz);'
    dot_n_v = 'float dotNV = clamp(dot(normal, view), 1e-5, 1);'
    ndf_gtr2 = 'float ndfGTR2 = GTR2(roughness, dotNV);'
    ndf = 'float ndf = ndfGTR2;'
    frag_output = 'fragOutput0 = vec4(vec3(1) * ndf, opacity);'
    
    fs_impl = compose_shader([
        normal, view, dot_n_v, ndf_gtr2, ndf, frag_output])
    
    shader_to_actor(obj_actor, 'fragment', impl_code=fs_impl, block='light')
    
    scene.add(obj_actor)
    
    show_m = window.ShowManager(
        scene=scene, size=(1920, 1080), reset_camera=False,
        order_transparent=True)
    
    control_panel = ui.Panel2D(
        (400, 90), position=(5, 240), color=(.25, .25, .25), opacity=.75,
        align='right')

    panel_label_control = ui.TextBlock2D(
        text='Control', font_size=18, bold=True)
    slider_label_opacity = ui.TextBlock2D(text='Roughness', font_size=16)
    
    label_pad_x = .06

    control_panel.add_element(panel_label_control, (.02, .70))
    control_panel.add_element(slider_label_opacity, (label_pad_x, .30))
    
    length = 260
    text_template = '{value:.1f}'

    slider_slice_opacity = ui.LineSlider2D(
        initial_value=roughness, max_value=1, length=length,
        text_template=text_template)

    slider_slice_opacity.on_change = change_slice_roughness
    
    slice_pad_x = .28

    control_panel.add_element(slider_slice_opacity, (slice_pad_x, .3))

    scene.add(control_panel)
    
    show_m.iren.AddObserver('KeyPressEvent', key_pressed)
    
    size = scene.GetSize()
    
    show_m.add_window_callback(win_callback)
    
    show_m.start()
