import os

from fury import actor, material, ui, window
from fury.shaders import compose_shader, import_fury_shader, shader_to_actor


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


scene = window.Scene()
scene.background((1, 1, 1))

sphere = actor.sphere([[0, 0, 0]], (0, 0, 0), radii=2, theta=64, phi=64)

pow5 = import_fury_shader(os.path.join('utils', 'pow5.glsl'))

schlick_weight = import_fury_shader(
    os.path.join('lighting', 'schlick_weight.frag')
)

fs_decl = compose_shader([pow5, schlick_weight])

shader_to_actor(sphere, 'fragment', decl_code=fs_decl)

normal = 'vec3 normal = normalVCVSOutput;'
view = 'vec3 view = normalize(-vertexVC.xyz);'
dot_n_v = 'float dotNV = clamp(dot(normal, view), 1e-5, 1);'
fsw = 'float fsw = schlickWeight(dotNV);'
frag_output = \
    """
    if(fsw <= .0022)
    {
        fragOutput0 = vec4(vec3(1, 0, 0), opacity);
    }
    else
    {
        if(fsw <= .22)
            fragOutput0 = vec4(vec3(1, 1, 0), opacity);
        else
            fragOutput0 = vec4(vec3(0, 1, 0), opacity);
    }
    """

fs_impl = compose_shader([normal, view, dot_n_v, fsw, frag_output])

shader_to_actor(sphere, 'fragment', impl_code=fs_impl, block='light')

scene.add(sphere)

show_m = window.ShowManager(
    scene=scene, size=(1920, 1080), reset_camera=False, order_transparent=True)

show_m.iren.AddObserver('KeyPressEvent', key_pressed)

size = scene.GetSize()

show_m.add_window_callback(win_callback)

show_m.start()
