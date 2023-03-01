import os

import numpy as np
from nilearn import datasets, surface

from fury import actor, window
from fury.data import fetch_viz_models, read_viz_models
from fury.io import load_polydata
from fury.lib import PolyData
from fury.shaders import compose_shader, import_fury_shader, shader_to_actor
from fury.utils import (
    get_actor_from_polydata,
    get_polydata_colors,
    get_polydata_triangles,
    get_polydata_vertices,
    normals_from_v_f,
    rotate,
    set_polydata_colors,
    set_polydata_normals,
    set_polydata_triangles,
    set_polydata_vertices,
)


def get_hemisphere_actor():
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
    points, triangles = surface.load_surf_mesh(fsaverage.infl_right)
    polydata = PolyData()
    set_polydata_vertices(polydata, points)
    set_polydata_triangles(polydata, triangles)
    hemi_actor = get_actor_from_polydata(polydata)
    rotate(hemi_actor, rotation=(-90, 0, 0, 1))
    rotate(hemi_actor, rotation=(-80, 1, 0, 0))
    return hemi_actor


def key_pressed(obj, event):
    global show_m
    key = obj.GetKeySym()
    if key == 's' or key == 'S':
        print('Saving image...')
        show_m.save_screenshot('screenshot.png', magnification=4)
        print('Image saved.')


def load_3D_model(model='molar.stl'):
    if model != 'molar.stl':
        fetch_viz_models()
    model = read_viz_models(model)
    polydata = load_polydata(model)
    verts = get_polydata_vertices(polydata)
    faces = get_polydata_triangles(polydata)
    norms = normals_from_v_f(verts, faces)
    set_polydata_normals(polydata, norms)
    model_actor = get_actor_from_polydata(polydata)
    rotate(model_actor, rotation=(-90, 1, 0, 0))
    return model_actor


def win_callback(obj, event):
    global control_panel, size
    if size != obj.GetSize():
        size_old = size
        size = obj.GetSize()
        size_change = [size[0] - size_old[0], 0]


if __name__ == '__main__':
    scene = window.Scene()
    scene.background((1, 1, 1))
    
    #obj_actor = actor.sphere([[0, 0, 0]], (0, 0, 0), radii=2, theta=64, phi=64)
    #obj_actor = get_hemisphere_actor()
    obj_actor = load_3D_model()
    
    pow5 = import_fury_shader(os.path.join('utils', 'pow5.glsl'))
    
    schlick_weight = import_fury_shader(
        os.path.join('lighting', 'schlick_weight.frag')
    )
    
    blinn_phong_model = import_fury_shader(
        os.path.join('lighting', 'blinn_phong_model.frag')
    )
    
    fs_decl = compose_shader([pow5, schlick_weight, blinn_phong_model])
    
    shader_to_actor(obj_actor, 'fragment', decl_code=fs_decl)
    
    normal = 'vec3 normal = normalVCVSOutput;'
    view = 'vec3 view = normalize(-vertexVC.xyz);'
    dot_n_v = 'float dotNV = clamp(dot(normal, view), 1e-5, 1);'
    fsw = 'float fsw = schlickWeight(dotNV);'
    frag_output = \
        """
        vec3 color = vec3(0);
        if(fsw <= .0022)
        {
            color = vec3(1, .5529, .5529);
            //color = vec3(1, .8510, .8510);
        }
        else
        {
            if(fsw <= .22)
            {
                color = vec3(1, 1, .5529);
                //color = vec3(1, 1, .8510);
            }
            else
            {
                color = vec3(.5529, 1, .5529);
                //color = vec3(.8510, 1, .8510);
            }
        }
        /*
        color = blinnPhongIllumModel(
            dotNV, lightColor0, color, specularPower, specularColor,
            ambientColor);
        */
        fragOutput0 = vec4(color, opacity);
        """
    
    fs_impl = compose_shader([normal, view, dot_n_v, fsw, frag_output])
    
    shader_to_actor(obj_actor, 'fragment', impl_code=fs_impl, block='light')
    
    scene.add(obj_actor)
    
    show_m = window.ShowManager(
        scene=scene, size=(1920, 1080), reset_camera=False,
        order_transparent=True)
    
    show_m.iren.AddObserver('KeyPressEvent', key_pressed)
    
    size = scene.GetSize()
    
    show_m.add_window_callback(win_callback)
    
    show_m.start()
