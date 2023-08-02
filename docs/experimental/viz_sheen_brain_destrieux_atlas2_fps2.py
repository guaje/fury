import gzip
import os
from datetime import timedelta
from time import time

import numpy as np
from matplotlib import cm
from nibabel import gifti
from nilearn import datasets, surface
from nilearn.connectome import ConnectivityMeasure

from fury import actor, window
from fury.lib import VTK_OBJECT, PolyData, calldata_type
from fury.shaders import (
    add_shader_callback,
    compose_shader,
    import_fury_shader,
    shader_to_actor,
)
from fury.utils import (
    get_actor_from_polydata,
    get_polydata_normals,
    normals_from_v_f,
    rotate,
    set_polydata_colors,
    set_polydata_normals,
    set_polydata_triangles,
    set_polydata_vertices,
    update_polydata_normals,
)


def colors_from_pre_cmap(textures, networks, pre_cmap, bg_colors=None):
    colors = np.zeros((textures.shape[0], 3))
    for i in range(textures.shape[0]):
        label = textures[i][0]
        if label > 0:
            idx = np.where(label == networks + 1)[0]
            if len(idx) > 0:
                idx = idx[0]
                colors[i] = pre_cmap[idx] * 255
        else:
            if bg_colors is not None:
                colors[i] = bg_colors[i]
            else:
                continue
    return colors


def compute_background_colors(bg_data, bg_cmap='gray_r'):
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


def get_hemisphere_actor(fname, colors=None, auto_normals='vtk'):
    points, triangles = surface.load_surf_mesh(fname)
    polydata = PolyData()
    set_polydata_vertices(polydata, points)
    set_polydata_triangles(polydata, triangles)
    if auto_normals.lower() == 'vtk':
        update_polydata_normals(polydata)
    elif auto_normals.lower() == 'fury':
        normals = normals_from_v_f(points, triangles)
        set_polydata_normals(polydata, normals)
    if colors is not None:
        if isinstance(colors, str):
            if colors.lower() == 'normals':
                if auto_normals.lower() == 'vtk':
                    normals = get_polydata_normals(polydata)
                if normals is not None:
                    colors = (normals + 1) / 2 * 255
        set_polydata_colors(polydata, colors)
    return get_actor_from_polydata(polydata)


def points_from_gzipped_gifti(fname):
    with gzip.open(fname) as f:
        as_bytes = f.read()
    parser = gifti.GiftiImage.parser()
    parser.parse(as_bytes)
    gifti_img = parser.img
    return gifti_img.darrays[0].data


def stylize_actor_with_sheen(actor, sheen=1, sheen_tint=1):
    sheen_params = {'sheen': sheen, 'sheen_tint': sheen_tint}

    @calldata_type(VTK_OBJECT)
    def uniforms_callback(_caller, _event, calldata=None):
        if calldata is not None:
            calldata.SetUniformf('sheen', sheen_params['sheen'])
            calldata.SetUniformf('sheenTint', sheen_params['sheen_tint'])

    add_shader_callback(actor, uniforms_callback)

    uniforms = \
    """
    uniform float sheen;
    uniform float sheenTint;
    """

    pow5 = import_fury_shader(os.path.join('utils', 'pow5.glsl'))

    gamma_to_linear = import_fury_shader(
        os.path.join('lighting', 'gamma_to_linear.frag')
    )

    cie_color_tint = import_fury_shader(
        os.path.join('lighting', 'cie_color_tint.frag')
    )

    schlick_weight = import_fury_shader(
        os.path.join('lighting', 'schlick_weight.frag')
    )

    sheen = import_fury_shader(
        os.path.join('lighting', 'principled', 'sheen.frag')
    )

    fs_dec = compose_shader([
        uniforms, pow5, gamma_to_linear, cie_color_tint, schlick_weight, sheen
    ])

    shader_to_actor(actor, 'fragment', decl_code=fs_dec)

    normal = 'vec3 normal = normalVCVSOutput;'
    view = 'vec3 view = normalize(-vertexVC.xyz);'
    dot_n_v = 'float dotNV = clamp(dot(normal, view), 1e-5, 1);'

    dot_n_v_validation = \
    """
    if(dotNV < 0)
        fragOutput0 = vec4(vec3(0), opacity);
    """

    linear_color = 'vec3 linColor = gamma2Linear(diffuseColor);'

    tint = 'vec3 tint = calculateTint(linColor);'

    fsw = 'float fsw = schlickWeight(dotNV);'

    sheen_rad = \
    """
    vec3 sheenRad = evaluateSheen(sheen, sheenTint, tint, fsw);
    """

    # NOTE: Here's the sheen modification
    frag_output = \
    """
    float len = length(sheenRad);
    //fragOutput0 = vec4(sheenRad, opacity);
    //fragOutput0 = vec4(sheenRad, len);
    if(len < .1)
    {
        discard;
    }
    else
    {
        fragOutput0 = vec4(sheenRad, len);
    }
    """

    fs_impl = compose_shader([
        normal, view, dot_n_v, dot_n_v_validation, linear_color, tint, fsw,
        sheen_rad, frag_output
    ])

    shader_to_actor(actor, 'fragment', impl_code=fs_impl, block='light')

    return sheen_params


def timer_callback(_obj, _event):
    global avg_fpss, fpss, prev_time, show_m, start_time

    fpss.append(show_m.frame_rate)
    fpss.pop(0)

    n_secs = 10

    time_diff = timedelta(seconds=time() - start_time)
    zoom_lvl = np.sin(time_diff.seconds / n_secs * 360) + 1.5

    show_m.scene.azimuth(5)

    show_m.render()

    # Runs for X seconds
    if time_diff.seconds > n_secs:
        show_m.exit()
    else:
        if time_diff.seconds > prev_time:
            avg_fps = np.mean(fpss)
            print(f'{time_diff} - {np.rint(avg_fps)} fps')
            avg_fpss.append(avg_fps)
            show_m.scene.zoom(zoom_lvl)
        prev_time = time_diff.seconds


if __name__ == '__main__':
    global avg_fpss, fpss, prev_time, show_m, start_time

    scene = window.Scene()

    scene_bg_color = (1, 1, 1)
    scene.background(scene_bg_color)

    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
    destrieux_labels = destrieux_atlas.labels
    left_parcellation = destrieux_atlas.map_left
    right_parcellation = destrieux_atlas.map_right

    _, left_unique_count = np.unique(left_parcellation, return_counts=True)
    _, right_unique_count = np.unique(right_parcellation, return_counts=True)

    #n_top = 13
    n_top = 24
    n_top_left_nets = np.argpartition(left_unique_count, -n_top)[-n_top:]
    n_top_right_nets = np.argpartition(right_unique_count, -n_top)[-n_top:]

    n_top_net = []
    for i in range(n_top):
        if n_top_left_nets[i] in n_top_right_nets:
            n_top_net.append(n_top_left_nets[i])
    n_top_net = np.array(n_top_net)
    num_top_net = len(n_top_net)

    n_top_net_labels = []
    n_top_left_parcellation = np.zeros(left_parcellation.shape, dtype=int)
    n_top_right_parcellation = np.zeros(right_parcellation.shape, dtype=int)
    for idx, label in enumerate(n_top_net):
        label += 1
        n_top_left_parcellation[left_parcellation == label] = label
        n_top_right_parcellation[right_parcellation == label] = label
        label += 1
        n_top_net_labels.append(destrieux_labels[label])
    n_top_net_labels = np.array(n_top_net_labels)

    fsaverage = datasets.fetch_surf_fsaverage()

    left_pial_mesh = surface.load_surf_mesh(fsaverage.pial_left)
    left_sulc_points = points_from_gzipped_gifti(fsaverage.sulc_left)

    right_pial_mesh = surface.load_surf_mesh(fsaverage.pial_right)
    right_sulc_points = points_from_gzipped_gifti(fsaverage.sulc_right)

    print('Computing background colors...')
    t = time()
    left_bg_colors = compute_background_colors(left_sulc_points)
    right_bg_colors = compute_background_colors(right_sulc_points)
    print('Time: {}'.format(timedelta(seconds=time() - t)))

    """
    left_excluded_colors = np.unique(left_bg_colors, axis=0)
    right_excluded_colors = np.unique(right_bg_colors, axis=0)

    excluded_colors = np.unique(np.vstack((
        left_excluded_colors, right_excluded_colors)), axis=0) / 255

    n_top_net_colors = distinguishable_colormap(
        bg=scene_bg_color, exclude=excluded_colors, nb_colors=len(n_top_net))
    n_top_net_colors = np.array(n_top_net_colors)
    """

    #n_top_net_cmap = cm.get_cmap('Paired')
    n_top_net_cmap = cm.get_cmap('tab20b')
    n_top_net_colors = np.array([
        n_top_net_cmap(i / (num_top_net - 1))[:3]
        for i in range(num_top_net)])

    fmri_data = datasets.fetch_surf_nki_enhanced(n_subjects=1)
    left_ts = surface.load_surf_data(fmri_data.func_left[0])
    right_ts = surface.load_surf_data(fmri_data.func_right[0])
    num_time_pnts = left_ts.shape[1]
    num_time_series = num_top_net * 2

    time_series = np.empty((num_time_pnts, num_time_series))
    label_coords = np.empty((num_time_series, 3))
    for idx, label in enumerate(n_top_net):
        label += 1
        # Left time series and node coordinates per label
        label_ts = left_ts[n_top_left_parcellation == label, :]
        time_series[:, idx] = np.mean(label_ts, axis=0)
        coords = left_pial_mesh.coordinates[
            n_top_left_parcellation == label, :]
        label_coords[idx, :] = np.mean(coords, axis=0)
        # Right time series and node coordinates per label
        label_ts = right_ts[n_top_right_parcellation == label, :]
        time_series[:, idx + num_top_net] = np.mean(label_ts, axis=0)
        coords = right_pial_mesh.coordinates[
            n_top_right_parcellation == label, :]
        label_coords[idx + num_top_net, :] = np.mean(coords, axis=0)

    corr_measure = ConnectivityMeasure(kind='correlation')
    corr_matrix = corr_measure.fit_transform([time_series])[0]

    min_coords = np.min(label_coords, axis=0)
    max_coords = np.max(label_coords, axis=0)

    max_val = np.max(np.abs(corr_matrix[~np.eye(num_time_series, dtype=bool)]))
    pos_edges_cmap = cm.get_cmap('summer_r')
    neg_edges_cmap = cm.get_cmap('autumn')

    #hemi_thr = 0
    hemi_thr = max_coords[0]
    thr = .7

    edges_coords = []
    edges_colors = []
    valid_corr_values = []
    show_nodes = [False] * num_time_series
    for i in range(num_time_series):
        coord_i = label_coords[i]
        if coord_i[0] < hemi_thr:
            for j in range(i + 1, num_time_series):
                coord_j = label_coords[j]
                if coord_j[0] < hemi_thr:
                    if corr_matrix[i, j] > thr:
                        show_nodes[i] = True
                        show_nodes[j] = True
                        valid_corr_values.append(corr_matrix[i, j])
                        edges_coords.append([label_coords[i], label_coords[j]])
                        val = (corr_matrix[i, j] - thr) / (max_val - thr)
                        edges_colors.append(pos_edges_cmap(val)[:3])
                    if corr_matrix[i, j] < -thr:
                        show_nodes[i] = True
                        show_nodes[j] = True
                        edges_coords.append([label_coords[i], label_coords[j]])
                        val = (corr_matrix[i, j] + max_val) / (-thr + max_val)
                        edges_colors.append(neg_edges_cmap(val)[:3])
    edges_coords = np.array(edges_coords)
    edges_colors = np.array(edges_colors)
    show_nodes = np.array(show_nodes)

    """
    vis_labels = []
    vis_colors = []
    for i in range(num_top_net):
        #if show_nodes[i]:
        if show_nodes[i] or show_nodes[i + num_top_net]:
            vis_labels.append(n_top_net_labels[i])
            vis_colors.append(n_top_net_colors[i])
    vis_labels = np.array(vis_labels)
    vis_colors = np.round(np.array(vis_colors) * 255).astype(int)
    """

    edges_actor = actor.streamtube(
        edges_coords, edges_colors, opacity=.5, linewidth=.5)

    scene.add(edges_actor)

    nodes_coords = []
    nodes_colors = []
    for i in range(num_top_net):
        net_color = n_top_net_colors[i]
        if show_nodes[i]:
            nodes_coords.append(label_coords[i])
            nodes_colors.append(net_color)
        if show_nodes[i + num_top_net]:
            nodes_coords.append(label_coords[i + num_top_net])
            nodes_colors.append(net_color)
    nodes_coords = np.array(nodes_coords)
    nodes_colors = np.array(nodes_colors)

    nodes_actor = actor.sphere(nodes_coords, nodes_colors, radii=2)

    scene.add(nodes_actor)

    """
    # Background opacities
    left_max_op_vals = -np.nanmin(left_sulc_points)
    left_min_op_vals = -np.nanmax(left_sulc_points)

    left_opacities = ((-left_sulc_points - left_min_op_vals) /
                      (left_max_op_vals - left_min_op_vals)) * 255
    #left_op_colors = np.tile(left_opacities[:, np.newaxis], (1, 3))

    right_max_op_vals = -np.nanmin(right_sulc_points)
    right_min_op_vals = -np.nanmax(right_sulc_points)

    right_opacities = ((-right_sulc_points - right_min_op_vals) /
                       (right_max_op_vals - right_min_op_vals)) * 255
    """

    t = time()
    left_colors = colors_from_pre_cmap(
        n_top_left_parcellation[:, np.newaxis], n_top_net, n_top_net_colors,
        bg_colors=left_bg_colors)
    print('Time: {}'.format(timedelta(seconds=time() - t)))

    #left_colors = np.hstack((left_colors, left_opacities[:, np.newaxis]))

    left_hemi_actor = get_hemisphere_actor(
        fsaverage.pial_left, colors=left_colors)

    t = time()
    right_colors = colors_from_pre_cmap(
        n_top_right_parcellation[:, np.newaxis], n_top_net, n_top_net_colors,
        bg_colors=right_bg_colors)
    print('Time: {}'.format(timedelta(seconds=time() - t)))

    #right_colors = np.hstack((right_colors, right_opacities[:, np.newaxis]))

    right_hemi_actor = get_hemisphere_actor(
        fsaverage.pial_right, colors=right_colors)

    sheen_params = {'sheen': 1, 'sheen_tint': 1}

    left_hemi_style = stylize_actor_with_sheen(left_hemi_actor, **sheen_params)
    right_hemi_style = stylize_actor_with_sheen(
        right_hemi_actor, **sheen_params)

    opacity = 0
    left_hemi_actor.GetProperty().SetOpacity(opacity)
    right_hemi_actor.GetProperty().SetOpacity(opacity)

    scene.add(left_hemi_actor)
    scene.add(right_hemi_actor)

    rotate(edges_actor, rotation=(-80, 1, 0, 0))
    rotate(nodes_actor, rotation=(-80, 1, 0, 0))
    rotate(left_hemi_actor, rotation=(-80, 1, 0, 0))
    rotate(right_hemi_actor, rotation=(-80, 1, 0, 0))

    scene.reset_camera()
    scene.reset_clipping_range()

    #window.show(scene)

    show_m = window.ShowManager(scene=scene, size=(1920, 1080),
                                reset_camera=False, order_transparent=True)
    show_m.initialize()

    prev_time = 0

    fps_len = 20
    fpss = [0] * fps_len

    avg_fpss = []

    show_m.add_timer_callback(True, 1, timer_callback)

    start_time = time()

    show_m.start()

    print(f'Avg. FPS = {np.mean(avg_fpss)}')
