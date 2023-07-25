import gzip
from datetime import timedelta
from time import time

import numpy as np
from matplotlib import cm
from nibabel import gifti
from nilearn import datasets, surface

from fury import window
from fury.lib import PolyData
from fury.utils import (
    get_actor_from_polydata,
    get_polydata_normals,
    normals_from_v_f,
    set_polydata_colors,
    set_polydata_normals,
    set_polydata_triangles,
    set_polydata_vertices,
    update_polydata_normals,
)


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
        if type(colors) == str:
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


if __name__ == '__main__':
    scene = window.Scene()
    scene.background((1, 1, 1))

    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

    left_pial_mesh = surface.load_surf_mesh(fsaverage.pial_left)
    left_sulc_points = points_from_gzipped_gifti(fsaverage.sulc_left)
    print(len(left_sulc_points))

    right_pial_mesh = surface.load_surf_mesh(fsaverage.pial_right)
    right_sulc_points = points_from_gzipped_gifti(fsaverage.sulc_right)
    print(len(right_sulc_points))

    print('Computing background colors...')
    t = time()
    left_colors = compute_background_colors(left_sulc_points)
    print('Time: {}'.format(timedelta(seconds=time() - t)))

    t = time()
    right_colors = compute_background_colors(right_sulc_points)
    print('Time: {}'.format(timedelta(seconds=time() - t)))

    left_hemi_actor = get_hemisphere_actor(
        fsaverage.pial_left, colors=left_colors)

    right_hemi_actor = get_hemisphere_actor(
        fsaverage.pial_right, colors=right_colors)

    """
    left_hemi_actor.GetProperty().SetRepresentationToSurface()
    left_hemi_actor.GetProperty().EdgeVisibilityOn()
    left_hemi_actor.GetProperty().SetEdgeColor(0, 0, 0)

    right_hemi_actor.GetProperty().SetRepresentationToSurface()
    right_hemi_actor.GetProperty().EdgeVisibilityOn()
    right_hemi_actor.GetProperty().SetEdgeColor(0, 0, 0)
    """

    scene.add(left_hemi_actor)
    scene.add(right_hemi_actor)

    scene.roll(90)
    scene.pitch(80)

    scene.reset_camera()
    scene.reset_clipping_range()

    window.show(scene)
