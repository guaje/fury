import numpy as np
import numpy.testing as npt

from fury import window
from fury.actors.billboard import BillboardActor


def test_fs_injection(interactive=False):
    scene = window.Scene()
    
    centers = np.array([
        [0, 0, 0], [5, -5, 5], [-7, 7, -7], [10, 10, 10], [10.5, 11.5, 11.5],
        [12, -12, -12], [-17, 17, 17], [-22, -22, 22]])
    colors = np.array([
        [1, 1, 0], [.5, .5, .5], [1, 0, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0],
        [0, 1, 0], [0, 1, 1]])
    scales = [6, 0.4, 1.2, 1, 0.2, 0.7, 3, 2]

    fake_sphere = \
        """
        float len = length(point);
        float radius = 1.;
        if(len > radius)
            discard;
        fragOutput0 = vec4(color, opacity);
    """

    billboard_actor = BillboardActor(
        centers, colors, scales, fs_impl=fake_sphere)
    
    scene.add(billboard_actor)

    if interactive:
        window.show(scene)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 8)


def test_type(interactive=False):
    scene = window.Scene()
    
    centers = np.array([
        [0, 0, 0], [-15, 15, -5], [10, -10, 5], [-30, 30, -10], [20, -20, 10]])
    colors = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0]])
    scales = [3, 1, 2, 1, 1.5]

    bb_point = \
        """
        float len = length(point);
        float radius = .2;
        if(len > radius)
            fragOutput0 = vec4(vec3(0), opacity);
        else
            fragOutput0 = vec4(color, 1);
        """

    bb_type = ['spherical', 'cylindrical_x', 'cylindrical_y']
    expected_val = [True, False, False]
    rotations = [[87, 0, -87, 87], [87, 0, -87, 87], [0, 87, 87, -87]]
    for i in range(3):
        billboard_actor = BillboardActor(
            centers, colors, scales, bb_type=bb_type[i], fs_impl=bb_point)

        scene.add(billboard_actor)
        
        if bb_type[i] == 'spherical':
            arr = window.snapshot(scene)
            report = window.analyze_snapshot(arr, colors=255 * colors)
            npt.assert_equal(report.colors_found, [True] * 5)

        scene.pitch(rotations[i][0])
        scene.yaw(rotations[i][1])
        
        if interactive:
            window.show(scene)

        scene.reset_camera()
        scene.reset_clipping_range()
        arr = window.snapshot(scene, offscreen=True)
        report = window.analyze_snapshot(arr, colors=255 * colors)
        npt.assert_equal(report.colors_found, [True] * 5)

        scene.pitch(rotations[i][2])
        scene.yaw(rotations[i][3])
        if interactive:
            window.show(scene)

        scene.reset_camera()
        scene.reset_clipping_range()
        arr = window.snapshot(scene, offscreen=True)
        report = window.analyze_snapshot(arr, colors=255 * colors)
        npt.assert_equal(report.colors_found, [expected_val[i]] * 5)

        scene.yaw(-87)
        scene.clear()
