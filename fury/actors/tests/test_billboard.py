import numpy as np
import numpy.testing as npt

from fury import window
from fury.actors.billboard import BillboardActor


def test_fs_injection(interactive=True):
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
        centers, colors=colors, scales=scales, fs_impl=fake_sphere)
    
    scene.add(billboard_actor)

    if interactive:
        window.show(scene)

    arr = window.snapshot(scene)
    report = window.analyze_snapshot(arr, colors=colors)
    npt.assert_equal(report.objects, 8)
