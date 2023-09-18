import numpy as np

from fury import actor, window
from fury.lib import FloatArray, ImageData, Texture, numpy_support
from fury.utils import rgb_to_vtk, set_polydata_tcoords


def np_array_to_vtk_img(data):
    grid = ImageData()
    grid.SetDimensions(data.shape[1], data.shape[0], 1)
    nd = data.shape[-1] if data.ndim == 3 else 1
    vtkarr = numpy_support.numpy_to_vtk(
        np.flip(data.swapaxes(0, 1), axis=1).reshape((-1, nd), order="F")
    )
    vtkarr.SetName("Image")
    grid.GetPointData().AddArray(vtkarr)
    grid.GetPointData().SetActiveScalars("Image")
    grid.GetPointData().Update()
    return grid


if __name__ == "__main__":
    scene = window.Scene()
    scene.background((1, 1, 1))

    centers = np.array(
        [[0, 0, 0], [-2.5, -2, 0], [-1, -2, 0], [1, -2, 0], [2.5, -2, 0]]
    )
    scales = [1, 0.5, 0.5, 0.5, 0.5]

    texture_actor = actor.billboard(centers, colors=(1, 1, 1), scales=scales)

    actor_pd = texture_actor.GetMapper().GetInput()

    # fmt: off
    uv_vals = np.array(
        [
            [0, 0], [0, 1], [1, 1], [1, 0],  # Full texture
            [0, 0.5], [0, 1], [0.5, 1], [0.5, 0.5],  # Top left color
            [0.5, 0.5], [0.5, 1], [1, 1], [1, 0.5],  # Top right color
            [0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0],  # Bottom left color
            [0.5, 0], [0.5, 0.5], [1, 0.5], [1, 0]  # Bottom right color
        ]
    )
    # fmt: on

    num_pnts = uv_vals.shape[0]

    t_coords = FloatArray()
    t_coords.SetNumberOfComponents(2)
    t_coords.SetNumberOfTuples(num_pnts)

    [t_coords.SetTuple(i, uv_vals[i]) for i in range(num_pnts)]

    set_polydata_tcoords(actor_pd, t_coords)

    np.random.seed(8)
    # arr = np.random.randn(2, 2, 3) * 255
    # arr = np.random.randn(2, 2) * 255
    # arr = np.array([[[255, 0, 0], [255, 255, 0]], [[0, 255, 0], [0, 0, 255]]])
    arr = np.array([[0, 83], [167, 250]])

    grid = np_array_to_vtk_img(arr.astype(np.uint8))

    texture = Texture()
    texture.SetInputDataObject(grid)
    texture.Update()

    # NOTE: Configure textures (needs TCoords on the mesh)
    # texture_actor.GetProperty().SetTexture("texture0", texture)
    texture_actor.SetTexture(texture)

    scene.add(texture_actor)

    window.show(scene)
