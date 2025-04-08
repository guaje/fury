import os

import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf

from fury import actor
from fury.lib import FloatArray, Texture
from fury.shaders import (
    attribute_to_actor,
    compose_shader,
    import_fury_shader,
    shader_to_actor,
)
from fury.texture.utils import uv_calculations
from fury.utils import (
    minmax_norm,
    numpy_to_vtk_image_data,
    set_polydata_tcoords,
)


def sh_odf_calc(centers, coeffs, sphere_type, scales, opacity):
    """
    Visualize one or many ODFs with different features.

    Parameters
    ----------
    centers : ndarray(N, 3)
        ODFs positions.
    coeffs : ndarray
        2D ODFs array in SH coefficients.
    scales : float or ndarray (N, )
        ODFs size.
    opacity : float
        Takes values from 0 (fully transparent) to 1 (opaque).

    Returns
    -------
    box_actor: Actor

    """
    odf_actor = actor.box(centers=centers, scales=1)
    odf_actor.GetMapper().SetVBOShiftScaleMethod(False)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(odf_actor, big_centers, "center")

    big_scales = np.repeat(scales, 8, axis=0)
    attribute_to_actor(odf_actor, big_scales, "scale")

    minmax = np.array([coeffs.min(axis=1), coeffs.max(axis=1)]).T
    big_minmax = np.repeat(minmax, 8, axis=0)
    attribute_to_actor(odf_actor, big_minmax, "minmax")

    sphere = get_sphere(name=sphere_type)
    sh_basis = "descoteaux07"
    n_coeffs = coeffs.shape[-1]
    sh_order = int((np.sqrt(8 * n_coeffs + 1) - 3) / 2)

    n_glyphs = coeffs.shape[0]

    sh = np.zeros((n_glyphs, 1, 1, n_coeffs))
    for i in range (n_glyphs):
        sh[i, 0, 0, :] = coeffs[i, :]

    tensor_sf = sh_to_sf(
        sh, sh_order_max=sh_order, basis_type=sh_basis, sphere=sphere
    )
    tensor_sf_max = abs(tensor_sf.reshape(n_glyphs, 100)).max(axis=1)

    sfmax = np.array(tensor_sf_max)
    big_sfmax = np.repeat(sfmax, 8, axis=0)
    attribute_to_actor(odf_actor, big_sfmax, "sfmax")

    odf_actor_pd = odf_actor.GetMapper().GetInput()

    uv_vals = np.array(uv_calculations(n_glyphs))
    num_pnts = uv_vals.shape[0]

    t_coords = FloatArray()
    t_coords.SetNumberOfComponents(2)
    t_coords.SetNumberOfTuples(num_pnts)
    [t_coords.SetTuple(i, uv_vals[i]) for i in range(num_pnts)]

    set_polydata_tcoords(odf_actor_pd, t_coords)

    arr = minmax_norm(coeffs)
    arr *= 255
    grid = numpy_to_vtk_image_data(arr.astype(np.uint8))

    texture = Texture()
    texture.SetInputDataObject(grid)
    texture.Update()

    odf_actor.GetProperty().SetTexture("texture0", texture)

    def_coeff = f"#define NCOEFF {n_coeffs}"

    # TODO: Set int uniform
    odf_actor.GetShaderProperty().GetFragmentCustomUniforms().SetUniformf(
        "numCoeffs", n_coeffs
    )

    vs_dec = """
        in vec3 center;
        in float scale;
        in vec2 minmax;
        in float sfmax;

        out vec4 vertexMCVSOutput;
        out vec3 centerMCVSOutput;
        out float scaleVSOutput;
        out vec2 minmaxVSOutput;
        out float sfmaxVSOutput;
        out vec3 camPosMCVSOutput;
    """

    vs_impl = """
        vertexMCVSOutput = vertexMC;
        centerMCVSOutput = center;
        scaleVSOutput = scale;
        minmaxVSOutput = minmax;
        sfmaxVSOutput = sfmax;
        vec3 camPos = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
    """

    shader_to_actor(odf_actor, "vertex", decl_code=vs_dec, impl_code=vs_impl)

    fs_defs = "#define PI 3.1415926535898"

    fs_unifs = """
        uniform mat4 MCVCMatrix;
    """

    fs_vs_vars = """
        in vec4 vertexMCVSOutput;
        in vec3 centerMCVSOutput;
        in float scaleVSOutput;
        in vec2 minmaxVSOutput;
        in float sfmaxVSOutput;
    """

    coeffs_norm = import_fury_shader(os.path.join("utils", "minmax_norm.glsl"))

    # Functions needed to calculate the associated Legendre polynomial
    factorial = import_fury_shader(os.path.join("utils", "factorial.glsl"))

    # Adapted from https://patapom.com/blog/SHPortal/
    # "Evaluate an Associated Legendre Polynomial P(l,m,x) at x
    # For more, see “Numerical Methods in C: The Art of Scientific Computing”,
    # Cambridge University Press, 1992, pp 252-254"
    legendre_polys = import_fury_shader(
        os.path.join("utils", "legendre_polynomial.glsl")
    )

    norm_const = import_fury_shader(
        os.path.join("utils", "sh_normalization_factor.glsl")
    )

    spherical_harmonics = import_fury_shader(
        os.path.join("ray_tracing", "odf", "sh_function.glsl")
    )

    sdf_map_1 = """
    vec3 map( in vec3 p )
    {
        p = p - centerMCVSOutput;
        vec3 p00 = p;

        float r, d; vec3 n, s, res;

        #define SHAPE (vec3(d-abs(r), sign(r),d))
        d=length(p00);
        n=p00 / d;
        float i = 1 / (numCoeffs * 2);
        float shCoeffs[NCOEFF];
        float maxCoeff = 0.0;
        for(int j=0; j < numCoeffs; j++){
            shCoeffs[j] = rescale(
                texture(
                    texture0,
                    vec2(i + j / numCoeffs, tcoordVCVSOutput.y)).x, 0, 1,
                    minmaxVSOutput.x, minmaxVSOutput.y
            );
        }
    """

    sh_list = """r = shCoeffs[0] * SH(0, 0, n);
        r += shCoeffs[1] * SH(2, -2, n);
        r += shCoeffs[2] * SH(2, -1, n);
        r += shCoeffs[3] * SH(2, 0, n);
        r += shCoeffs[4] * SH(2, 1, n);
        r += shCoeffs[5] * SH(2, 2, n);
        r += shCoeffs[6] * SH(4, -4, n);
        r += shCoeffs[7] * SH(4, -3, n);
        r += shCoeffs[8] * SH(4, -2, n);
        r += shCoeffs[9] * SH(4, -1, n);
        r += shCoeffs[10] * SH(4, 0, n);
        r += shCoeffs[11] * SH(4, 1, n);
        r += shCoeffs[12] * SH(4, 2, n);
        r += shCoeffs[13] * SH(4, 3, n);
        r += shCoeffs[14] * SH(4, 4, n);
        r += shCoeffs[15] * SH(6, -6, n);
        r += shCoeffs[16] * SH(6, -5, n);
        r += shCoeffs[17] * SH(6, -4, n);
        r += shCoeffs[18] * SH(6, -3, n);
        r += shCoeffs[19] * SH(6, -2, n);
        r += shCoeffs[20] * SH(6, -1, n);
        r += shCoeffs[21] * SH(6, 0, n);
        r += shCoeffs[22] * SH(6, 1, n);
        r += shCoeffs[23] * SH(6, 2, n);
        r += shCoeffs[24] * SH(6, 3, n);
        r += shCoeffs[25] * SH(6, 4, n);
        r += shCoeffs[26] * SH(6, 5, n);
        r += shCoeffs[27] * SH(6, 6, n);
        r += shCoeffs[28] * SH(8, -8, n);
        r += shCoeffs[29] * SH(8, -7, n);
        r += shCoeffs[30] * SH(8, -6, n);
        r += shCoeffs[31] * SH(8, -5, n);
        r += shCoeffs[32] * SH(8, -4, n);
        r += shCoeffs[33] * SH(8, -3, n);
        r += shCoeffs[34] * SH(8, -2, n);
        r += shCoeffs[35] * SH(8, -1, n);
        r += shCoeffs[36] * SH(8, 0, n);
        r += shCoeffs[37] * SH(8, 1, n);
        r += shCoeffs[38] * SH(8, 2, n);
        r += shCoeffs[39] * SH(8, 3, n);
        r += shCoeffs[40] * SH(8, 4, n);
        r += shCoeffs[41] * SH(8, 5, n);
        r += shCoeffs[42] * SH(8, 6, n);
        r += shCoeffs[43] * SH(8, 7, n);
        r += shCoeffs[44] * SH(8, 8, n);
    """
    sdf_map_2 = """
        r /= abs(sfmaxVSOutput);
        r *= scaleVSOutput * .9;

        s = SHAPE;
        res=s;
        return vec3(res.x, .5 + .5 * res.y, res.z);
    }
    """

    sdf_map = sdf_map_1 + "\n".join(sh_list.splitlines()[:n_coeffs]) + sdf_map_2
    #sdf_map = import_fury_shader(os.path.join("sdf", "sd_spherical_harmonics.#frag"))

    cast_ray = import_fury_shader(
        os.path.join("ray_tracing", "odf", "sh_cast_ray.frag")
    )

    central_diffs_normals = """
    vec3 centralDiffsNormals( in vec3 pos )
    {
        const vec2 eps = vec2(0.0001,0.0);

        return normalize( vec3(
                                map(pos+eps.xyy).x - map(pos-eps.xyy).x,
                                map(pos+eps.yxy).x - map(pos-eps.yxy).x,
                                map(pos+eps.yyx).x - map(pos-eps.yyx).x ) );
    }
    """

    # Applies the non-linearity that maps linear RGB to sRGB
    linear_to_srgb = import_fury_shader(
        os.path.join("lighting", "linear_to_srgb.frag")
    )

    # Inverse of linear_to_srgb()
    srgb_to_linear = import_fury_shader(
        os.path.join("lighting", "srgb_to_linear.frag")
    )

    # Turns a linear RGB color (i.e. rec. 709) into sRGB
    linear_rgb_to_srgb = import_fury_shader(
        os.path.join("lighting", "linear_rgb_to_srgb.frag")
    )

    # Inverse of linear_rgb_to_srgb()
    srgb_to_linear_rgb = import_fury_shader(
        os.path.join("lighting", "srgb_to_linear_rgb.frag")
    )

    # Logarithmic tonemapping operator. Input and output are linear RGB.
    tonemap = import_fury_shader(os.path.join("lighting", "tonemap.frag"))

    blinn_phong_model = import_fury_shader(
        os.path.join("lighting", "blinn_phong_model.frag")
    )

    # fmt: off
    fs_dec = compose_shader([
        fs_defs, fs_unifs, fs_vs_vars, coeffs_norm, factorial, legendre_polys,
        norm_const, spherical_harmonics, def_coeff, sdf_map, cast_ray,
        central_diffs_normals, linear_to_srgb, srgb_to_linear,
        linear_rgb_to_srgb, srgb_to_linear_rgb, tonemap, blinn_phong_model
    ])
    # fmt: on

    shader_to_actor(odf_actor, "fragment", decl_code=fs_dec, debug=False)

    sdf_frag_impl = """
        vec3 pnt = vertexMCVSOutput.xyz;

        vec3 ro = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);

        vec3 rd = normalize(pnt - ro);

        vec3 ld = normalize(ro - pnt);

        ro += pnt - ro;

        vec3 t = castRay(ro, rd);

        if(t.y > -.5)
        {
            vec3 pos = ro - centerMCVSOutput + t.x * rd;
            vec3 colorDir = srgbToLinearRgb(abs(normalize(pos)));
            fragOutput0 = vec4(colorDir, opacity);
        }
        else
        {
            discard;
        }
    """

    shader_to_actor(
        odf_actor, "fragment", impl_code=sdf_frag_impl, block="picking"
    )

    return odf_actor
