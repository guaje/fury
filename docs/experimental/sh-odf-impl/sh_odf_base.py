"""
This script includes TEXTURE experimentation for passing SH coefficients
"""

import os

import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf

from fury import actor, window
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

if __name__ == "__main__":
    show_man = window.ShowManager(size=(1920, 1080))
    show_man.scene.background((1, 1, 1))

    # fmt: off
    coeffs = np.array([
        [
            0.2820735, 0.15236554, -0.04038717, -0.11270988, -0.04532376,
            0.14921817, 0.00257928, 0.0040734, -0.05313807, 0.03486542,
            0.04083064, 0.02105767, -0.04389586, -0.04302812, 0.1048641
        ],
        [
            0.28549338, 0.0978267, -0.11544838, 0.12525354, -0.00126003,
            0.00320594, 0.04744155, -0.07141446, 0.03211689, 0.04711322,
            0.08064896, 0.00154299, 0.00086506, 0.00162543, -0.00444893
        ],
        [
            0.28208936, -0.13133252, -0.04701012, -0.06303016, -0.0468775,
            0.02348355, 0.03991898, 0.02587433, 0.02645416, 0.00668765,
            0.00890633, 0.02189304, 0.00387415, 0.01665629, -0.01427194
        ],
        [
            -0.2739740312099, 0.2526670396328, 1.8922271728516,
            0.2878578901291, -0.5339795947075, -0.2620058953762,
            0.1580424904823, 0.0329004973173, -0.1322413831949,
            -0.1332057565451, 1.0894461870193, -0.6319401264191,
            -0.0416776277125, -1.0772529840469,  0.1423762738705
        ]
    ])
    # fmt: on

    centers = np.array([[0, -1, 0], [1, -1, 0], [2, -1, 0], [3, -1, 0]])
    scales = np.ones(4) * 0.5

    odf_actor = actor.box(centers=centers, scales=1)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(odf_actor, big_centers, "center")

    big_scales = np.repeat(scales, 8, axis=0)
    attribute_to_actor(odf_actor, big_scales, "scale")

    minmax = np.array([coeffs.min(axis=1), coeffs.max(axis=1)]).T
    big_minmax = np.repeat(minmax, 8, axis=0)
    attribute_to_actor(odf_actor, big_minmax, "minmax")

    sphere = get_sphere(name="repulsion100")
    num_verts = sphere.vertices.shape[0]

    num_glyphs = coeffs.shape[0]

    max_num_coeffs = coeffs.shape[-1]
    max_sh_degree = int((np.sqrt(8 * max_num_coeffs + 1) - 3) / 2)
    max_poly_degree = 2 * max_sh_degree + 2
    viz_sh_degree = max_sh_degree

    # TODO: Find a way to avoid reshaping the SH coefficients
    sh = np.zeros((max_sh_degree, 1, 1, max_num_coeffs))
    sh[0, 0, 0, :] = coeffs[0, :]
    sh[1, 0, 0, :] = coeffs[1, :]
    sh[2, 0, 0, :] = coeffs[2, :]
    sh[3, 0, 0, :] = coeffs[3, :]

    sh_basis = "descoteaux07"

    fODFs = sh_to_sf(
        sh, sh_order_max=max_sh_degree, basis_type=sh_basis, sphere=sphere
    )

    max_fODFs = abs(fODFs.reshape(num_glyphs, num_verts)).max(axis=1)
    big_max_fodfs = np.repeat(max_fODFs, 8, axis=0)
    attribute_to_actor(odf_actor, big_max_fodfs, "maxfODFs")

    odf_actor_pd = odf_actor.GetMapper().GetInput()

    uv_vals = np.array(uv_calculations(num_glyphs))
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

    odf_actor.GetShaderProperty().GetVertexCustomUniforms().SetUniformf(
        "shDegree", viz_sh_degree
    )

    vs_dec = """
    in vec3 center;
    in vec2 minmax;
    in float maxfODFs;
    in float scale;

    flat out float numCoeffsVSOutput;

    out vec4 vertexMCVSOutput;
    out vec3 camPosMCVSOutput;
    out vec3 centerMCVSOutput;
    out vec2 minmaxVSOutput;
    out float maxfODFsVSOutput;
    out float scaleVSOutput;
    """

    vs_impl = """
    camPosMCVSOutput = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
    centerMCVSOutput = center;
    maxfODFsVSOutput = maxfODFs;
    minmaxVSOutput = minmax;
    numCoeffsVSOutput = (shDegree + 1) * (shDegree + 2) / 2;
    scaleVSOutput = scale;
    vertexMCVSOutput = vertexMC;
    """

    shader_to_actor(odf_actor, "vertex", decl_code=vs_dec, impl_code=vs_impl)

    max_ray_steps = 250  # TODO: Turn to actor param
    odf_actor.GetShaderProperty().GetFragmentCustomUniforms().SetUniformi(
        "maxRaySteps", max_ray_steps
    )

    fs_def_pi = "#define PI 3.1415926535898"

    fs_def_text_bracket = f"#define TEXT_BRACKET {1 / (2 * max_num_coeffs)}"

    fs_unifs = """
    uniform mat4 MCVCMatrix;
    """

    fs_vs_vars = """
    flat in float numCoeffsVSOutput;

    in vec4 vertexMCVSOutput;
    in vec3 camPosMCVSOutput;
    in vec3 centerMCVSOutput;
    in vec2 minmaxVSOutput;
    in float maxfODFsVSOutput;
    in float scaleVSOutput;
    """

    coeffs_norm = import_fury_shader(os.path.join("utils", "minmax_norm.glsl"))

    # Functions needed to calculate the associated Legendre polynomial
    # TODO: Precompute factorial values
    factorial = import_fury_shader(os.path.join("utils", "factorial.glsl"))

    # Adapted from https://patapom.com/blog/SHPortal/
    # "Evaluate an Associated Legendre Polynomial P(l,m,x) at x
    # For more, see “Numerical Methods in C: The Art of Scientific Computing”,
    # Cambridge University Press, 1992, pp 252-254"
    assoc_legendre_poly = import_fury_shader(
        os.path.join("sdf", "odf", "assoc_legendre_poly.frag")
    )

    norm_fact = import_fury_shader(
        os.path.join("sdf", "odf", "sh_norm_factor.frag")
    )

    spherical_harmonics = import_fury_shader(
        os.path.join("sdf", "odf", "sh_function.frag")
    )

    sdf_map = """
    vec3 map(vec3 p)
    {
        vec3 centeredPnt = p - centerMCVSOutput;

        float pntLen = length(centeredPnt);

        vec3 normPnt = centeredPnt / pntLen;

        int l = 0;  // Order
        int m = 0;  // Degree
        float r = 0.0;
        for(int i = 0; i < numCoeffsVSOutput; i++){
            // TODO: Move to vertex shader
            float textVal = texture(
                texture0,
                vec2(i / numCoeffsVSOutput + TEXT_BRACKET, tcoordVCVSOutput.y)
            ).x;
            // TODO: Move to vertex shader and output to fragment shader
            float rescaledSHCoeff = rescale(
                textVal, 0, 1, minmaxVSOutput.x, minmaxVSOutput.y
            );
            // TODO: Retreive from vertex shader
            r += rescaledSHCoeff * calculateSH(l, m, normPnt);

            if (m == l) {
                l += 2;
                m = -l;
            } else {
                m++;
            }
        }

        r /= maxfODFsVSOutput;
        r *= scaleVSOutput * .9;

        vec3 res = vec3(pntLen - abs(r), sign(r), pntLen);
        return vec3(res.x, 0.5 * res.y + 0.5, res.z);
    }
    """

    cast_ray = """
    vec3 castRay(in vec3 ro, vec3 rd, int maxSteps)
    {
        vec3 res = vec3(1e10, -1, 1);

        float maxd = 1;
        float h = 1;
        float t = 0;
        vec2  m = vec2(-1);

        for(int i = 0; i < maxSteps; i++)
        {
            if(h < 0.01 || t > maxd)
                break;
            vec3 res = map(ro + rd * t);
            h = res.x;
            m = res.yz;
            t += h * 0.1;
        }

        if(t < maxd && t < res.x)
            res = vec3(t, m);

        return res;
    }
    """

    sdf_eval = """
    float sdfEval(vec3 p)
    {
        return map(p).x;
    }
    """

    central_diffs_normals = import_fury_shader(
        os.path.join("sdf", "central_diffs.frag")
    )

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
        fs_def_pi, fs_def_text_bracket, fs_unifs, fs_vs_vars, coeffs_norm,
        factorial, assoc_legendre_poly, norm_fact, spherical_harmonics,
        sdf_map, cast_ray, sdf_eval, central_diffs_normals, linear_to_srgb,
        srgb_to_linear, linear_rgb_to_srgb, srgb_to_linear_rgb, tonemap,
        blinn_phong_model
    ])
    # fmt: on

    shader_to_actor(odf_actor, "fragment", decl_code=fs_dec, debug=False)

    sdf_frag_impl = """
    vec3 pnt = vertexMCVSOutput.xyz;

    vec3 ro = camPosMCVSOutput;

    vec3 rd = normalize(pnt - ro);

    vec3 ld = normalize(ro - pnt);

    ro += pnt - ro;

    vec3 t = castRay(ro, rd, maxRaySteps);

    vec3 color = vec3(1);

    if(t.y >= 0)
    {
        vec3 pos = ro - centerMCVSOutput + t.x * rd;
        vec3 normal = centralDiffsNormals(pos, 0.0001);

        vec3 colorDir = srgbToLinearRgb(abs(normalize(pos)));

        float attenuation = clamp(dot(ld, normal), 0, 1);

        vec3 mate = blinnPhongIllumModel(
            attenuation, lightColor0, colorDir, specularPower,
            specularColor, colorDir);

        // Material
        // mix(color1, color2, t.y) to use sign to select between two colors
        //vec3 mate = 0.5 * mix(vec3(1, 0.6, 0.15), vec3(0.2, 0.4, 0.5), t.y);

        float occ = clamp(2 * t.z, 0, 1);

        float sss = pow(clamp(1 + dot(normal, rd), 0, 1), 1);
        //float sss = clamp(1 + dot(normal, rd), 0, 1);

        // Lighting
        vec3 lin  = 2.5 * occ * vec3(1) * (0.4 * normal.y + 0.6);
        lin += 1 * sss * vec3(1, 0.95, 0.7) * occ;

        color = mate * lin;
        //color = mate;
        //color = colorDir * lin;
        //color = colorDir;
    }
    else
    {
        discard;
    }

    vec3 outColor = linearRgbToSrgb(tonemap(color));
    fragOutput0 = vec4(outColor, opacity);
    """

    shader_to_actor(
        odf_actor, "fragment", impl_code=sdf_frag_impl, block="picking"
    )

    show_man.scene.add(odf_actor)

    sphere = get_sphere(name="repulsion724")

    sh_basis = "descoteaux07"
    # sh_basis = "tournier07"
    sh_order = 4

    sh = np.zeros((4, 1, 1, 15))
    sh[0, 0, 0, :] = coeffs[0, :]
    sh[1, 0, 0, :] = coeffs[1, :]
    sh[2, 0, 0, :] = coeffs[2, :]
    sh[3, 0, 0, :] = coeffs[3, :]

    fODFs = sh_to_sf(
        sh,
        sh_order_max=sh_order,
        basis_type=sh_basis,
        sphere=sphere,
        legacy=True,
    )
    odf_slicer_actor = actor.odf_slicer(fODFs, sphere=sphere, norm=True)

    show_man.scene.add(odf_slicer_actor)

    show_man.start()
