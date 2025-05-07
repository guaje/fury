"""
Fury's implementation of "Ray Tracing Spherical Harmonics Glyphs":
https://momentsingraphics.de/VMV2023.html
The fragment shader is based on: https://www.shadertoy.com/view/dlGSDV
(c) 2023, Christoph Peters
This work is licensed under a CC0 1.0 Universal License. To the extent
possible under law, Christoph Peters has waived all copyright and related or
neighboring rights to the following code. This work is published from
Germany. https://creativecommons.org/publicdomain/zero/1.0/

This script includes the original implementation as seen in shadertoy.com.
"""

import os

import numpy as np

from fury import actor, window
from fury.shaders import (
    attribute_to_actor,
    compose_shader,
    import_fury_shader,
    shader_to_actor,
)

if __name__ == "__main__":
    show_man = window.ShowManager(size=(1280, 720))
    show_man.scene.background((1, 1, 1))

    centers = np.array([[0, 0, 0]])
    scales = np.array([4])

    odf_actor = actor.box(centers=centers, scales=scales)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(odf_actor, big_centers, "center")

    big_scales = np.repeat(scales, 8, axis=0)
    attribute_to_actor(odf_actor, big_scales, "scale")

    vs_dec = """
    in vec3 center;
    in float scale;

    out vec4 vertexMCVSOutput;
    out vec3 centerMCVSOutput;
    out vec3 camPosMCVSOutput;
    out float scaleVSOutput;
    """

    vs_impl = """
    vertexMCVSOutput = vertexMC;
    centerMCVSOutput = center;
    scaleVSOutput = scale;
    camPosMCVSOutput = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
    """

    shader_to_actor(odf_actor, "vertex", decl_code=vs_dec, impl_code=vs_impl)

    # The index of the highest used band of the spherical harmonics basis. Must
    # be even, at least 2 and at most 12.
    def_sh_degree = "#define SH_DEGREE 4"

    # The number of spherical harmonics basis functions
    def_sh_count = "#define SH_COUNT (((SH_DEGREE + 1) * (SH_DEGREE + 2)) / 2)"

    # Degree of polynomials for which we have to find roots
    def_max_degree = "#define MAX_DEGREE (2 * SH_DEGREE + 2)"

    # If GL_EXT_control_flow_attributes is available, these defines should be
    # defined as [[unroll]] and [[loop]] to give reasonable hints to the
    # compiler. That avoids register spilling, which makes execution
    # considerably faster.
    def_gl_ext_control_flow_attributes = """
    #ifndef _unroll_
        #define _unroll_
    #endif
    #ifndef _loop_
        #define _loop_
    #endif
    """

    # When there are fewer intersections/roots than theoretically possible,
    # some array entries are set to this value
    def_no_intersection = "#define NO_INTERSECTION 3.4e38"

    # pi and its reciprocal
    def_pis = """
    #define M_PI 3.141592653589793238462643
    #define M_INV_PI 0.318309886183790671537767526745
    """

    fs_vs_vars = """
    in vec4 vertexMCVSOutput;
    in vec3 centerMCVSOutput;
    in vec3 camPosMCVSOutput;
    in float scaleVSOutput;
    """

    eval_sh_2 = import_fury_shader(
        os.path.join("ray_tracing", "odf", "descoteaux", "eval_sh_2.frag")
    )

    eval_sh_4 = import_fury_shader(
        os.path.join("ray_tracing", "odf", "descoteaux", "eval_sh_4.frag")
    )

    eval_sh_6 = import_fury_shader(
        os.path.join("ray_tracing", "odf", "descoteaux", "eval_sh_6.frag")
    )

    eval_sh_8 = import_fury_shader(
        os.path.join("ray_tracing", "odf", "descoteaux", "eval_sh_8.frag")
    )

    eval_sh_10 = import_fury_shader(
        os.path.join("ray_tracing", "odf", "descoteaux", "eval_sh_10.frag")
    )

    eval_sh_12 = import_fury_shader(
        os.path.join("ray_tracing", "odf", "descoteaux", "eval_sh_12.frag")
    )

    eval_sh_grad_2 = import_fury_shader(
        os.path.join("ray_tracing", "odf", "descoteaux", "eval_sh_grad_2.frag")
    )

    eval_sh_grad_4 = import_fury_shader(
        os.path.join("ray_tracing", "odf", "descoteaux", "eval_sh_grad_4.frag")
    )

    eval_sh_grad_6 = import_fury_shader(
        os.path.join("ray_tracing", "odf", "descoteaux", "eval_sh_grad_6.frag")
    )

    eval_sh_grad_8 = import_fury_shader(
        os.path.join("ray_tracing", "odf", "descoteaux", "eval_sh_grad_8.frag")
    )

    eval_sh_grad_10 = import_fury_shader(
        os.path.join(
            "ray_tracing", "odf", "descoteaux", "eval_sh_grad_10.frag"
        )
    )

    eval_sh_grad_12 = import_fury_shader(
        os.path.join(
            "ray_tracing", "odf", "descoteaux", "eval_sh_grad_12.frag"
        )
    )

    # Searches a single root of a polynomial within a given interval.
    #   param out_root The location of the found root.
    #   param out_end_value The value of the given polynomial at end.
    #   param poly Coefficients of the polynomial for which a root should be
    #       found.
    #       Coefficient poly[i] is multiplied by x^i.
    #   param begin The beginning of an interval where the polynomial is
    #       monotonic.
    #   param end The end of said interval.
    #   param begin_value The value of the given polynomial at begin.
    #   param error_tolerance The error tolerance for the returned root
    #       location.
    #       Typically the error will be much lower but in theory it can be
    #       bigger.
    #
    #   return true if a root was found, false if no root exists.
    newton_bisection = import_fury_shader(
        os.path.join("utils", "newton_bisection.frag")
    )

    # Finds all roots of the given polynomial in the interval [begin, end] and
    # writes them to out_roots. Some entries will be NO_INTERSECTION but other
    # than that the array is sorted. The last entry is always NO_INTERSECTION.
    find_roots = import_fury_shader(os.path.join("utils", "find_roots.frag"))

    # Evaluates the spherical harmonics basis in bands 0, 2, ..., SH_DEGREE.
    # Conventions are as in the following paper.
    # M. Descoteaux, E. Angelino, S. Fitzgibbons, and R. Deriche. Regularized,
    # fast, and robust analytical q-ball imaging. Magnetic Resonance in
    # Medicine, 58(3), 2007. https://doi.org/10.1002/mrm.21277
    #   param out_shs Values of SH basis functions in bands 0, 2, ...,
    #       SH_DEGREE in this order.
    #   param point The point on the unit sphere where the basis should be
    #       evaluated.
    eval_sh = import_fury_shader(
        os.path.join("ray_tracing", "odf", "eval_sh.frag")
    )

    # Evaluates the gradient of each basis function given by eval_sh() and the
    # basis itself
    eval_sh_grad = import_fury_shader(
        os.path.join("ray_tracing", "odf", "eval_sh_grad.frag")
    )

    # Outputs a matrix that turns equidistant samples on the unit circle of a
    # homogeneous polynomial into coefficients of that polynomial.
    get_inv_vandermonde = import_fury_shader(
        os.path.join("ray_tracing", "odf", "get_inv_vandermonde.frag")
    )

    # Determines all intersections between a ray and a spherical harmonics
    # glyph.
    #   param out_ray_params The ray parameters at intersection points. The
    #       points themselves are at ray_origin + out_ray_params[i] * ray_dir.
    #       Some entries may be NO_INTERSECTION but other than that the array
    #       is sorted.
    #   param sh_coeffs SH_COUNT spherical harmonic coefficients defining the
    #       glyph. Their exact meaning is defined by eval_sh().
    #   param ray_origin The origin of the ray, relative to the glyph center.
    #   param ray_dir The normalized direction vector of the ray.
    ray_sh_glyph_intersections = import_fury_shader(
        os.path.join("ray_tracing", "odf", "ray_sh_glyph_intersections.frag")
    )

    # Provides a normalized normal vector for a spherical harmonics glyph.
    #   param sh_coeffs SH_COUNT spherical harmonic coefficients defining the
    #       glyph. Their exact meaning is defined by eval_sh().
    #   param point A point on the surface of the glyph, relative to its
    #       center.
    #
    #   return A normalized surface normal pointing away from the origin.
    get_sh_glyph_normal = import_fury_shader(
        os.path.join("ray_tracing", "odf", "get_sh_glyph_normal.frag")
    )

    # This is the glTF BRDF for dielectric materials, exactly as described
    # here:
    # https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
    #   param incoming The normalized incoming light direction.
    #   param outgoing The normalized outgoing light direction.
    #   param normal The normalized shading normal.
    #   param roughness An artist friendly roughness value between 0 and 1.
    #   param base_color The albedo used for the Lambertian diffuse component.
    #
    #   return The BRDF for the given directions.
    gltf_dielectric_brdf = """
    vec3 gltf_dielectric_brdf(vec3 incoming, vec3 outgoing, vec3 normal, float roughness, vec3 base_color)
    {
        float ni = dot(normal, incoming);
        float no = dot(normal, outgoing);
        // Early out if incoming or outgoing direction are below the horizon
        if (ni <= 0.0 || no <= 0.0)
            return vec3(0.0);
        // Save some work by not actually computing the half-vector. If the half-
        // vector were h, ih = dot(incoming, h) and
        // sqrt(nh_ih_2 / ih_2) = dot(normal, h).
        float ih_2 = dot(incoming, outgoing) * 0.5 + 0.5;
        float sum = ni + no;
        float nh_ih_2 = 0.25 * sum * sum;
        float ih = sqrt(ih_2);

        // Evaluate the GGX normal distribution function
        float roughness_2 = roughness * roughness;
        float roughness_4  = roughness_2 * roughness_2;
        float roughness_flip = 1.0 - roughness_4;
        float denominator = ih_2 - nh_ih_2 * roughness_flip;
        float ggx = (roughness_4 * M_INV_PI * ih_2) / (denominator * denominator);
        // Evaluate the "visibility" (i.e. masking-shadowing times geometry terms)
        float vi = ni + sqrt(roughness_4 + roughness_flip * ni * ni);
        float vo = no + sqrt(roughness_4 + roughness_flip * no * no);
        float v = 1.0 / (vi * vo);
        // That completes the specular BRDF
        float specular = v * ggx;

        // The diffuse BRDF is Lambertian
        vec3 diffuse = M_INV_PI * base_color;

        // Evaluate the Fresnel term using the Fresnel-Schlick approximation
        const float ior = 1.5;
        const float f0 = ((1.0 - ior) / (1.0 + ior)) * ((1.0 - ior) / (1.0 + ior));
        float ih_flip = 1.0 - ih;
        float ih_flip_2 = ih_flip * ih_flip;
        float fresnel = f0 + (1.0 - f0) * ih_flip * ih_flip_2 * ih_flip_2;

        // Mix the two components
        return mix(diffuse, vec3(specular), fresnel);
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

    # fmt: off
    fs_dec = compose_shader([
        def_sh_degree, def_sh_count, def_max_degree,
        def_gl_ext_control_flow_attributes, def_no_intersection, def_pis,
        fs_vs_vars, eval_sh_2, eval_sh_4, eval_sh_6, eval_sh_8, eval_sh_10,
        eval_sh_12, eval_sh_grad_2, eval_sh_grad_4, eval_sh_grad_6,
        eval_sh_grad_8, eval_sh_grad_10, eval_sh_grad_12, newton_bisection,
        find_roots, eval_sh, eval_sh_grad, get_inv_vandermonde,
        ray_sh_glyph_intersections, get_sh_glyph_normal, gltf_dielectric_brdf,
        linear_to_srgb, srgb_to_linear, linear_rgb_to_srgb, srgb_to_linear_rgb,
        tonemap
    ])
    # fmt: on

    shader_to_actor(odf_actor, "fragment", decl_code=fs_dec)

    point_from_vs = "vec3 pnt = vertexMCVSOutput.xyz;"

    # Ray origin is the camera position in world space
    ray_origin = """
    vec3 ro = camPosMCVSOutput;
    //vec3 camera_pos = vec3(0.0, -5.0, 0.0);
    //vec3 camera_pos = ro;
    """

    # TODO: Check aspect for automatic scaling
    # Ray direction is the normalized difference between the fragment and the
    # camera position/ray origin
    ray_direction = """
    vec3 rd = normalize(pnt - ro);
    //float aspect = float(iResolution.x) / float(iResolution.y);
    //float aspect = 1;
    //float zoom = .8;
    //vec3 right = (aspect / zoom) * vec3(3.0, 0.0, 0.0);
    //vec3 up = (1.0 / zoom) * vec3(0.0, 0.0, 3.0);
    //vec3 bottom_left = -0.5 * (right + up);
    //vec2 frag_coord = gl_FragCoord.xy;
    //vec2 uv = frag_coord / vec2(iResolution.xy);
    //vec3 ray_dir = normalize(bottom_left + uv.x * right + uv.y * up - camera_pos);
    //vec3 ray_dir = rd;
    """

    # Light direction in a retroreflective model is the normalized difference
    # between the camera position/ray origin and the fragment
    light_direction = "vec3 ld = normalize(ro - pnt);"

    # Define SH coefficients (measured up to band 8, noise beyond that)
    sh_coeffs = """
    float shCoeffs[SH_COUNT];
    shCoeffs[0] = -0.2739740312099; shCoeffs[1] = 0.2526670396328;
    shCoeffs[2] = 1.8922271728516; shCoeffs[3] = 0.2878578901291;
    shCoeffs[4] = -0.5339795947075; shCoeffs[5] = -0.2620058953762;
    #if SH_DEGREE >= 4
        shCoeffs[6] = 0.1580424904823; shCoeffs[7] = 0.0329004973173;
        shCoeffs[8] = -0.1322413831949; shCoeffs[9] = -0.1332057565451;
        shCoeffs[10] = 1.0894461870193; shCoeffs[11] = -0.6319401264191;
        shCoeffs[12] = -0.0416776277125; shCoeffs[13] = -1.0772529840469;
        shCoeffs[14] = 0.1423762738705;
    #endif
    #if SH_DEGREE >= 6
        shCoeffs[15] = 0.7941166162491; shCoeffs[16] = 0.7490307092667;
        shCoeffs[17] = -0.3428381681442; shCoeffs[18] = 0.1024847552180;
        shCoeffs[19] = -0.0219132602215; shCoeffs[20] = 0.0499043911695;
        shCoeffs[21] = 0.2162453681231; shCoeffs[22] = 0.0921059995890;
        shCoeffs[23] = -0.2611238956451; shCoeffs[24] = 0.2549301385880;
        shCoeffs[25] = -0.4534865319729; shCoeffs[26] = 0.1922748684883;
        shCoeffs[27] = -0.6200597286224;
    #endif
    #if SH_DEGREE >= 8
        shCoeffs[28] = -0.0532187558711; shCoeffs[29] = -0.3569841980934;
        shCoeffs[30] = 0.0293972902000; shCoeffs[31] = -0.1977960765362;
        shCoeffs[32] = -0.1058669015765; shCoeffs[33] = 0.2372217923403;
        shCoeffs[34] = -0.1856198310852; shCoeffs[35] = -0.3373193442822;
        shCoeffs[36] = -0.0750469490886; shCoeffs[37] = 0.2146576642990;
        shCoeffs[38] = -0.0490148440003; shCoeffs[39] = 0.1288588196039;
        shCoeffs[40] = 0.3173974752426; shCoeffs[41] = 0.1990085393190;
        shCoeffs[42] = -0.1736343950033; shCoeffs[43] = -0.0482443645597;
        shCoeffs[44] = 0.1749017387629;
    #endif
    #if SH_DEGREE >= 10
        shCoeffs[45] = -0.0151847425660; shCoeffs[46] = 0.0418366046081;
        shCoeffs[47] = 0.0863263587216; shCoeffs[48] = -0.0649211244490;
        shCoeffs[49] = 0.0126096132283; shCoeffs[50] = 0.0545089217982;
        shCoeffs[51] = -0.0275142164626; shCoeffs[52] = 0.0399986574832;
        shCoeffs[53] = -0.0468244261610; shCoeffs[54] = -0.1292105653111;
        shCoeffs[55] = -0.0786858322658; shCoeffs[56] = -0.0663828464882;
        shCoeffs[57] = 0.0382439706831; shCoeffs[58] = -0.0041550330365;
        shCoeffs[59] = -0.0502800566338; shCoeffs[60] = -0.0732471630735;
        shCoeffs[61] = 0.0181751900972; shCoeffs[62] = -0.0090119333757;
        shCoeffs[63] = -0.0604443282359; shCoeffs[64] = -0.1469985252752;
        shCoeffs[65] = -0.0534046899715;
    #endif
    #if SH_DEGREE >= 12
        shCoeffs[66] = -0.0896672753415; shCoeffs[67] = -0.0130841364808;
        shCoeffs[68] = -0.0112942893801; shCoeffs[69] = 0.0272257498541;
        shCoeffs[70] = 0.0626717616331; shCoeffs[71] = -0.0222197983050;
        shCoeffs[72] = -0.0018541504308; shCoeffs[73] = -0.1653251944056;
        shCoeffs[74] = 0.0409697402846; shCoeffs[75] = 0.0749921454327;
        shCoeffs[76] = -0.0282830872616; shCoeffs[77] = 0.0006909458525;
        shCoeffs[78] = 0.0625599842287; shCoeffs[79] = 0.0812529816082;
        shCoeffs[80] = 0.0914693020772; shCoeffs[81] = -0.1197222726745;
        shCoeffs[82] = 0.0376277453183; shCoeffs[83] = -0.0832617004142;
        shCoeffs[84] = -0.0482175038043; shCoeffs[85] = -0.0839003635737;
        shCoeffs[86] = -0.0349423908400; shCoeffs[87] = 0.1204519568256;
        shCoeffs[88] = 0.0783745984003; shCoeffs[89] = 0.0297401205976;
        shCoeffs[90] = -0.0505947662525;
    #endif
    """

    # Perform the intersection test
    intersection_test = """
    float rayParams[MAX_DEGREE];
    rayGlyphIntersections(
        rayParams, shCoeffs, ro - centerMCVSOutput, rd, int(shDegree),
        int(numCoeffsVSOutput), int(maxPolyDegreeVSOutput), M_PI,
        NO_INTERSECTION
    );
    """

    # Identify the first intersection
    first_intersection = """
    float first_ray_param = NO_INTERSECTION;
    _unroll_
    for (int i = 0; i != MAX_DEGREE; ++i) {
        if (ray_params[i] != NO_INTERSECTION && ray_params[i] > 0.0) {
            first_ray_param = ray_params[i];
            break;
        }
    }
    """

    # Evaluate shading for a directional light
    directional_light = """
    vec3 color = vec3(1.0);
    if (first_ray_param != NO_INTERSECTION) {
        //vec3 intersection = camera_pos + first_ray_param * ray_dir;
        vec3 intersection = ro + first_ray_param * rd;
        vec3 normal = getShGlyphNormal(sh_coeffs, intersection);
        vec3 base_color = srgb_to_linear_rgb(abs(normalize(intersection)));
        const vec3 incoming = normalize(vec3(1.23, -4.56, 7.89));
        float ambient = 0.04;
        float exposure = 4.0;
        //vec3 outgoing = -ray_dir;
        vec3 outgoing = -rd;
        vec3 brdf = gltf_dielectric_brdf(incoming, outgoing, normal, 0.45, base_color);
        color = exposure * (brdf * max(0.0, dot(incoming, normal)) + base_color * ambient);
    }
    """

    frag_output = """
    //vec4 out_color = vec4(linear_rgb_to_srgb(tonemap(color)), 1.0);
    vec3 out_color = linear_rgb_to_srgb(tonemap(color));
    fragOutput0 = vec4(out_color, opacity);
    """

    # fmt: off
    fs_impl = compose_shader([
        point_from_vs, ray_origin, ray_direction, light_direction, sh_coeffs,
        intersection_test, first_intersection, directional_light, frag_output
    ])
    # fmt: on

    shader_to_actor(odf_actor, "fragment", impl_code=fs_impl, block="light")

    show_man.scene.add(odf_actor)

    show_man.start()
