import os
import warnings

from fury.lib import Actor
from fury.primitive import prim_square, repeat_primitive
from fury.shaders import (
    attribute_to_actor,
    compose_shader,
    import_fury_shader,
    replace_shader_in_actor,
    shader_to_actor,
)
from fury.utils import get_actor_from_primitive


class BillboardActor(Actor):
    def __init__(
        self, centers, colors, scales, vs_dec=None, vs_impl=None, gs_prog=None,
        fs_dec=None, fs_impl=None, bb_type='spherical'):
        verts, faces = prim_square()
        
        res = repeat_primitive(
            verts, faces, centers, colors=colors, scales=scales)
        
        big_verts, big_faces, big_colors, big_centers = res
        
        prim_count = len(centers)
        
        self.__mapper = get_actor_from_primitive(
            big_verts, big_faces, colors=big_colors,
            prim_count=prim_count).GetMapper()
        self.__mapper.SetVBOShiftScaleMethod(False)
        self.__mapper.GetProperty().BackfaceCullingOff()
        self.__mapper.Update()
        
        self.SetMapper(self.__mapper)
        
        attribute_to_actor(self, big_centers, 'center')
        
        bb_norm = import_fury_shader(
            os.path.join('utils', 'billboard_normalization.glsl'))
        
        if bb_type.lower() == 'cylindrical_x':
            bb_type_sd = import_fury_shader(
                os.path.join('billboard', 'cylindrical_x.glsl'))
            v_pos_mc = \
                """
                vec3 vertexPositionMC = cylindricalXVertexPos(
                    center, MCVCMatrix, normalizedVertexMCVSOutput, shape);
                """
        elif bb_type.lower() == 'cylindrical_y':
            bb_type_sd = import_fury_shader(
                os.path.join('billboard', 'cylindrical_y.glsl'))
            v_pos_mc = \
                """
                vec3 vertexPositionMC = cylindricalYVertexPos(
                    center, MCVCMatrix, normalizedVertexMCVSOutput, shape);
                """
        elif bb_type.lower() == 'spherical':
            bb_type_sd = import_fury_shader(
                os.path.join('billboard', 'spherical.glsl'))
            v_pos_mc = \
                """
                vec3 vertexPositionMC = sphericalVertexPos(
                    center, MCVCMatrix, normalizedVertexMCVSOutput, shape);
                """
        else:
            bb_type_sd = import_fury_shader(
                os.path.join('billboard', 'spherical.glsl'))
            v_pos_mc = \
                """
                vec3 vertexPositionMC = sphericalVertexPos(
                    center, MCVCMatrix, normalizedVertexMCVSOutput, shape);
                """
            warnings.warn(
                'Invalid option. The billboard will be generated with the '
                'default spherical option. ', UserWarning)
        
        gl_position = \
            """
            gl_Position = MCDCMatrix * vec4(vertexPositionMC, 1.);
            """
        
        billboard_dec_vert = \
            """
            in vec3 center;
            
            out vec3 centerVertexMCVSOutput;
            out vec3 normalizedVertexMCVSOutput;
            """
        
        billboard_impl_vert = \
            """
            centerVertexMCVSOutput = center;
            normalizedVertexMCVSOutput = bbNorm(vertexMC.xyz, center);
            float scalingFactor = 1. / abs(normalizedVertexMCVSOutput.x);
            float size = abs((vertexMC.xyz - center).x) * 2;
            vec2 shape = vec2(size, size); // Fixes the scaling issue
            """
        
        billboard_dec_frag = \
            """
            in vec3 centerVertexMCVSOutput;
            in vec3 normalizedVertexMCVSOutput;
            """
        
        billboard_impl_frag = \
            """
            vec3 color = vertexColorVSOutput.rgb;
            vec3 point = normalizedVertexMCVSOutput;
            fragOutput0 = vec4(color, 1.);
            """
        
        billboard_vert_impl = compose_shader(
            [billboard_impl_vert, v_pos_mc, gl_position])
        
        vs_dec_code = compose_shader(
            [billboard_dec_vert, compose_shader(vs_dec), bb_norm, bb_type_sd])
        
        vs_impl_code = compose_shader(
            [compose_shader(vs_impl), billboard_vert_impl])
        
        gs_code = compose_shader(gs_prog)
        
        fs_dec_code = compose_shader(
            [billboard_dec_frag, compose_shader(fs_dec)])
        
        fs_impl_code = compose_shader(
            [billboard_impl_frag, compose_shader(fs_impl)])
        
        shader_to_actor(
            self, 'vertex', impl_code=vs_impl_code, decl_code=vs_dec_code)
        
        replace_shader_in_actor(self, 'geometry', gs_code)
        
        shader_to_actor(self, 'fragment', decl_code=fs_dec_code)
        
        shader_to_actor(
            self, 'fragment', impl_code=fs_impl_code, block='light')
