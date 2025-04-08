vec3 centralDiffsNormals(in vec3 p, float eps)
{
    vec2 h = vec2(eps, 0);
    return normalize(
        vec3(
            sdfEval(p + h.xyy) - sdfEval(p - h.xyy),
            sdfEval(p + h.yxy) - sdfEval(p - h.yxy),
            sdfEval(p + h.yyx) - sdfEval(p - h.yyx)
        )
    );
}
