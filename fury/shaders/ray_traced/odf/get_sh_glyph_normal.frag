vec3 getShGlyphNormal(float shCoeffs[SH_COUNT], vec3 point, int numCoeffs)
{
    float shs[SH_COUNT];
    vec3 grads[SH_COUNT];
    float lengthInv = inversesqrt(dot(point, point));
    vec3 normalized = point * lengthInv;
    eval_sh_grad(shs, grads, normalized);
    float value = 0.0;
    vec3 grad = vec3(0.0);
    _unroll_
    for (int i = 0; i != numCoeffs; ++i) {
        value += shCoeffs[i] * shs[i];
        grad += shCoeffs[i] * grads[i];
    }
    return normalize(point - (value * lengthInv) * (grad - dot(grad, normalized) * normalized));
}
