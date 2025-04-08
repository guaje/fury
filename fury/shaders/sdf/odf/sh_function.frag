float calculateSH(int l, int m, in vec3 s)
{
    vec3 normS = normalize(s);
    float thetaX = normS.z;
    float phi = atan(normS.y, normS.x);
    float result = calculateKFactor(l, abs(m)) * calcAssocLegendrePoly(l, abs(m), thetaX);
    if(m != 0)
        result *= sqrt(2);
    if(m > 0)
        result *= sin(m * phi);
    if(m < 0)
        result *= cos(-m * phi);

    return result;
}
