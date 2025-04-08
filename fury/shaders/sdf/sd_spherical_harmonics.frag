vec3 map( in vec3 p )
{
    p = p - centerMCVSOutput;
    vec3 p00 = p;

    float r, d; vec3 n, s, res;

    #define SHAPE (vec3(d-abs(r), sign(r),d))
    d=length(p00);
    n=p00 / d;
    float i = 1 / (numCoeffs * 2);
    float shCoeffs[15];
    float maxCoeff = 0.0;
    for(int j=0; j < numCoeffs; j++){
        shCoeffs[j] = rescale(
            texture(
                texture0,
                vec2(i + j / numCoeffs, tcoordVCVSOutput.y)).x, 0, 1,
                minmaxVSOutput.x, minmaxVSOutput.y
        );
    }
    r = shCoeffs[0] * SH(0, 0, n);
    r += shCoeffs[1]* SH(2, -2, n);
    r += shCoeffs[2]* SH(2, -1, n);
    r += shCoeffs[3] * SH(2, 0, n);
    r += shCoeffs[4] * SH(2, 1, n);
    r += shCoeffs[5] * SH(2, 2, n);
    r += shCoeffs[6] * SH(4, -4, n);
    r += shCoeffs[7] * SH(4, -3, n);
    r += shCoeffs[8]* SH(4, -2, n);
    r += shCoeffs[9]* SH(4, -1, n);
    r += shCoeffs[10]* SH(4, 0, n);
    r += shCoeffs[11]* SH(4, 1, n);
    r += shCoeffs[12]* SH(4, 2, n);
    r += shCoeffs[13]* SH(4, 3, n);
    r += shCoeffs[14]* SH(4, 4, n);

    r /= abs(sfmaxVSOutput);
    r *= scaleVSOutput * .9;

    s = SHAPE;
    res=s;
    return vec3(res.x, .5 + .5 * res.y, res.z);
}