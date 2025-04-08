vec3 castRay(in vec3 ro, vec3 rd)
{
    vec3 res = vec3(1e10, -1, 1);

    float maxd = 1;
    float h = 1;
    float t = 0;
    vec2  m = vec2(-1);

    for(int i = 0; i < 2000; i++)
    {
        if(h < .01 || t > maxd)
            break;
        vec3 res = map(ro + rd * t);
        h = res.x;
        m = res.yz;
        t += h * .1;
    }

    if(t < maxd && t < res.x)
        res = vec3(t, m);

    return res;
}