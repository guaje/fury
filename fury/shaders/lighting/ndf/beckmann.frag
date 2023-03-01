float beckmann(float alpha, float dotHN)
{
    float a2 = square(alpha);
    float dotHN2 = square(dotHN);
    float d = PI * a2 * square(dotHN2);
    float e = exp((dotHN2 - 1) / (a2 * dotHN2));
    return (1 / d) * e;
}
