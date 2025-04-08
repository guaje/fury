float calculateKFactor(int l, int m)
{
    float num = (2.0 * l + 1.0) * factorial(l - m);
    float den = 4.0 * PI * factorial(l + m);
    return sqrt(num / den);
}
