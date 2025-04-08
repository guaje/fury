float calcAssocLegendrePoly(int l, int m, float x)
{
    float legendrePmm = 1.0; // P(m, m, x)
    float sqrtTerm = sqrt((1.0 - x) * (1.0 + x)); // sqrt(1 - x^2) term
    float factor = 1.0;

    // Calculate P(m, m, x)
    for (int i = 1; i <= m; ++i) {
        legendrePmm *= -factor * sqrtTerm;
        factor += 2.0;
    }

    if (l == m) {
        return legendrePmm;
    }

    float legendrePmmPlus1 = x * (2.0 * m + 1.0) * legendrePmm; // P(m+1, m, x)
    if (l == m + 1) {
        return legendrePmmPlus1;
    }

    float legendrePlm = 0.0; // P(l, m, x)
    float prevPmm = legendrePmm; // P(l-2, m, x) in iteration
    float prevPmmPlus1 = legendrePmmPlus1; // P(l-1, m, x) in iteration

    // Calculate P(l, m, x) using recurrence relation for l > m + 1
    for (int lIndex = m + 2; lIndex <= l; ++lIndex) {
        legendrePlm = ((2.0 * lIndex - 1.0) * x * prevPmmPlus1 - (lIndex + m - 1.0) * prevPmm) / (lIndex - m);
        prevPmm = prevPmmPlus1;
        prevPmmPlus1 = legendrePlm;
    }

    return legendrePlm;
}
