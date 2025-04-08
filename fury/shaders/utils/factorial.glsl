int factorial(int v)
{
    int t = 1;
    for(int i = 2; i <= v; i++)
    {
        t *= i;
    }
    return t;
}
