#include "SolveNLE.h"


double foo(const double x)
{
    //18. x^2 - 1 - ln(x) = 0
    return x * x - 1.0 - log(x);
}

double bar(const double x)
{
    return 2 * x - 1 / x;
}

//double barbar(const double x)
//{
//    return 2 + 1 / (x * x);
//}

void solveEq()
{
    std::cout << "x^2 - 1 - ln(x) = 0\n";
    double l = 0.0;
    double r = 2.0;
    std::cout << l << "..." << r << "\n";
    int n = 50;
    double d = (r - l) / n;
    double l1 = l, r1 = r;
    bool f = false;
    for (int i = 0; i < n; ++i)
    {
        if (foo(l + i * d) * foo(l + (i + 1) * d) <= 0)
        {
            l1 = l + i * d;
            r1 = l + (i + 1) * d;
            f = true;
            break;
        }
    }
    if (f)
    {
        std::cout << "Root is somewhere here: " << l1 << ' ' << r1 << '\n';
        double x = l1;
        double next = 0.0;
        do
        {
            next = x;
            x -= foo(x) / bar(x);
        } while (abs(x - next) > 10e-6);
        std::cout << "x = " << x << "; check: " << foo(x) << '\n';
    } else
    {
        std::cout << "there is no root\n";
    }
    
}


