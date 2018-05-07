#include <array>
#include <iostream>
#include <cmath>
#include <chrono>

#include "opencv2/core.hpp"

#include "SolveNLE.h"
#include "SolveSNLE.h"
#include "PLU.h"

void second()
{
    double xtmp[] = { 0.5, 0.5, 1.5, -1.0, -0.2, 1.5, 0.5, -0.5, 1.5, -1.5 };
    cv::Mat x(10, 1, cv::DataType<double>::type, xtmp);

    cv::Mat ans = NewtonSolve(x, 10e-6);
    std::cout << ans << '\n' << Equations(ans) << "\n\n";

    ans = NewtonModSolve(x, 10e-6);
    std::cout << ans << '\n' << Equations(ans) << "\n\n";

    ans = NewtonMixSolve(x, 8, 10e-6);
    std::cout << ans << '\n' << Equations(ans) << "\n\n";

    ans = NewtonHybridSolve(x, 5, 10e-6); 
    std::cout << ans << '\n' << Equations(ans) << "\n\n";
}

void third()
{
    double xtmp[] = { 0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5 };
    cv::Mat x(10, 1, cv::DataType<double>::type, xtmp);

    NewtonSolve(x, 10e-6);
    NewtonModSolve(x, 10e-6);
    NewtonHybridSolve(x, 3, 10e-6);
    NewtonMixSolve(x, 3, 10e-6);

    std::chrono::time_point<std::chrono::system_clock> start;
    int elapsed_seconds = 0;
    cv::Mat ans;
    int N = 100;
    start = std::chrono::system_clock::now();
    for (int i = 0; i < N; ++i)
    {      
        ans = NewtonSolve(x);      
    }
    elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::system_clock::now() - start).count();
    std::cout << "Classic: " << elapsed_seconds / N << " mcs\n";

    start = std::chrono::system_clock::now();
    for (int i = 0; i < N; ++i)
    {
        ans = NewtonModSolve(x);
    }
    elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::system_clock::now() - start).count();
    std::cout << "Mod: " << elapsed_seconds / N << " mcs\n";

    start = std::chrono::system_clock::now();
    for (int i = 0; i < N; ++i)
    {
        ans = NewtonMixSolve(x, 3);
    }
    elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::system_clock::now() - start).count();
    std::cout << "Mix: " << elapsed_seconds / N << " mcs\n";

    start = std::chrono::system_clock::now();
    for (int i = 0; i < N; ++i)
    {
        ans = NewtonHybridSolve(x, 3);
    }
    elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
        (std::chrono::system_clock::now() - start).count();
    std::cout << "Hybrid: " << elapsed_seconds / N << " mcs\n";
}
int main()
{
    //solveEq();
    //second();
    third();
    std::cin.get();
    return 0;
}
