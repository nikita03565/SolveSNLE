#include "SolveSNLE.h"
#include <iostream>
#include "PLU.h"
#include <chrono>

cv::Mat Jacobi(cv::Mat x)
{
    cv::Mat tmp(10, 10, cv::DataType<double>::type);
    tmp.at<double>(0, 0) = -sin(x.at<double>(0, 0) * x.at<double>(1, 0)) * x.at<double>(1, 0);
    tmp.at<double>(0, 1) = -sin(x.at<double>(0, 0) * x.at<double>(1, 0)) * x.at<double>(0, 0);
    tmp.at<double>(0, 2) = 3. * exp(-(3 * x.at<double>(2, 0)));
    tmp.at<double>(0, 3) = x.at<double>(4, 0) * x.at<double>(4, 0);
    tmp.at<double>(0, 4) = 2 * x.at<double>(3, 0) * x.at<double>(4, 0);
    tmp.at<double>(0, 5) = -1;
    tmp.at<double>(0, 6) = 0;
    tmp.at<double>(0, 7) = -2. * cosh((2 * x.at<double>(7, 0))) * x.at<double>(8, 0);
    tmp.at<double>(0, 8) = -sinh((2 * x.at<double>(7, 0)));
    tmp.at<double>(0, 9) = 2;
    tmp.at<double>(1, 0) = cos(x.at<double>(0, 0) * x.at<double>(1, 0)) * x.at<double>(1, 0);
    tmp.at<double>(1, 1) = cos(x.at<double>(0, 0) * x.at<double>(1, 0)) * x.at<double>(0, 0);
    tmp.at<double>(1, 2) = x.at<double>(8, 0) * x.at<double>(6, 0);
    tmp.at<double>(1, 3) = 0;
    tmp.at<double>(1, 4) = 6 * x.at<double>(4, 0);
    tmp.at<double>(1, 5) = -exp(-x.at<double>(9, 0) + x.at<double>(5, 0)) - x.at<double>(7, 0) - 0.1e1;
    tmp.at<double>(1, 6) = x.at<double>(2, 0) * x.at<double>(8, 0);
    tmp.at<double>(1, 7) = -x.at<double>(5, 0);
    tmp.at<double>(1, 8) = x.at<double>(2, 0) * x.at<double>(6, 0);
    tmp.at<double>(1, 9) = exp(-x.at<double>(9, 0) + x.at<double>(5, 0));
    tmp.at<double>(2, 0) = 1;
    tmp.at<double>(2, 1) = -1;
    tmp.at<double>(2, 2) = 1;
    tmp.at<double>(2, 3) = -1;
    tmp.at<double>(2, 4) = 1;
    tmp.at<double>(2, 5) = -1;
    tmp.at<double>(2, 6) = 1;
    tmp.at<double>(2, 7) = -1;
    tmp.at<double>(2, 8) = 1;
    tmp.at<double>(2, 9) = -1;
    tmp.at<double>(3, 0) = -x.at<double>(4, 0) * pow(x.at<double>(2, 0) + x.at<double>(0, 0), -2.);
    tmp.at<double>(3, 1) = -2. * cos(x.at<double>(1, 0) * x.at<double>(1, 0)) * x.at<double>(1, 0);
    tmp.at<double>(3, 2) = -x.at<double>(4, 0) * pow(x.at<double>(2, 0) + x.at<double>(0, 0), -2.);
    tmp.at<double>(3, 3) = -2. * sin(-x.at<double>(8, 0) + x.at<double>(3, 0));
    tmp.at<double>(3, 4) = 1. / (x.at<double>(2, 0) + x.at<double>(0, 0));
    tmp.at<double>(3, 5) = 0;
    tmp.at<double>(3, 6) = -2. * cos(x.at<double>(6, 0) * x.at<double>(9, 0)) * sin(x.at<double>(6, 0) * x.at<double>(9, 0)) * x.at<double>(9, 0);
    tmp.at<double>(3, 7) = -1;
    tmp.at<double>(3, 8) = 2. * sin(-x.at<double>(8, 0) + x.at<double>(3, 0));
    tmp.at<double>(3, 9) = -2. * cos(x.at<double>(6, 0) * x.at<double>(9, 0)) * sin(x.at<double>(6, 0) * x.at<double>(9, 0)) * x.at<double>(6, 0);
    tmp.at<double>(4, 0) = 2 * x.at<double>(7, 0);
    tmp.at<double>(4, 1) = -2. * sin(x.at<double>(1, 0));
    tmp.at<double>(4, 2) = 2 * x.at<double>(7, 0);
    tmp.at<double>(4, 3) = pow(-x.at<double>(8, 0) + x.at<double>(3, 0), -2.);
    tmp.at<double>(4, 4) = cos(x.at<double>(4, 0));
    tmp.at<double>(4, 5) = x.at<double>(6, 0) * exp(-x.at<double>(6, 0) * (-x.at<double>(9, 0) + x.at<double>(5, 0)));
    tmp.at<double>(4, 6) = -(x.at<double>(9, 0) - x.at<double>(5, 0)) * exp(-x.at<double>(6, 0) * (-x.at<double>(9, 0) + x.at<double>(5, 0)));
    tmp.at<double>(4, 7) = (2 * x.at<double>(2, 0)) + 2. * x.at<double>(0, 0);
    tmp.at<double>(4, 8) = -pow(-x.at<double>(8, 0) + x.at<double>(3, 0), -2.);
    tmp.at<double>(4, 9) = -x.at<double>(6, 0) * exp(-x.at<double>(6, 0) * (-x.at<double>(9, 0) + x.at<double>(5, 0)));
    tmp.at<double>(5, 0) = exp(x.at<double>(0, 0) - x.at<double>(3, 0) - x.at<double>(8, 0));
    tmp.at<double>(5, 1) = -3. / 2. * sin(3. * x.at<double>(9, 0) * x.at<double>(1, 0)) * x.at<double>(9, 0);
    tmp.at<double>(5, 2) = -x.at<double>(5, 0);
    tmp.at<double>(5, 3) = -exp(x.at<double>(0, 0) - x.at<double>(3, 0) - x.at<double>(8, 0));
    tmp.at<double>(5, 4) = 2 * x.at<double>(4, 0) / x.at<double>(7, 0);
    tmp.at<double>(5, 5) = -x.at<double>(2, 0);
    tmp.at<double>(5, 6) = 0;
    tmp.at<double>(5, 7) = -x.at<double>(4, 0) * x.at<double>(4, 0) * pow(x.at<double>(7, 0), (-2));
    tmp.at<double>(5, 8) = -exp(x.at<double>(0, 0) - x.at<double>(3, 0) - x.at<double>(8, 0));
    tmp.at<double>(5, 9) = -3. / 2. * sin(3. * x.at<double>(9, 0) * x.at<double>(1, 0)) * x.at<double>(1, 0);
    tmp.at<double>(6, 0) = cos(x.at<double>(3, 0));
    tmp.at<double>(6, 1) = 3. * x.at<double>(1, 0) * x.at<double>(1, 0) * x.at<double>(6, 0);
    tmp.at<double>(6, 2) = 1;
    tmp.at<double>(6, 3) = -(x.at<double>(0, 0) - x.at<double>(5, 0)) * sin(x.at<double>(3, 0));
    tmp.at<double>(6, 4) = cos(x.at<double>(9, 0) / x.at<double>(4, 0) + x.at<double>(7, 0)) * x.at<double>(9, 0) * pow(x.at<double>(4, 0), (-2));
    tmp.at<double>(6, 5) = -cos(x.at<double>(3, 0));
    tmp.at<double>(6, 6) = pow(x.at<double>(1, 0), 3.);
    tmp.at<double>(6, 7) = -cos(x.at<double>(9, 0) / x.at<double>(4, 0) + x.at<double>(7, 0));
    tmp.at<double>(6, 8) = 0;
    tmp.at<double>(6, 9) = -cos(x.at<double>(9, 0) / x.at<double>(4, 0) + x.at<double>(7, 0)) / x.at<double>(4, 0);
    tmp.at<double>(7, 0) = 2. * x.at<double>(4, 0) * (x.at<double>(0, 0) - 2. * x.at<double>(5, 0));
    tmp.at<double>(7, 1) = -x.at<double>(6, 0) * exp(x.at<double>(1, 0) * x.at<double>(6, 0) + x.at<double>(9, 0));
    tmp.at<double>(7, 2) = -2. * cos(-x.at<double>(8, 0) + x.at<double>(2, 0));
    tmp.at<double>(7, 3) = 0.15e1;
    tmp.at<double>(7, 4) = pow(x.at<double>(0, 0) - 2. * x.at<double>(5, 0), 2.);
    tmp.at<double>(7, 5) = -4. * x.at<double>(4, 0) * (x.at<double>(0, 0) - 2. * x.at<double>(5, 0));
    tmp.at<double>(7, 6) = -x.at<double>(1, 0) * exp(x.at<double>(1, 0) * x.at<double>(6, 0) + x.at<double>(9, 0));
    tmp.at<double>(7, 7) = 0;
    tmp.at<double>(7, 8) = 2. * cos(-x.at<double>(8, 0) + x.at<double>(2, 0));
    tmp.at<double>(7, 9) = -exp(x.at<double>(1, 0) * x.at<double>(6, 0) + x.at<double>(9, 0));
    tmp.at<double>(8, 0) = -3;
    tmp.at<double>(8, 1) = -2. * x.at<double>(7, 0) * x.at<double>(9, 0) * x.at<double>(6, 0);
    tmp.at<double>(8, 2) = 0;
    tmp.at<double>(8, 3) = exp((x.at<double>(4, 0) + x.at<double>(3, 0)));
    tmp.at<double>(8, 4) = exp((x.at<double>(4, 0) + x.at<double>(3, 0)));
    tmp.at<double>(8, 5) = -0.7e1 * pow(x.at<double>(5, 0), -2.);
    tmp.at<double>(8, 6) = -2. * x.at<double>(1, 0) * x.at<double>(7, 0) * x.at<double>(9, 0);
    tmp.at<double>(8, 7) = -2. * x.at<double>(1, 0) * x.at<double>(9, 0) * x.at<double>(6, 0);
    tmp.at<double>(8, 8) = 3;
    tmp.at<double>(8, 9) = -2. * x.at<double>(1, 0) * x.at<double>(7, 0) * x.at<double>(6, 0);
    tmp.at<double>(9, 0) = x.at<double>(9, 0);
    tmp.at<double>(9, 1) = x.at<double>(8, 0);
    tmp.at<double>(9, 2) = -x.at<double>(7, 0);
    tmp.at<double>(9, 3) = cos(x.at<double>(3, 0) + x.at<double>(4, 0) + x.at<double>(5, 0)) * x.at<double>(6, 0);
    tmp.at<double>(9, 4) = cos(x.at<double>(3, 0) + x.at<double>(4, 0) + x.at<double>(5, 0)) * x.at<double>(6, 0);
    tmp.at<double>(9, 5) = cos(x.at<double>(3, 0) + x.at<double>(4, 0) + x.at<double>(5, 0)) * x.at<double>(6, 0);
    tmp.at<double>(9, 6) = sin(x.at<double>(3, 0) + x.at<double>(4, 0) + x.at<double>(5, 0));
    tmp.at<double>(9, 7) = -x.at<double>(2, 0);
    tmp.at<double>(9, 8) = x.at<double>(1, 0);
    tmp.at<double>(9, 9) = x.at<double>(0, 0);
    return tmp;
}

cv::Mat Equations(cv::Mat x)
{
    cv::Mat tmp(10, 1, cv::DataType<double>::type);
    tmp.at<double>(0, 0) = cos(x.at<double>(0, 0) * x.at<double>(1, 0)) - exp(-3 * x.at<double>(2, 0)) + x.at<double>(3, 0) * x.at<double>(4, 0) * x.at<double>(4, 0) - x.at<double>(5, 0) - sinh(2 * x.at<double>(7, 0)) * x.at<double>(8, 0) + 2 * x.at<double>(9, 0) + 2.0004339741653854440;
    tmp.at<double>(1, 0) = sin(x.at<double>(0, 0) * x.at<double>(1, 0)) + x.at<double>(2, 0) * x.at<double>(8, 0) * x.at<double>(6, 0) - exp(-x.at<double>(9, 0) + x.at<double>(5, 0)) + 3 * x.at<double>(4, 0) * x.at<double>(4, 0) - x.at<double>(5, 0) * (x.at<double>(7, 0) + 1) + 10.886272036407019994;
    tmp.at<double>(2, 0) = x.at<double>(0, 0) - x.at<double>(1, 0) + x.at<double>(2, 0) - x.at<double>(3, 0) + x.at<double>(4, 0) - x.at<double>(5, 0) + x.at<double>(6, 0) - x.at<double>(7, 0) + x.at<double>(8, 0) - x.at<double>(9, 0) - 3.1361904761904761904;
    tmp.at<double>(3, 0) = 2 * cos(-x.at<double>(8, 0) + x.at<double>(3, 0)) + x.at<double>(4, 0) / (x.at<double>(2, 0) + x.at<double>(0, 0)) - sin(x.at<double>(1, 0) * x.at<double>(1, 0)) + cos(x.at<double>(6, 0) * x.at<double>(9, 0)) * cos(x.at<double>(6, 0) * x.at<double>(9, 0)) - x.at<double>(7, 0) - 0.170747270502230475;
    tmp.at<double>(4, 0) = sin(x.at<double>(4, 0)) + 2 * x.at<double>(7, 0) * (x.at<double>(2, 0) + x.at<double>(0, 0)) - exp(-x.at<double>(6, 0) * (-x.at<double>(9, 0) + x.at<double>(5, 0))) + 2 * cos(x.at<double>(1, 0)) - 1 / (x.at<double>(3, 0) - x.at<double>(8, 0)) - 0.3685896273101277862;
    tmp.at<double>(5, 0) = exp(x.at<double>(0, 0) - x.at<double>(3, 0) - x.at<double>(8, 0)) + x.at<double>(4, 0) * x.at<double>(4, 0) / x.at<double>(7, 0) + 0.5 * cos(3 * x.at<double>(9, 0) * x.at<double>(1, 0)) - x.at<double>(5, 0) * x.at<double>(2, 0) + 2.0491086016771875115;
    tmp.at<double>(6, 0) = x.at<double>(1, 0) * x.at<double>(1, 0) * x.at<double>(1, 0) * x.at<double>(6, 0) - sin(x.at<double>(9, 0) / x.at<double>(4, 0) + x.at<double>(7, 0)) + (x.at<double>(0, 0) - x.at<double>(5, 0)) * cos(x.at<double>(3, 0)) + x.at<double>(2, 0) - 0.738043007620279801;
    tmp.at<double>(7, 0) = x.at<double>(4, 0) * (x.at<double>(0, 0) - 2 * x.at<double>(5, 0)) * (x.at<double>(0, 0) - 2 * x.at<double>(5, 0)) - 2 * sin(-x.at<double>(8, 0) + x.at<double>(2, 0)) + 1.5 * x.at<double>(3, 0) - exp(x.at<double>(1, 0) * x.at<double>(6, 0) + x.at<double>(9, 0)) + 3.566832198969380904;
    tmp.at<double>(8, 0) = 7 / x.at<double>(5, 0) + exp(x.at<double>(4, 0) + x.at<double>(3, 0)) - 2 * x.at<double>(1, 0) * x.at<double>(7, 0) * x.at<double>(9, 0) * x.at<double>(6, 0) + 3 * x.at<double>(8, 0) - 3 * x.at<double>(0, 0) - 8.439473450838325749;
    tmp.at<double>(9, 0) = x.at<double>(0, 0) * x.at<double>(9, 0) + x.at<double>(1, 0) * x.at<double>(8, 0) - x.at<double>(2, 0) * x.at<double>(7, 0) + sin(x.at<double>(3, 0) + x.at<double>(4, 0) + x.at<double>(5, 0)) * x.at<double>(6, 0) - 0.7823809523809523809;
    return tmp;
}

cv::Mat NewtonSolve(const cv::Mat x, const double eps)
{
    cv::Mat xtmp = x.clone();
    cv::Mat y = x.clone();;
    int k = 0;
    do
    {
        y = xtmp.clone();      
        xtmp += SSLE(Jacobi(xtmp), -Equations(xtmp));
        ++k;
    } while (cv::norm(xtmp - y) >= eps);
    //std::cout << k << " iterations (classic)\n";
    return xtmp;
}

cv::Mat NewtonModSolve(const cv::Mat x, const double eps)
{
    cv::Mat xtmp = x.clone();
    cv::Mat y = x.clone();;
    int k = 0;
    cv::Mat jacobi = Jacobi(xtmp);
    do
    {
        y = xtmp.clone();
        xtmp += SSLE(jacobi, -Equations(xtmp));
        ++k;
    } while (cv::norm(xtmp - y) >= eps);
    //std::cout << k << " iterations (mod)\n";
    return xtmp;
}

cv::Mat NewtonMixSolve(const cv::Mat x, const int k, const double eps)
{
    cv::Mat xtmp = x.clone();
    cv::Mat y = x.clone();;
    cv::Mat jacobi = Jacobi(xtmp);
    int n = 0;
    do
    {
        y = xtmp.clone();
        if (n < k)
        {
            jacobi = Jacobi(xtmp);
        }
        xtmp += SSLE(jacobi, -Equations(xtmp));
        ++n;
    } while (cv::norm(xtmp - y) >= eps && n < 100);
    //std::cout << n << " iterations (mix)\n";
    return xtmp;
}

cv::Mat NewtonHybridSolve(const cv::Mat x, const int k, const double eps)
{
    cv::Mat xtmp = x.clone();
    cv::Mat y = x.clone();;
    cv::Mat jacobi = Jacobi(xtmp);
    int n = 0;
    do
    {
        y = xtmp.clone();
        ++n;
        if (n % k == 0)
        {
            jacobi = Jacobi(xtmp);
        }
        xtmp += SSLE(jacobi, -Equations(xtmp));  
    } while (cv::norm(xtmp - y) >= eps && n < 100);
    //std::cout << n << " iterations (hybrid)\n";
    return xtmp;
}