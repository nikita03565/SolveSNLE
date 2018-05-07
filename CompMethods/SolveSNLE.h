#pragma once
#include <array>
#include "opencv2/core.hpp"

cv::Mat Jacobi(const cv::Mat x);

cv::Mat Equations(const cv::Mat x);

cv::Mat NewtonSolve(const cv::Mat x, double eps = 10e-6);

cv::Mat NewtonModSolve(const cv::Mat x, const double eps = 10e-6);

cv::Mat NewtonMixSolve(const cv::Mat x, const int k, const double eps = 10e-6);

cv::Mat NewtonHybridSolve(const cv::Mat x, const int k, const double eps = 10e-6);