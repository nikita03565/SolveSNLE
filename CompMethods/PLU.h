#ifndef __PLU__
#define __PLU__

#include <opencv2/core.hpp>

void row_swap(cv::Mat mat, const int src, const int dst);

void col_swap(cv::Mat mat, const int src, const int dst);

void PLUQ_Decomposition(const cv::Mat& src, cv::Mat& p, cv::Mat& l, cv::Mat& u, cv::Mat& q, int& rank);

void PLU_Decomposition(cv::Mat src, cv::Mat& p, cv::Mat& l, cv::Mat& u, int& rank);

void check_PLUQ(cv::Mat mat);

void check_PLU(cv::Mat mat);

double U_det_PLU(cv::Mat U);

cv::Mat SSLE(cv::Mat A, cv::Mat B);

void check_SSLE(cv::Mat A, cv::Mat B);

double det(cv::Mat A);

int rank(cv::Mat A);

cv::Mat inverse(cv::Mat A);

void check_inverse(cv::Mat A);

double conv(cv::Mat A);

#endif //__PLU__
