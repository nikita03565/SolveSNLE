#include "PLU.h"
#include <iostream>
#include <chrono>

void row_swap(cv::Mat mat, const int src, const int dst)
{
    cv::Mat tmp = mat.row(dst).clone();
    mat.row(src).copyTo(mat.row(dst));
    tmp.copyTo(mat.row(src));
}

void col_swap(cv::Mat mat, const int src, const int dst)
{
    cv::Mat tmp = mat.col(dst).clone();
    mat.col(src).copyTo(mat.col(dst));
    tmp.copyTo(mat.col(src));
}

void PLUQ_Decomposition(const cv::Mat& src, cv::Mat& p, cv::Mat& l, cv::Mat& u, cv::Mat& q, int& rank)
{
    cv::Mat src_clone = src.clone();
    p = cv::Mat::eye(src_clone.rows, src_clone.cols, cv::DataType<double>::type);
    q = cv::Mat::eye(src_clone.rows, src_clone.cols, cv::DataType<double>::type);
    l = cv::Mat::eye(src_clone.rows, src_clone.cols, cv::DataType<double>::type);
    u = cv::Mat::zeros(src_clone.rows, src_clone.cols, cv::DataType<double>::type);

    rank = src_clone.rows;
    int  row_with_max_elem = -1;
    int  col_with_max_elem = -1;
    for (int k = 0; k < src_clone.rows; ++k)
    {
        double pivot_value = 0.0;

        for (int i = k; i < src_clone.rows; ++i)
            for (int j = k; j < src_clone.cols; ++j)
                if (abs(src_clone.at<double>(i, j)) > pivot_value)
                {
                    pivot_value = abs(src_clone.at<double>(i, j));
                    row_with_max_elem = i;
                    col_with_max_elem = j;
                }
        if (pivot_value < 10e-16)
        {
            rank -= 1;
            continue;
        }
        row_swap(src_clone, k, row_with_max_elem);
        col_swap(src_clone, k, col_with_max_elem);

        row_swap(p, k, row_with_max_elem);
        col_swap(q, k, col_with_max_elem);

        for (int i = k + 1; i < src_clone.rows; ++i)
        {
            src_clone.at<double>(i, k) /= src_clone.at<double>(k, k);
            for (int j = k + 1; j < src_clone.cols; ++j)
                src_clone.at<double>(i, j) -= src_clone.at<double>(i, k) * src_clone.at<double>(k, j);
        }
    }

    for (int i = 0; i < src_clone.rows; ++i)
        for (int j = 0; j < src_clone.cols; ++j)
            if (i > j)
                l.at<double>(i, j) = src_clone.at<double>(i, j);
            else
                u.at<double>(i, j) = src_clone.at<double>(i, j);
}

void PLU_Decomposition(cv::Mat src, cv::Mat& p, cv::Mat& l, cv::Mat& u, int& rank)
{
    cv::Mat src_clone = src.clone();
    p = cv::Mat::eye(src_clone.rows, src_clone.cols, cv::DataType<double>::type);
    l = cv::Mat::zeros(src_clone.rows, src_clone.cols, cv::DataType<double>::type);
    u = cv::Mat::zeros(src_clone.rows, src_clone.cols, cv::DataType<double>::type);

    rank = src_clone.rows;
    int  row_with_max_elem = -1;

    for (int k = 0; k < src_clone.rows; ++k)
    {
        double pivot_value = 0.0;

        for (int i = k; i < src_clone.rows; ++i)
            for (int j = k; j < src_clone.cols; ++j)
                if (abs(src_clone.at<double>(i, j)) > pivot_value)
                {
                    pivot_value = abs(src_clone.at<double>(i, j));
                    row_with_max_elem = i;

                }
        if (pivot_value < 10e-16)
        {
            rank -= 1;
            continue;
        }
        row_swap(src_clone, k, row_with_max_elem);
        row_swap(p, k, row_with_max_elem);


        for (int i = k + 1; i < src_clone.rows; ++i)
        {
            src_clone.at<double>(i, k) /= src_clone.at<double>(k, k);
            for (int j = k + 1; j < src_clone.cols; ++j)
                src_clone.at<double>(i, j) -= src_clone.at<double>(i, k) * src_clone.at<double>(k, j);
        }
    }

    for (int i = 0; i < src_clone.rows; ++i)
    {
        l.at<double>(i, i) = 1.0;
        for (int j = 0; j < src_clone.cols; ++j)
            if (i > j)
                l.at<double>(i, j) = src_clone.at<double>(i, j);
            else
                u.at<double>(i, j) = src_clone.at<double>(i, j);
    }
}

void check_PLUQ(cv::Mat mat)
{
    std::cout << "mat:\n" << mat << "\n\n";

    cv::Mat p, l, u, q;
    int rank = -1;
    PLUQ_Decomposition(mat, p, l, u, q, rank);
    cv::Mat res = p.t() * l * u * q.t();

    std::cout << "p:\n" << p << '\n';
    std::cout << "l:\n" << l << '\n';
    std::cout << "u:\n" << u << '\n';
    std::cout << "q:\n" << q << "\n\n";

    std::cout << "res:\n" << res << '\n';
}

void check_PLU(cv::Mat mat)
{
    std::cout << "mat:\n" << mat << "\n\n";

    cv::Mat p, l, u;
    int rank = -1;
    PLU_Decomposition(mat, p, l, u, rank);
    cv::Mat res = p.t() * l * u;

    std::cout << "p:\n" << p << '\n';
    std::cout << "l:\n" << l << '\n';
    std::cout << "u:\n" << u << '\n';


    std::cout << "res:\n" << res << '\n';
}

double U_det_PLU(cv::Mat U)
{
    double det = 1.0;
    for (int i = 0; i < U.rows; ++i)
    {
        det *= U.at<double>(i, i);
    }
    if ((U.rows % 2 == 0) && (U.rows != 1))
    {
        det *= -1;
    }
    return det;
}

cv::Mat SSLE(cv::Mat A, cv::Mat B)
{
    cv::Mat p, l, u, q;
    int rank = -1;
    
    PLUQ_Decomposition(A, p, l, u, q, rank);    
    
    cv::Mat y(B.rows, B.cols, cv::DataType<double>::type);

    cv::Mat B_copy = p * B;

    y.at<double>(0) = B_copy.at<double>(0, 0);

    for (int i = 1; i < A.rows; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < i; ++j)
            sum += y.at<double>(j) * l.at<double>(i, j);
        y.at<double>(i) = B_copy.at<double>(i, 0) - sum;
    }
    cv::Mat x = cv::Mat::zeros(A.rows, 1, cv::DataType<double>::type);
    //std::cout << '\n' << x << '\n';
    int rows = A.rows;
    int cols = A.cols;
    for (int i = A.rows - 1; i > 0; --i)
    {
        bool f = true;
        for (int j = 0; j < cols; ++j)
        {
            if (abs(u.at<double>(i, j) - x.at<double>(j, 0)) > 10e-6)
            {
                f = false;
            }
        }
        if (f && abs(y.at<double>(i)) < 10e-6 )
        {
            rows -= 1;
            cols -= 1;
        }
            
        else 
        {
            bool f1 = true;
            for (int j = 0; j < cols; ++j)
            {
                if (abs(u.at<double>(i, j) - x.at<double>(j, 0)) > 10e-6)
                {
                    f = false;
                }
            }
            if (f && abs(y.at<double>(i)) >= 10e-6)
            {
                std::cout << "no solutuins\n";
                return x;
            }
        }   
    }
    x.at<double>(A.rows - 1, 0) = y.at<double>(A.rows - 1) / u.at<double>(A.rows - 1, A.cols - 1);

    for (int i = 2; i < A.rows + 1; ++i)
    {
        double sum = 0;
        for (int j = 1; j < i; ++j)
        {
            sum += x.at<double>(A.rows - j, 0) * u.at<double>(A.rows - i, A.cols - j);
            x.at<double>(A.rows - i, 0) = (y.at<double>(A.rows - i) - sum) / u.at<double>(A.rows - i, A.rows - i);
        }
    }
    return q * x;
}

void check_SSLE(cv::Mat A, cv::Mat B)
{
    std::cout << "Ax - B =\n" << A * SSLE(A, B) - B << "\n";
}

double det(cv::Mat A)
{
    cv::Mat p, l, u;
    int rank = -1;
    PLU_Decomposition(A, p, l, u, rank);
    return U_det_PLU(u);
}

int rank(cv::Mat A)
{
    cv::Mat p, l, u;
    int rank = -1;
    PLU_Decomposition(A, p, l, u, rank);
    return rank;
}

cv::Mat inverse(cv::Mat A)
{
    cv::Mat inv = cv::Mat::zeros(A.rows, A.cols, cv::DataType<double>::type);
    for (int i = 0; i < A.rows; ++i)
    {
        cv::Mat z = cv::Mat::zeros(A.rows, 1, cv::DataType<double>::type);
        z.at<double>(i, 0) = 1;
        cv::Mat x = SSLE(A, z);

        for (int j = 0; j < A.rows; ++j)
            inv.at<double>(j, i) = x.at<double>(j, 0);
    }
    return inv;
}

void check_inverse(cv::Mat A)
{
    std::cout << "A^-1 * A =\n" << inverse(A) * A << '\n';
    std::cout << "A * A^-1 =\n" << A * inverse(A) << '\n';
}

double conv(cv::Mat A)
{
    return cv::norm(A) * cv::norm(inverse(A));
}
