#ifndef REFEDGES_H
#define REFEDGES_H

#include "matrix.h"

QImage refEdges(const QImage& original, const QImage& grayscale, float sigma, float tmin, float tmax,
                float eps, float rho, float lambda1, float lambda2, float treshold);

Matrix<int> imageToMatrix (const QImage&);

Matrix<float> buildWLab (const Matrix<float>& C);
Matrix<float> buildWVar (const QImage& original);
Matrix<float> buildC (const Matrix<float>& L, const Matrix<float>& a, const Matrix<float>& b,
                      float sigma, float tmin, float tmax);
Matrix<float> buildOmega (float lambda1, float lambda2, float sigma, float tmin, float tmax, const QImage& original);
Matrix<float> buildPhi (const QImage& grayscale, float sigma);
Matrix<float> l1BallProjection (const Matrix<float>& E, float eps);
Matrix<float> C01Projection (const Matrix<float>& E,const QImage& Egvn);
Matrix<float> mult (float a,const Matrix<float>& B);

//Convert RGB to Lab for one pixel
void rgb2lab(int r0, int g0, int b0, float &L, float &a, float &b);

//Convert RGB to Lab for matrix
void rgb2lab (const QImage& input, Matrix<float>& L, Matrix<float>& a, Matrix<float>& b);

//Gradient
void gradient(const QImage& input, float sigma, Matrix<int>& gx,Matrix<int>& gy);
void gradient(const Matrix<float> labChanel, int chanelIndex, float sigma, Matrix<float>& gx, Matrix<float>& gy);

template<typename T> void directionalDerivative(const Matrix<T>& gx, const Matrix<T>& gy,Matrix<T>& verPl, Matrix<T>& horMin,
                           Matrix<T>& verPlhorMin, Matrix<T>& verPlhorPl);



template<typename T>
void directionalDerivative(const Matrix<T>& gx, const Matrix<T>& gy,Matrix<T>& verPl, Matrix<T>& horMin,
                                                Matrix<T>& verPlhorMin, Matrix<T>& verPlhorPl)
{
    verPl = -gx;
    horMin = -gy;
    verPlhorMin = -gx - gy;
    verPlhorPl = -gx + gy;
}



#endif // REFEDGES_H

