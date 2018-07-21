#include <QtWidgets>
#include <cmath>
#include <utility>
#include <queue>
#include <limits>
#include <set>
#include <stdlib.h>
#include <algorithm>
#include "kernels.h"
#include "algorithms.h"
#include "refedges.h"
#include "matrix.h"


using namespace std;


template <typename T> Matrix<T> entrywiseMult(const Matrix<T>& A,const Matrix<T>& B);


template <typename T> Matrix<T> entrywiseMult(const Matrix<T>& A,const Matrix<T>& B)
{
    unsigned ass(A.getStrSize()), acs(A.getClmnSize()), bss(B.getStrSize()), bcs(B.getClmnSize());
    if (ass!=bss || acs!=bcs) throw exception();
    auto f=[&A, &B](unsigned i, unsigned j)->T{return A[i][j]*B[i][j];};
    return Matrix<T>(acs,ass,f);
}

Matrix<int> imageToMatrix(const QImage& input)
{
    Matrix<int> res(input.height(), input.width());
    const quint8 *line;
    for (int y = 0; y < input.height(); y++)
    {
        line = input.constScanLine(y);
        for (int x = 0; x < input.width(); x++)
        {
            res[y][x] = line[x];
        }
    }

    return res;
}


//Convert RGB to Lab for one pixel
void rgb2lab(int r0, int g0, int b0, float &L1, float &a1, float &b1)
{
    float r = r0 / 255.0;
    float g = g0 / 255.0;
    float b = b0 / 255.0;

    float x, y, z;

    r = (r > 0.04045) ? pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = (g > 0.04045) ? pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = (b > 0.04045) ? pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;

    x = (x > 0.008856) ? pow(x, 1/3) : (7.787 * x) + 16/116;
    y = (y > 0.008856) ? pow(y, 1/3) : (7.787 * y) + 16/116;
    z = (z > 0.008856) ? pow(z, 1/3) : (7.787 * z) + 16/116;

    L1 = (116 * y) - 16;
    a1 = 500 * (x - y);
    b1 = 200 * (y - z);
}

//Convert RGB to Lab for matrix
void rgb2lab (const QImage& input, Matrix<float>& L, Matrix<float>& a, Matrix<float>& b)
{
    L = Matrix<float> (input.height(), input.width());
    a = Matrix<float> (input.height(), input.width());
    b = Matrix<float> (input.height(), input.width());

    for (int y = 0; y < input.height(); y++)
    {
        for (int x = 0; x < input.width(); x++)
        {
            QRgb pixel = input.pixel(y,x);

            int r1 = qRed(pixel);
            int g1 = qGreen(pixel);
            int b1 = qBlue(pixel);

            float LPixel, aPixel, bPixel;

            rgb2lab(r1, g1, b1, LPixel, aPixel, bPixel);

            L[y][x] = LPixel;
            a[y][x] = aPixel;
            b[y][x] = bPixel;
        }
    }
}

void gradient(const QImage& input, float sigma, Matrix<int>& gx,Matrix<int>& gy)
{
    QImage tmp = convolution(gaussian_kernel(sigma), input); // Gaussian blur
           // Gradients
    QImage gxImage = convolution(sobelx, tmp);
    QImage gyImage = convolution(sobely, tmp);

    gx = imageToMatrix(gxImage);
    gy = imageToMatrix(gyImage);
}

void gradient(const Matrix<float> labChanel, int chanelIndex, float sigma, Matrix<float>& gx, Matrix<float>& gy)
{
    Matrix<float> tmp = convolution(gaussian_kernel(sigma), labChanel, chanelIndex); // Gaussian blur
           // Gradients

    gx = convolution(sobelx,tmp, chanelIndex);
    gy = convolution(sobely, tmp, chanelIndex);
}

Matrix<float> buildC (const Matrix<float>& L, const Matrix<float>& a, const Matrix<float>& b,
                      float sigma, float tmin, float tmax)
{
   Matrix<float> Lgx, Lgy, agx, agy, bgx, bgy;
   gradient(L, 0, sigma, Lgx, Lgy);
   gradient(a, 1, sigma, agx, agy);
   gradient(b, 2, sigma, bgx, bgy);

   Matrix<float> LverPl, LhorMin, LverPlhorMin, LverPlhorPl;
   Matrix<float> averPl, ahorMin, averPlhorMin, averPlhorPl;
   Matrix<float> bverPl, bhorMin, bverPlhorMin, bverPlhorPl;

   directionalDerivative(Lgx, Lgy, LverPl, LhorMin, LverPlhorMin, LverPlhorPl);
   directionalDerivative(agx, agy, averPl, ahorMin, averPlhorMin, averPlhorPl);
   directionalDerivative(bgx, bgy, bverPl, bhorMin, bverPlhorMin, bverPlhorPl);

   auto fverPl = [&LverPl, &averPl, &bverPl](unsigned i, unsigned j)->float{
       return sqrt(LverPl[i][j]*LverPl[i][j]+averPl[i][j]*averPl[i][j]+bverPl[i][j]*bverPl[i][j]);};

   auto fhorMin = [&LhorMin, &ahorMin, &bhorMin](unsigned i, unsigned j)->float{
       return sqrt(LhorMin[i][j]*LhorMin[i][j]+ahorMin[i][j]*ahorMin[i][j]+bhorMin[i][j]*bhorMin[i][j]);};

   auto fverPlhorMin = [&LverPlhorMin, &averPlhorMin, &bverPlhorMin](unsigned i, unsigned j)->float{
       return sqrt(LverPlhorMin[i][j]*LverPlhorMin[i][j]+averPlhorMin[i][j]*averPlhorMin[i][j]
                   + bverPlhorMin[i][j]*bverPlhorMin[i][j]);};

   auto fverPlhorPl = [&LverPlhorPl, &averPlhorPl, &bverPlhorPl](unsigned i, unsigned j)->float{
       return sqrt(LverPlhorPl[i][j]*LverPlhorPl[i][j]+averPlhorPl[i][j]*averPlhorPl[i][j]
                   + bverPlhorPl[i][j]*bverPlhorPl[i][j]);};

   Matrix<float> CverPl(LverPl.getClmnSize(), LverPl.getStrSize(),fverPl);
   Matrix<float> ChorMin(LverPl.getClmnSize(), LverPl.getStrSize(),fhorMin);
   Matrix<float> CverPlhorMin(LverPl.getClmnSize(),LverPl.getStrSize(),fverPlhorMin);
   Matrix<float> CverPlhorPl(LverPl.getClmnSize(), LverPl.getStrSize(),fverPlhorPl);

   return CverPl + ChorMin + CverPlhorMin + CverPlhorPl;
}

Matrix<float> buildWLab (const Matrix<float>& C)
{
    auto f = [&C](unsigned i, unsigned j)->float
    {
        float mLab = - numeric_limits<float>::max(); //the smallest float
        float res = 0;

        for (int k = - 1; k <= 1; ++k)
        {
            if (i + k < 0 || i + k >= C.getClmnSize()) continue;
            for (int l = - 1; l <= 1; ++l)
            {
                if (j + l < 0 || j + l >= C.getStrSize() ) continue;
                mLab = max(mLab, C[i+k][j+l]);
                res += C[i+k][j+l];
            }
        }
        return res/mLab;
    };
    return Matrix<float> (C.getClmnSize(), C.getStrSize(), f);
}

Matrix<float> buildWVar (const QImage& original)
{
    auto fRed = [&original](unsigned i, unsigned j)->float
    {
       int card = 4; //мощность, кол-во елем в этом мн
       float mean = 0; //среднее

       for (int k = - 1; k <= 1; ++k)
       {
           if (i + k < 0 || i - k >= original.height() || i - k < 0 || i + k >= original.height() ) continue;
           for (int l = - 1; l <= 0; ++l)
           {
               if (k == 0 && l == 0 ) continue;
               if (j + l < 0 || j - l >= original.width()) continue;

               ++card;

               if (k == 1) ++card;

               QRgb pixel1 = original.pixel(i+k, j+l);
               QRgb pixel2 = original.pixel(i-k, j-l);

               mean += qRed(pixel1) - qRed(pixel2);
           }
       }
       mean /= card;

       float res = 0;

       for (int k = - 1; k <= 1; ++k)
       {
           if (i + k < 0 || i - k >= original.height() || i - k < 0 || i + k >= original.height() ) continue;
           for (int l = - 1; l <= 1; ++l)
           {
               if (j + l < 0 || j + l >= original.width() || j - l < 0 || j - l >= original.width() ) continue;

               QRgb pixel1 = original.pixel(i+k, j+l);
               QRgb pixel2 = original.pixel(i-k, j-l);

               float tmp = qRed(pixel1) - qRed(pixel2) - mean;
               res += tmp * tmp;
           }
       }
       return res / card;
    };


    auto fGreen = [&original](unsigned i, unsigned j)->float
    {
       int card = 4; //мощность, кол-во елем в этом мн
       float mean = 0; //среднее

       for (int k = - 1; k <= 1; ++k)
       {
           if (i + k < 0 || i - k >= original.height() || i - k < 0 || i + k >= original.height() ) continue;
           for (int l = - 1; l <= 0; ++l)
           {
               if (k == 0 && l == 0 ) continue;
               if (j + l < 0 || j - l >= original.width() ) continue;

               ++card;

               if (k == 1) ++card;

               QRgb pixel1 = original.pixel(i+k, j+l);
               QRgb pixel2 = original.pixel(i-k, j-l);

               mean += qGreen(pixel1) - qGreen(pixel2);
           }
       }
       mean /= card;

       float res = 0;

       for (int k = - 1; k <= 1; ++k)
       {
           if (i + k < 0 || i - k >= original.height() || i - k < 0 || i + k >= original.height() ) continue;
           for (int l = - 1; l <= 1; ++l)
           {
               if (j + l < 0 || j + l >= original.width() || j - l < 0 || j - l >= original.width() ) continue;

               QRgb pixel1 = original.pixel(i+k, j+l);
               QRgb pixel2 = original.pixel(i-k, j-l);

               float tmp = qGreen(pixel1) - qGreen(pixel2) - mean;
               res += tmp * tmp;
           }
       }
       return res / card;
    };


    auto fBlue = [&original](unsigned i, unsigned j)->float
    {
       int card = 4; //мощность, кол-во елем в этом мн
       float mean = 0; //среднее

       for (int k = - 1; k <= 1; ++k)
       {
           if (i + k < 0 || i - k >= original.height() || i - k < 0 || i + k >= original.height() ) continue;
           for (int l = - 1; l <= 0; ++l)
           {
               if (k == 0 && l == 0 ) continue;
               if (j + l < 0 || j - l >= original.width() ) continue;

               ++card;

               if (k == 1) ++card;

               QRgb pixel1 = original.pixel(i+k, j+l);
               QRgb pixel2 = original.pixel(i-k, j-l);

               mean += qBlue(pixel1) - qBlue(pixel2);
           }
       }
       mean /= card;

       float res = 0;

       for (int k = - 1; k <= 1; ++k)
       {
           if (i + k < 0 || i - k >= original.height() || i - k < 0 || i + k >= original.height() ) continue;
           for (int l = - 1; l <= 1; ++l)
           {
               if (j + l < 0 || j + l >= original.width() || j - l < 0 || j - l >= original.width() ) continue;

               QRgb pixel1 = original.pixel(i+k, j+l);
               QRgb pixel2 = original.pixel(i-k, j-l);

               float tmp = qBlue(pixel1) - qBlue(pixel2) - mean;
               res += tmp * tmp;
           }
       }
       return res / card;
    };

    Matrix<float> V1 (original.height(), original.width(), fRed);
    Matrix<float> V2 (original.height(), original.width(), fGreen);
    Matrix<float> V3 (original.height(), original.width(), fBlue);

    Matrix<float> V = V1 + V2 + V3;

    auto f = [&V](unsigned i, unsigned j)->float
    {
        float mVar = - numeric_limits<float>::max(); //the smallest float
        float res = 0;

        for (int k = - 1; k <= 1; ++k)
        {
            if (i + k < 0 || i + k >= V.getClmnSize() ) continue;
            for (int l = - 1; l <= 1; ++l)
            {
                if (j + l < 0 || j + l >= V.getStrSize() ) continue;
                mVar = max(mVar, V[i+k][j+l]);
                res += V[i+k][j+l];
            }
        }
        return res / mVar;
    };
    return Matrix<float> (original.height(),original.width(), f);
}

Matrix<float> buildOmega (float lambda1, float lambda2, float sigma, float tmin, float tmax, const QImage& original)
{
    Matrix<float> L, a, b;
    rgb2lab(original,L, a ,b);

    Matrix<float> C = buildC(L, a ,b, sigma, tmin, tmax);

    Matrix<float> WLab = buildWLab(C);
    Matrix<float> WVar = buildWVar(original);

    auto f = [lambda1, lambda2, &WLab, &WVar](unsigned i, unsigned j) -> float
    {
     return sqrt(lambda1 * WLab[i][j] * WLab[i][j] + lambda2 * WVar[i][j] * WVar[i][j]);
    };

    return Matrix<float> (original.height(), original.width(), f);
}

Matrix<float> buildPhi (const QImage& grayscale, float sigma)
{
    Matrix<int> gx, gy;
    gradient(grayscale, sigma, gx, gy);
    Matrix<int> verPl, horMin, verPlhorMin, verPlhorPl;
    directionalDerivative (gx, gy, verPl, horMin, verPlhorMin, verPlhorPl);
    auto f = [&verPlhorPl](unsigned i, unsigned j) -> float {return verPlhorPl [i][j];};

    return Matrix<float> (grayscale.height(), grayscale.width(), f);
}

Matrix<float> l1BallProjection (const Matrix<float>& E, float eps)
{
    set<int> U;
    int n = E.getStrSize() * E.getClmnSize();
    for (int i=0; i < n; ++i)
    {
        U.insert(i);
    }
    float s = 0, rho = 0;
    while (!U.empty())
    {
        set<int> G, L;
        int position = rand() % U.size();
        set<int>::iterator it(U.begin());
        for (int i = 0; i < position; ++i)
        {
           ++it;
        }
        //advance(it, position);
        int k = *it;
        int i = k / E.getStrSize();
        int j = k - i * E.getStrSize();
        float Ek = E[i][j];
        float deltaS = 0;

        for (set<int>::const_iterator it = U.cbegin(); it != U.cend(); ++it)
        {
            int k = *it;
            int i = k / E.getStrSize();
            int j = k - i * E.getStrSize();

            if(E[i][j] >= Ek) {G.insert(k); deltaS += E[i][j];}
            else L.insert(k);
        }

        int deltaRho = G.size();

        if (s + deltaS - (rho + deltaRho) * Ek < eps)
        {
            s = s + deltaS;
            rho = rho + deltaRho;
            U = L;
        }

        else
        {
            G.erase(k);
            U = G;
        }
    }

    float theta = (s - eps) / rho;
    auto f = [&E, theta](unsigned i, unsigned j) -> float { return max<float> (E[i][j] - theta, 0);};

    return Matrix<float> (E.getClmnSize(), E.getStrSize(), f);
}

Matrix<float> C01Projection (const Matrix<float>& E,const QImage& Egvn)
{
    auto f = [&E, &Egvn](unsigned i, unsigned j) -> float
    {
        float res = 0;
        const quint8* line = Egvn.constScanLine(i);

        if (line[j] == 0xFF && 0 <= E[i][j] && E[i][j] <= 1) res = E[i][j];
        else if (line[j] == 0xFF && E[i][j] > 1) res = 1;

        return res;
    };

    return Matrix<float> (Egvn.height(), Egvn.width(), f);
}

Matrix<float> specialInverse (const Matrix<float>& E)
{
    auto f = [&E](unsigned i, unsigned j) -> float {return 1. / E[i][j];};
    return Matrix<float> (E.getClmnSize(), E.getStrSize(), f);
}

Matrix<float> mult (float a,const Matrix<float>& B)
{
    return Matrix<float> (B.getClmnSize(), B.getStrSize(), [a, &B](unsigned i, unsigned j) -> float {return a * B[i][j];});
}

QImage refEdges(const QImage& original, const QImage& grayscale, float sigma, float tmin, float tmax,
                float eps, float rho, float lambda1, float lambda2, float treshold)
{
    QImage Egvn = canny(grayscale, sigma, tmin, tmax);

    Matrix<float> Phi = buildPhi(grayscale, sigma);
    Matrix<float> Omega = buildOmega(lambda1, lambda2, sigma, tmin, tmax, original);
    Matrix<int> tmp = imageToMatrix(Egvn);
    Matrix<float> E (tmp.getClmnSize(), tmp.getStrSize(), [&tmp](unsigned i, unsigned j) -> float {return tmp[i][j] / 255.0;});
    Matrix<float> Y1 = entrywiseMult(Phi, E);
    Matrix<float> Theta1 = Y1;
    Matrix<float> Y2 = E;
    Matrix<float> Theta2 = E;

    Matrix<float> I (original.height(), original.height(), [](unsigned i, unsigned j) -> float {return 1;});

    for (int k = 0; k < 15; ++k)
    {
        Matrix<float> tmp1 = specialInverse(Omega * Matrix<float>::transpos(Omega)
                                           + mult(rho, Phi * Matrix<float>::transpos(Phi) + I));
        Matrix<float> tmp2 = entrywiseMult(Phi, mult(rho, Y1) - Theta1) + mult(rho, Y2) - Theta2;
        E = C01Projection(tmp1 * tmp2, Egvn);
        Y1 = mult(rho, entrywiseMult(Phi, E) + mult(1 / rho, Theta1));
        Y2 = l1BallProjection( E + mult(1 / rho, Theta2), eps);
        Theta1 = Theta1 + mult(rho, entrywiseMult(Phi, E) - Y1);
        Theta2 = Theta2 +  mult(rho, E - Y2);
    }

    E = Matrix<float>::transpos(E);

    QImage res(E.getClmnSize(), E.getStrSize(), QImage::Format_Grayscale8);

    for (int i = 0; i < E.getClmnSize(); ++i)
    {
        for (int j = 0; j < E.getStrSize(); ++j)
        {
            if (E[i][j] >= treshold) res.setPixel(i,j, qRgb(1,1,1));
            else res.setPixel(i,j, qRgb(0,0,0));
        }
    }

    return res;

}

