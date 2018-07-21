#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "matrix.h"

matrix<float, 5, 5> gaussian_kernel(float);
void magnitude(QImage&, const QImage&, const QImage&);
template<typename T>
QImage convolution(const T&, const QImage&);
// chanelIndex: 0 - L, 1 - a, 2 - b
template<typename T, typename Y>
Matrix<Y> convolution(const T& kernel, const Matrix<Y>& labChanel, int chanelIndex);
QImage canny(const QImage&, float, float, float);
QImage sobel(const QImage&);
QImage prewitt(const QImage&);
QImage roberts(const QImage&);
QImage scharr(const QImage&);
QImage hysteresis(const QImage&, float, float);

template<typename T>
QImage convolution(const T& kernel, const QImage& image)
{
    int kw = kernel[0].size(), kh = kernel.size(),
        offsetx = kw / 2, offsety = kw / 2;
    QImage out(image.size(), image.format());
    float sum;

    quint8 *line;
    const quint8 *lookup_line;

    for (int y = 0; y < image.height(); y++)
    {
        line = out.scanLine(y);
        for (int x = 0; x < image.width(); x++)
        {
            sum = 0;

            for (int j = 0; j < kh; j++)
            {
                if (y + j < offsety || y + j >= image.height())
                    continue;
                lookup_line = image.constScanLine(y + j - offsety);
                for (int i = 0; i < kw; i++)
                {
                    if (x + i < offsetx || x + i >= image.width())
                        continue;
                    sum += kernel[j][i] * lookup_line[x + i - offsetx];
                }
            }

            line[x] = qBound(0x00, static_cast<int>(sum), 0xFF);
        }
    }

    return out;
}

template<typename T, typename Y>
Matrix<Y> convolution(const T& kernel, const Matrix<Y>& labChanel, int chanelIndex)
{
    int kw = kernel[0].size(), kh = kernel.size(),
        offsetx = kw / 2, offsety = kw / 2;
    Matrix<Y> out(labChanel.getStrSize(), labChanel.getClmnSize());
    float sum;

    for (int y = 0; y < labChanel.getStrSize(); y++)
    {
        for (int x = 0; x < labChanel.getClmnSize(); x++)
        {
            sum = 0;

            for (int j = 0; j < kh; j++)
            {
                if (y + j < offsety || y + j >= labChanel.getClmnSize())
                    continue;
                for (int i = 0; i < kw; i++)
                {
                    if (x + i < offsetx || x + i >= labChanel.getStrSize())
                        continue;
                    sum += kernel[j][i] * labChanel[y + j - offsety][x + i - offsetx];
                }
            }
            out[y][x] = sum;
            switch (chanelIndex)
            {
            case 0:
                if (sum < 0) out[y][x] = 0;
                else if (sum > 100) out[y][x] = 100;
                else out [y][x] = sum;
                break;
            case 1:
                if (sum < -86.185) out[y][x] = -86.185;
                else if (sum > 98.254) out[y][x] = 98.254;
                else out [y][x] = sum;
                break;
            case 2:
                if (sum < -107.863) out[y][x] =  -107.863;
                else if (sum >  94.482) out[y][x] = 94.482;
                else out [y][x] = sum;
                break;
            }
        }
    }
    return out;
}


#endif // ALGORITHMS_H
