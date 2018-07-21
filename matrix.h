#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
// #include <functional>
#include <exception>

using namespace std;

template<typename T>
class Matrix
{
public:
    Matrix();
    Matrix(unsigned str, unsigned clmn); //создание матрицы по
    Matrix(vector<vector<T>> m_); //создание матрицы по вектору векторов
    Matrix(unsigned str, unsigned clmn, vector<T> b); //создание матрицы по вектору
   // Matrix(unsigned str, unsigned clmn, function<T(unsigned, unsigned)> f);
    template <typename Func>
    Matrix(unsigned str, unsigned clmn, Func f);
    Matrix(const Matrix &A); //copy constructor
    unsigned getStrSize() const; //size of string
    unsigned getClmnSize() const; //size of collumn
    Matrix<T> operator=(const Matrix<T> &other); //copy assignment
    vector<T> operator[](unsigned i) const; //return i-th string
    vector<T>& operator[](unsigned i);
    template<typename Y> friend Matrix<Y> operator+(const Matrix<Y> &A, const Matrix<Y> &B);
    template<typename Y> friend Matrix<Y> operator-(const Matrix<Y> &A, const Matrix<Y> &B);
    template<typename Y> friend Matrix<Y> operator*(const Matrix<Y> &A, const Matrix<Y> &B);
    Matrix<T> operator-() const;
    static Matrix<T> transpos(const Matrix &m);
    template<typename Y> friend ostream& operator<<(ostream &os, const Matrix<Y> A); //output
private:
    vector<vector<T>> m;
};

template<typename T>
Matrix<T>::Matrix() {T t(0); vector<T> tmp; tmp.push_back(t); m.push_back(tmp);}

template<typename T>
Matrix<T>::Matrix(unsigned str, unsigned clmn)
{
    T t(0);
    vector<T> tmp;
    for(unsigned i=0; i<clmn; ++i) tmp.push_back(t);
    for(unsigned i=0; i<str; ++i) m.push_back(tmp);
}

template<typename T>
Matrix<T>::Matrix(vector<vector<T>> m_):m(m_) {}

template<typename T>
Matrix<T>::Matrix(unsigned str, unsigned clmn, vector<T> b)
{
    m.resize(str);
    for(unsigned i=0; i<str; ++i) m[i].resize(clmn);
    unsigned k = 0;
    for (unsigned i = 0; i < str; i++)
    {
        for (unsigned j = 0; j < clmn; j++)
        {
            m[i][j] = b[k];
            k++;
        }
    }
}

template <typename T> template <typename Func>
Matrix<T>::Matrix(unsigned str, unsigned clmn, Func f)
{
    m.resize(str);
    for(unsigned i=0; i<str; ++i) m[i].resize(clmn);
    for (unsigned i = 0; i < str; i++)
    {
        for (unsigned j = 0; j < clmn; j++)
        {
            m[i][j] = f(i,j);
        }
    }
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T> &A):m(A.m) {}

template<typename T>
unsigned Matrix<T>::getStrSize() const {return m.empty() ? 0 : m[0].size();}

template<typename T>
unsigned Matrix<T>::getClmnSize() const {return m.size();}

template<typename T>
Matrix<T> Matrix<T>::operator=(const Matrix<T> &other)
{
    m=other.m;
    return *this;
}

template<typename T>
vector<T> Matrix<T>::operator[](unsigned i) const {return m[i];}

template<typename T>
vector<T>& Matrix<T>::operator[](unsigned i) {return m[i];}

template<typename T>
Matrix<T> operator+(const Matrix<T> &A, const Matrix<T> &B)
{
    unsigned ass(A.m.size()), acs(A.m[0].size()), bss(B.m.size()), bcs(B.m[0].size());
    unsigned str=std::max(ass,bss), clmn=std::max(acs,bcs);
    Matrix<T> res(str,clmn);
    for(unsigned i=0; i<str; ++i)
    {
        for(unsigned j=0; j<clmn; ++j)
        {
            if(i<ass && j<acs) res.m[i][j]=res.m[i][j]+A.m[i][j];
            if(i<bss && j<bcs) res.m[i][j]=res.m[i][j]+B.m[i][j];
        }
    }
    return res;
}

template<typename T>
Matrix<T> operator-(const Matrix<T> &A, const Matrix<T> &B)
{
    unsigned ass(A.m.size()), acs(A.m[0].size()), bss(B.m.size()), bcs(B.m[0].size());
    unsigned str=std::max(ass,bss), clmn=std::max(acs,bcs);
    Matrix<T> res(str,clmn);
    for(unsigned i=0; i<str; ++i)
    {
        for(unsigned j=0; j<clmn; ++j)
        {
            if(i<ass && j<acs) res.m[i][j]=res.m[i][j]+A.m[i][j];
            if(i<bss && j<bcs) res.m[i][j]=res.m[i][j]-B.m[i][j];
        }
    }
    return res;
}

template<typename T>
Matrix<T> Matrix<T>::transpos(const Matrix &m)
{
    unsigned str=m.getStrSize(), clmn=m.getClmnSize();
    Matrix<T> res(str, clmn);
    for(unsigned i=0; i<str; ++i)
    {
        for(unsigned j=0; j<clmn; ++j)
        {
            res[i][j]=m[j][i];
        }
    }
    return res;
}

template<typename T>
Matrix<T> operator*(const Matrix<T> &A, const Matrix<T> &B)
{
    unsigned ass(A.m.size()), acs(A.m[0].size()), bss(B.m.size()), bcs(B.m[0].size());
    //ass - A string size, bss - B string size
    if(acs!=bss) throw exception();
    Matrix<T> res(ass,bcs);
    for(unsigned i=0; i<ass; ++i)
    {
        for(unsigned j=0; j<bcs; ++j)
        {
            for(unsigned k=0; k<bss; ++k)
            {
                res.m[i][j]=res.m[i][j]+A.m[i][k]*B.m[k][j];
            }
        }
    }
    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator -() const
{
    auto f = [this](unsigned i, unsigned j)->T{return -this->m[i][j];};
    return Matrix<T>(getClmnSize(), getStrSize(), f);
}

template<typename T>
ostream& operator<<(ostream &os, const Matrix<T> &m)
{
    for(unsigned i=0; i<m.getStrSize(); ++i)
    {
        for(unsigned j=0; j<m.getClmnSize(); ++j)
        {
            os<<m[i][j];
            if(j!=m.getClmnSize()-1) os<<" ";
        }
        os<<"\n";
    }
    return os;
}



#endif // MATRIX_H
