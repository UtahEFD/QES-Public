#pragma once

#include <type_traits>
#include <iostream>
#include "Vector3.h"


template<class T>
class Matrix3;

template<typename T>
std::ostream &operator<<(std::ostream &, const Matrix3<T> &);

template<class T>
class Matrix3sym;

template<typename T>
std::ostream &operator<<(std::ostream &, const Matrix3sym<T> &);

/* --------------------------------------------------------------------------------------
 *
 * MATRIX 3 
 *
 * -------------------------------------------------------------------------------------- */

template<class T>
class Matrix3
{
  friend std::ostream &operator<<<T>(std::ostream &, const Matrix3<T> &);

protected:
  T m_val[9];

public:
  Matrix3()
  {
    for (int k = 0; k < 9; k++)
      m_val[k] = (0);
  }

  template<typename X>
  Matrix3(const X a11, const X a12, const X a13, const X a21, const X a22, const X a23, const X a31, const X a32, const X a33)
  {
    m_val[0] = a11;
    m_val[1] = a12;
    m_val[2] = a13;
    m_val[3] = a21;
    m_val[4] = a22;
    m_val[5] = a23;
    m_val[6] = a31;
    m_val[7] = a32;
    m_val[6] = a33;
  }

  // accesses the value at position i,j
  T &operator[](const int i) const
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 9);
    return m_val[i];
  }

  // accesses the value at position i,j
  T &operator[](const int i)
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 9);
    return m_val[i];
  }

  // accesses the value at position i,j
  T &operator()(const int i, const int j) const
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 3 && j >= 0 && j < 3);
    return m_val[3 * i + j];
  }

  // accesses the value at position i,j
  T &operator()(const int i, const int j)
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 3 && j >= 0 && j < 3);
    return m_val[3 * i + j];
  }


  // returns if two Vector3 values of the same type are equal
  bool operator==(const Matrix3<T> &v)
  {
    if (std::is_same<T, float>::value || std::is_same<T, double>::value)
      return FLOATS_ARE_EQUAL(m_val[0], v.m_val[0]) && FLOATS_ARE_EQUAL(m_val[1], v.m_val[1]) && FLOATS_ARE_EQUAL(m_val[2], v.m_val[2]) && FLOATS_ARE_EQUAL(m_val[3], v.m_val[3]) && FLOATS_ARE_EQUAL(m_val[4], v.m_val[4]) && FLOATS_ARE_EQUAL(m_val[5], v.m_val[5]) && FLOATS_ARE_EQUAL(m_val[6], v.m_val[6]) && FLOATS_ARE_EQUAL(m_val[7], v.m_val[7]) && FLOATS_ARE_EQUAL(m_val[8], v.m_val[8]);

    else
      return v.m_val[0] == m_val[0] && v.m_val[1] == m_val[1] && v.m_val[2] == m_val[2] && v.m_val[3] == m_val[3] && v.m_val[4] == m_val[4] && v.m_val[5] == m_val[5] && v.m_val[6] == m_val[6] && v.m_val[7] == m_val[7] && v.m_val[8] == m_val[8];
  }

  // assignment operator
  Matrix3<T> &operator=(const Matrix3<T> &M)
  {
    for (int k = 0; k < 9; k++)
      m_val[k] = M.m_val[k];
    return *this;
  }

  // assignment operator (from a symetric matrix)
  Matrix3<T> &operator=(Matrix3sym<T> &M)
  {
    m_val[0] = M[0];
    m_val[1] = M[1];
    m_val[2] = M[2];
    m_val[3] = M[1];
    m_val[4] = M[3];
    m_val[5] = M[4];
    m_val[6] = M[2];
    m_val[7] = M[4];
    m_val[8] = M[5];
    return *this;
  }

  // assignment addition operator
  Matrix3<T> &operator+=(const Matrix3<T> &M)
  {
    for (int k = 0; k < 9; k++)
      m_val[k] += M.m_val[k];
    return *this;
  }

  // assignmment substartion operator
  Matrix3<T> &operator-=(const Matrix3<T> &M)
  {
    for (int k = 0; k < 9; k++)
      m_val[k] -= M.m_val[k];
    return *this;
  }

  // assignmment multiplication operator
  Matrix3<T> &operator*=(const T &a)
  {
    for (int k = 0; k < 9; k++)
      m_val[k] *= a;
    return *this;
  }

  // assignment division operator
  Matrix3<T> &operator/=(const T &a)
  {
    for (int k = 0; k < 9; k++)
      m_val[k] /= a;
    return *this;
  }

  // addition operator
  Matrix3<T> operator+(const Matrix3<T> &M)
  {
    Matrix3<T> R;
    for (int k = 0; k < 9; k++)
      R.m_val[k] = m_val[k] + M.m_val[k];

    return R;
  }

  // substraction operator
  Matrix3<T> operator-(const Matrix3<T> &M)
  {
    Matrix3<T> R;
    for (int k = 0; k < 9; k++)
      R.m_val[k] = m_val[k] - M.m_val[k];
    return R;
  }

  // matrix multiplication
  Matrix3<T> operator*(const Matrix3<T> &M) const
  {
    Matrix3<T> R(0);
    for (int i(0); i < 3; i++)// line
      for (int j(0); j < 3; j++)// colunm
        for (int k(0); k < 3; k++)
          R.m_val[i + 3 * j] += m_val[i + 3 * k] * M.m_val[k + 3 * j];
    return R;
  }

  // matrix multiplication (with symetric matrix)
  Matrix3<T> operator*(const Matrix3sym<T> &M) const
  {
    Matrix3<T> R(0);
    for (int i(0); i < 3; i++)// line
      for (int j(0); j < 3; j++)// colunm
        for (int k(0); k < 3; k++)
          R.m_val[i + 3 * j] += m_val[i + 3 * k] * M.m_val[k + 3 * j];
    return R;
  }

  // vector matrix multiplication
  Vector3<T> operator*(const Vector3<T> &V) const
  {
    Vector3<T> R(0);
    for (int i(0); i < 3; i++)// line
      for (int k(0); k < 3; k++)
        R.m_val[i] += m_val[i + 3 * k] * V.m_val[k];
    return R;
  }

  // multiplication by scaler
  friend Matrix3<T> operator*(const T &a, const Matrix3<T> &M)
  {
    Matrix3<T> R;
    for (int k = 0; k < 9; k++)
      R.m_val[k] = a * M.m_val[k];
    return R;
  }

  // division by scalar
  Matrix3<T> operator/(const T &a)
  {
    Matrix3<T> R;
    for (int k = 0; k < 9; k++)
      R.m_val[k] = m_val[k] / a;
    return R;
  }

  // set identity matrix
  void setIdentity()
  {
    m_val[0] = 1;
    m_val[1] = 0;
    m_val[2] = 0;
    m_val[3] = 0;
    m_val[4] = 1;
    m_val[5] = 0;
    m_val[6] = 0;
    m_val[7] = 0;
    m_val[6] = 1;
  }

  // inplace inversion
  Matrix3<T> invert()
  {
    // now calculate the determinant
    T det = m_val[0] * (m_val[4] * m_val[8] - m_val[5] * m_val[7]) - m_val[1] * (m_val[3] * m_val[8] - m_val[5] * m_val[6]) + m_val[2] * (m_val[1] * m_val[7] - m_val[4] * m_val[6]);
    // check for near zero value determinants
    if (std::abs(det) < 1e-10) {
      std::cout << "WARNING (Plume::invert3): matrix nearly singular" << std::endl;
      std::cout << "abs(det) = \"" << std::abs(det) << std::endl;
      det = 10e10;
    }

    // calculate the inverse (inverted matrix depends on other components of the matrix)
    Matrix3<T> Minv;
    Minv[0] = (m_val[4] * m_val[8] - m_val[5] * m_val[7]) / det;
    Minv[1] = -(m_val[1] * m_val[8] - m_val[2] * m_val[7]) / det;
    Minv[2] = (m_val[1] * m_val[5] - m_val[4] * m_val[2]) / det;

    Minv[3] = -(m_val[1] * m_val[8] - m_val[5] * m_val[6]) / det;
    Minv[4] = (m_val[0] * m_val[8] - m_val[2] * m_val[6]) / det;
    Minv[5] = -(m_val[0] * m_val[5] - m_val[2] * m_val[1]) / det;

    Minv[6] = (m_val[1] * m_val[7] - m_val[6] * m_val[4]) / det;
    Minv[7] = -(m_val[0] * m_val[7] - m_val[1] * m_val[6]) / det;
    Minv[8] = (m_val[0] * m_val[4] - m_val[1] * m_val[3]) / det;

    for (int k = 0; k < 9; k++)
      m_val[k] = Minv.m_val[k];

    return *this;
  }

  // read matrix from istream
  friend std::istream &operator>>(std::istream &is, Matrix3<T> &M)
  {
    is >> M.m_val[0] >> M.m_val[1] >> M.val[2] >> M.m_val[3] >> M.m_val[4] >> M.val[5] >> M.m_val[6] >> M.m_val[7] >> M.val[8];
    return is;
  }
};

// output to ostream
template<typename T>
std::ostream &operator<<(std::ostream &out, const Matrix3<T> &M)
{
  out << "[";
  out << "[" << M.m_val[0] << ", " << M.m_val[1] << ", " << M.m_val[2] << "]";
  out << "[" << M.m_val[3] << ", " << M.m_val[4] << ", " << M.m_val[5] << "]";
  out << "[" << M.m_val[6] << ", " << M.m_val[7] << ", " << M.m_val[8] << "]";
  out << "]";
  out << "[";
  return out;
}


/* --------------------------------------------------------------------------------------
 *
 * MATRIX 3 SYMETRIC
 *
 * -------------------------------------------------------------------------------------- */

template<class T>
class Matrix3sym
{
  friend std::ostream &operator<<<T>(std::ostream &, const Matrix3sym<T> &);

protected:
  T m_val[6];

public:
  Matrix3sym()
  {
    for (int k = 0; k < 6; k++)
      m_val[k] = (0);
  }

  template<typename X>
  Matrix3sym(const X a11, const X a12, const X a13, const X a22, const X a23, const X a33)
  {
    m_val[0] = a11;
    m_val[1] = a12;
    m_val[2] = a13;
    m_val[3] = a22;
    m_val[4] = a23;
    m_val[5] = a33;
  }

  // accesses the value at position i,j
  T &operator[](const int i) const
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 6);
    return m_val[i];
  }

  // accesses the value at position i,j
  T &operator[](const int i)
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 6);
    return m_val[i];
  }

  // accesses the value at position i,j
  T &operator()(const int i, const int j) const
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 3 && j >= i && j < 3);
    return m_val[i + j + ceil(0.5 * i)];
  }

  // accesses the value at position i,j
  T &operator()(const int i, const int j)
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 3 && j >= 0 && j < 3);
    return m_val[i + j + ceil(0.5 * i)];
  }

  // returns if two Vector3 values of the same type are equal
  bool operator==(const Matrix3<T> &v)
  {
    if (std::is_same<T, float>::value || std::is_same<T, double>::value)
      return FLOATS_ARE_EQUAL(m_val[0], v.m_val[0]) && FLOATS_ARE_EQUAL(m_val[1], v.m_val[1]) && FLOATS_ARE_EQUAL(m_val[2], v.m_val[2]) && FLOATS_ARE_EQUAL(m_val[3], v.m_val[3]) && FLOATS_ARE_EQUAL(m_val[4], v.m_val[4]) && FLOATS_ARE_EQUAL(m_val[5], v.m_val[5]);
    else
      return v.m_val[0] == m_val[0] && v.m_val[1] == m_val[1] && v.m_val[2] == m_val[2] && v.m_val[3] == m_val[3] && v.m_val[4] == m_val[4] && v.m_val[5] == m_val[5];
  }

  // assignment operator
  Matrix3sym<T> &operator=(const Matrix3sym<T> &M)
  {
    for (int k = 0; k < 6; k++)
      m_val[k] = M.m_val[k];
    return *this;
  }

  // assignment operator
  Matrix3sym<T> &operator+=(const Matrix3sym<T> &M)
  {
    for (int k = 0; k < 6; k++)
      m_val[k] += M.m_val[k];
    return *this;
  }

  // assignment operator
  Matrix3sym<T> &operator-=(const Matrix3sym<T> &M)
  {
    for (int k = 0; k < 6; k++)
      m_val[k] -= M.m_val[k];
    return *this;
  }

  // assignment operator
  Matrix3sym<T> &operator*=(const T &a)
  {
    for (int k = 0; k < 6; k++)
      m_val[k] *= a;
    return *this;
  }

  // assignment operator
  Matrix3sym<T> &operator/=(const T &a)
  {
    for (int k = 0; k < 6; k++)
      m_val[k] /= a;
    return *this;
  }

  // addition operator
  Matrix3sym<T> operator+(const Matrix3sym<T> &M)
  {
    Matrix3sym<T> R;
    for (int k = 0; k < 6; k++)
      R.m_val[k] = m_val[k] + M.m_val[k];
    return R;
  }

  // substraction operator
  Matrix3sym<T> operator-(const Matrix3sym<T> &M)
  {
    Matrix3sym<T> R;
    for (int k = 0; k < 6; k++)
      R.m_val[k] = m_val[k] - M.m_val[k];
    return R;
  }

  // multiplication by scaler
  friend Matrix3sym<T> operator*(const T &a, const Matrix3sym<T> &M)
  {
    Matrix3sym<T> R;
    for (int k = 0; k < 6; k++)
      R.m_val[k] = a * M.m_val[k];
    return R;
  }

  // division by scalar
  Matrix3sym<T> operator/(const T &a)
  {
    Matrix3sym<T> R;
    for (int k = 0; k < 6; k++)
      R.m_val[k] = m_val[k] / a;
    return R;
  }

  void setIdentity()
  {
    m_val[0] = 1;
    m_val[1] = 0;
    m_val[2] = 0;
    m_val[3] = 1;
    m_val[4] = 0;
    m_val[5] = 1;
  }

  // calculate invariants
  Vector3<T> calcInvariants()
  {
    Vector3<T> invar;
    invar[0] = m_val[0] + m_val[3] + m_val[5];
    invar[1] = m_val[0] * m_val[3] + m_val[0] * m_val[5] + m_val[3] * m_val[5] - m_val[1] * m_val[1] - m_val[2] * m_val[2] - m_val[4] * m_val[4];
    invar[3] = m_val[0] * (m_val[3] * m_val[5] - m_val[4] * m_val[4]) - m_val[1] * (m_val[1] * m_val[5] - m_val[4] * m_val[2]) + m_val[2] * (m_val[1] * m_val[4] - m_val[3] * m_val[2]);
    return invar;
  }

  // make realizable
  Matrix3sym<T> &makeRealizable(const T &invarianceTol)
  {
    // first calculate the invariants and see if they are already realizable
    Vector3<T> invar = this.calcInvariants();

    if (invar[0] > invarianceTol && invar[1] > invarianceTol && invar[2] > invarianceTol) {
      return *this;// is already realizable -> what should be return here??? -Pete
    }

    // since tau is not already realizable, need to make it realizeable
    // start by making a guess of ks, the subfilter scale tke
    // I keep wondering if we can use the input Turb->tke for this or if we should leave it as is
    T b = 4.0 / 3.0 * invar[0];
    T c = invar[1];
    T ks = 1.01 * (-b + std::sqrt(b * b - 16.0 / 3.0 * c)) / (8.0 / 3.0);

    // if the initial guess is bad, use the straight up invar_xx value
    if (ks < invarianceTol || isnan(ks)) {
      ks = 0.5 * std::abs(invar[0]);// also 0.5*abs(invar_xx)
    }

    Matrix3sym<T> M;
    M[0] = m_val[0] + 2.0 / 3.0 * ks;
    M[1] = m_val[1];
    M[2] = m_val[2];
    M[3] = m_val[3] + 2.0 / 3.0 * ks;
    M[4] = m_val[4];
    M[5] = m_val[5] + 2.0 / 3.0 * ks;

    invar = M.calcInvariants();

    // now adjust the diagonals by 0.05% of the trace, till M is realizable
    int iter = 0;
    while ((invar[0] < invarianceTol || invar[1] < invarianceTol || invar[2] < invarianceTol) && iter < 1000) {
      iter = iter + 1;

      // increase trace by 5%
      ks = ks * 1.050;

      // note that the right hand side is not M, to force to only increase by increasing ks
      M[0] = m_val[0] + 2.0 / 3.0 * ks;
      M[3] = m_val[3] + 2.0 / 3.0 * ks;
      M[5] = m_val[5] + 2.0 / 3.0 * ks;

      invar = M.calcInvariants();
    }

    if (iter == 999) {
      std::cout << "WARNING (Plume::makeRealizable): unable to make stress tensor realizble.";
    }

    for (int k = 0; k < 6; k++)
      m_val[k] = M.m_val[k];

    return *this;
  }

  // read from istream
  friend std::istream &operator>>(std::istream &is, Matrix3sym<T> &M)
  {
    is >> M.m_val[0] >> M.m_val[1] >> M.val[2] >> M.m_val[3] >> M.val[4] >> M.val[5];
    return is;
  }
};

// write to ostream
template<typename T>
std::ostream &operator<<(std::ostream &out, const Matrix3sym<T> &M)
{
  out << "[";
  out << "[" << M.m_val[0] << ", " << M.m_val[1] << ", " << M.m_val[2] << "]";
  out << "[" << M.m_val[1] << ", " << M.m_val[3] << ", " << M.m_val[4] << "]";
  out << "[" << M.m_val[2] << ", " << M.m_val[4] << ", " << M.m_val[5] << "]";
  out << "]";
  return out;
}
