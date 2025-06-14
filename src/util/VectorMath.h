/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
 *
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file VectorMath.h
 * @brief
 */

#pragma once

#include <iostream>
#include <cstdlib>
#include <cmath>

typedef struct mat3sym
{
  float _11;
  float _12;
  float _13;
  float _22;
  float _23;
  float _33;

  /*mat3sym()
    : _11(0.0), _12(0.0), _13(0.0),
      _22(0.0), _23(0.0),
      _33(0.0)
  {}
  mat3sym(const float &a11, const float &a12, const float &a13, const float &a22, const float &a23, const float &a33)
    : _11(a11), _12(a12), _13(a13),
      _22(a22), _23(a23),
      _33(a33)
      {}*/
} mat3sym;


typedef struct mat3
{
  float _11;
  float _12;
  float _13;
  float _21;
  float _22;
  float _23;
  float _31;
  float _32;
  float _33;

  /*mat3()
    : _11(0.0), _12(0.0), _13(0.0),
      _21(0.0), _22(0.0), _23(0.0),
      _31(0.0), _32(0.0), _33(0.0)
  {}
  mat3(const float &a11, const float &a12, const float &a13, const float &a21, const float &a22, const float &a23, const float &a31, const float &a32, const float &a33)
    : _11(a11), _12(a12), _13(a13),
      _21(a21), _22(a22), _23(a23),
      _31(a31), _32(a32), _33(a33)
  {}
  explicit mat3(const mat3sym &s)
    : _11(s._11), _12(s._12), _13(s._13),
      _21(s._12), _22(s._22), _23(s._23),
      _31(s._13), _32(s._23), _33(s._33)
      {}*/
} mat3;


typedef struct
{
  float _1;
  float _2;
  float _3;
} vec3;


class VectorMath
{
private:
  VectorMath() = default;

public:
  static vec3 add(const vec3 &x, const vec3 &y);
  static vec3 abs(const vec3 &x);
  static vec3 subtract(const vec3 &x, const vec3 &y);
  static vec3 multiply(const float &a, const vec3 &x);
  static float length(const vec3 &);
  static float dot(const vec3 &, const vec3 &);
  static void reflect(const vec3 &, vec3 &);
  static float distance(const vec3 &, const vec3 &);

  static void calcInvariants(const mat3sym &, vec3 &);
  static void makeRealizable(const float &, mat3sym &);
  static float determinant(const mat3 &);
  static bool invert(mat3 &);
  static void multiply(const mat3 &, const vec3 &, vec3 &);
  static vec3 multiply(const mat3 &, const vec3 &);
};

inline vec3 VectorMath::add(const vec3 &x, const vec3 &y)
{
  return { x._1 + y._1, x._2 + y._2, x._3 + y._3 };
}

inline vec3 VectorMath::abs(const vec3 &x)
{
  return { std::abs(x._1), std::abs(x._2), std::abs(x._3) };
}

inline vec3 VectorMath::subtract(const vec3 &x, const vec3 &y)
{
  return { x._1 - y._1, x._2 - y._2, x._3 - y._3 };
}

inline vec3 VectorMath::multiply(const float &a, const vec3 &x)
{
  return { a * x._1, a * x._2, a * x._3 };
}

inline float VectorMath::length(const vec3 &x)
{
  return sqrt(x._1 * x._1 + x._2 * x._2 + x._3 * x._3);
}

inline float VectorMath::dot(const vec3 &a, const vec3 &b)
{
  return a._1 * b._1 + a._2 * b._2 + a._3 * b._3;
}

inline void VectorMath::reflect(const vec3 &n, vec3 &v)
{
  float s = dot(n, v);
  v._1 = v._1 - 2.0f * s * n._1;
  v._2 = v._2 - 2.0f * s * n._2;
  v._3 = v._3 - 2.0f * s * n._3;
}

inline float VectorMath::distance(const vec3 &a, const vec3 &b)
{
  return (sqrt((a._1 - b._1) * (a._1 - b._1)
               + (a._2 - b._2) * (a._2 - b._2)
               + (a._3 - b._3) * (a._3 - b._3)));
}

inline void VectorMath::calcInvariants(const mat3sym &tau, vec3 &invar)
{
  // since the x doesn't depend on itself, can just set the output without doing
  // any temporary variables (copied from Bailey's code)
  invar._1 = tau._11 + tau._22 + tau._33;
  invar._2 = tau._11 * tau._22 + tau._11 * tau._33 + tau._22 * tau._33
             - (tau._12 * tau._12 + tau._13 * tau._13 + tau._23 * tau._23);
  invar._3 = tau._11 * (tau._22 * tau._33 - tau._23 * tau._23)
             - tau._12 * (tau._12 * tau._33 - tau._23 * tau._13)
             + tau._13 * (tau._12 * tau._23 - tau._22 * tau._13);
}

inline void VectorMath::makeRealizable(const float &invarianceTol, mat3sym &tau)
{
  // first calculate the invariants and see if they are already realizable
  vec3 invar = { 0.0, 0.0, 0.0 };

  calcInvariants(tau, invar);

  if (invar._1 > invarianceTol && invar._2 > invarianceTol && invar._3 > invarianceTol) {
    return;// tau is already realizable
  }

  // make it realizeable
  // start by making a guess of ks, the subfilter scale tke
  float b = 4.0f / 3.0f * invar._1;
  float c = invar._2;
  float ks = 1.01f * (-b + std::sqrt(b * b - 16.0f / 3.0f * c)) / (8.0f / 3.0f);

  // if the initial guess is bad, use the straight up invar_xx value
  if (ks < invarianceTol || std::isnan(ks)) {
    ks = 0.5f * std::abs(invar._1);// also 0.5*abs(invar_xx)
  }

  // to avoid increasing tau by more than ks increasing by 0.05%, use a separate
  // stress tensor and always increase the separate stress tensor using the
  // original stress tensor, only changing ks for each iteration notice that
  // through all this process, only the diagonals are really increased by a
  // value of 0.05% of the subfilter tke ks start by initializing the separate
  // stress tensor
  mat3sym tau_new = { tau._11 + 2.0f / 3.0f * ks,
                      tau._12,
                      tau._13,
                      tau._22 + 2.0f / 3.0f * ks,
                      tau._23,
                      tau._33 + 2.0f / 3.0f * ks };

  /*tau_new._11 = tau._11 + 2.0 / 3.0 * ks;
  tau_new._12 = tau._12;
  tau_new._13 = tau._13;
  tau_new._22 = tau._22 + 2.0 / 3.0 * ks;
  tau_new._23 = tau._23;
  tau_new._33 = tau._33 + 2.0 / 3.0 * ks;*/

  calcInvariants(tau_new, invar);

  // now adjust the diagonals by 0.05% of the subfilter tke, which is ks, till
  // tau is realizable or if too many iterations go on, give a warning. I've had
  // trouble with this taking too long
  //  if it isn't realizable, so maybe another approach for when the iterations
  //  are reached might be smart
  int iter = 0;
  while ((invar._1 < invarianceTol || invar._2 < invarianceTol || invar._3 < invarianceTol) && iter < 1000) {
    iter = iter + 1;

    // increase subfilter tke by 5%
    ks = ks * 1.050;

    // note that the right hand side is not tau_new, to force tau to only
    // increase by increasing ks
    tau_new._11 = tau._11 + 2.0f / 3.0f * ks;
    tau_new._22 = tau._22 + 2.0f / 3.0f * ks;
    tau_new._33 = tau._33 + 2.0f / 3.0f * ks;

    calcInvariants(tau_new, invar);
  }

  if (iter == 999) {
    std::cout << "WARNING (Plume::makeRealizable): unable to make stress tensor realizable." << std::endl;
  }

  // now set the output actual stress tensor using the separate temporary stress
  // tensor
  tau = tau_new;
}

inline float VectorMath::determinant(const mat3 &A)
{
  // calculate the determinant
  return A._11 * (A._22 * A._33 - A._23 * A._32)
         - A._12 * (A._21 * A._33 - A._23 * A._31)
         + A._13 * (A._21 * A._32 - A._22 * A._31);
}

inline bool VectorMath::invert(mat3 &A)
{

  // calculate the determinant
  float det = determinant(A);

  // check for near zero value determinants
  if (std::abs(det) < 1e-10) {
    det = 10e10;
    A = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    return false;
  } else {

    // calculate the inverse (cannot be done implace)
    mat3 A_inv = { (A._22 * A._33 - A._23 * A._32) / det,
                   -(A._12 * A._33 - A._13 * A._32) / det,
                   (A._12 * A._23 - A._22 * A._13) / det,
                   -(A._21 * A._33 - A._23 * A._31) / det,
                   (A._11 * A._33 - A._13 * A._31) / det,
                   -(A._11 * A._23 - A._13 * A._21) / det,
                   (A._21 * A._32 - A._31 * A._22) / det,
                   -(A._11 * A._32 - A._12 * A._31) / det,
                   (A._11 * A._22 - A._12 * A._21) / det };

    /*mat3 Ainv;
    Ainv._11 = (A._22 * A._33 - A._23 * A._32) / det,
    Ainv._12 = -(A._12 * A._33 - A._13 * A._32) / det;
    Ainv._13 = (A._12 * A._23 - A._22 * A._13) / det;
    Ainv._21 = -(A._21 * A._33 - A._23 * A._31) / det;
    Ainv._22 = (A._11 * A._33 - A._13 * A._31) / det;
    Ainv._23 = -(A._11 * A._23 - A._13 * A._21) / det;
    Ainv._31 = (A._21 * A._32 - A._31 * A._22) / det;
    Ainv._32 = -(A._11 * A._32 - A._12 * A._31) / det;
    Ainv._33 = (A._11 * A._22 - A._12 * A._21) / det;*/

    // set the input reference A matrix
    A = A_inv;

    return true;
  }
}

inline void VectorMath::multiply(const mat3 &A, const vec3 &b, vec3 &x)
{
  // now calculate the x=Ab
  x._1 = b._1 * A._11 + b._2 * A._12 + b._3 * A._13;
  x._2 = b._1 * A._21 + b._2 * A._22 + b._3 * A._23;
  x._3 = b._1 * A._31 + b._2 * A._32 + b._3 * A._33;
}

inline vec3 VectorMath::multiply(const mat3 &A, const vec3 &b)
{
  // now calculate the x=Ab
  return { b._1 * A._11 + b._2 * A._12 + b._3 * A._13,
           b._1 * A._21 + b._2 * A._22 + b._3 * A._23,
           b._1 * A._31 + b._2 * A._32 + b._3 * A._33 };
}
