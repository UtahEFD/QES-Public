/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/**
 * @file vectorMath.cu
 * @brief :document this:
 */

#include "vectorMath.h"

__device__ void calcInvariants(const mat3sym &tau, vec3 &invar)
{
  // since the x doesn't depend on itself, can just set the output without doing
  // any temporary variables (copied from Bailey's code)
  invar._1 = tau._11 + tau._22 + tau._33;
  invar._2 = tau._11 * tau._22 + tau._11 * tau._33 + tau._22 * tau._33
             - (tau._12 * tau._12 + tau._13 * tau._13 + tau._23 * tau._23);
  invar._3 = tau._11 * (tau._22 * tau._33 - tau._23 * tau._23)
             - tau._12 * (tau._23 * tau._33 - tau._23 * tau._13)
             + tau._13 * (tau._12 * tau._23 - tau._22 * tau._13);
}

__device__ void makeRealizable(float invarianceTol, mat3sym &tau)
{
  // first calculate the invariants and see if they are already realizable
  vec3 invar = { 0.0, 0.0, 0.0 };

  calcInvariants(tau, invar);

  if (invar._1 > invarianceTol && invar._2 > invarianceTol && invar._3 > invarianceTol) {
    return;// tau is already realizable
  }

  // make it realizeable
  // start by making a guess of ks, the subfilter scale tke
  float b = 4.0 / 3.0 * invar._1;
  float c = invar._2;
  float ks = 1.01 * (-b + std::sqrt(b * b - 16.0 / 3.0 * c)) / (8.0 / 3.0);

  // if the initial guess is bad, use the straight up invar_xx value
  if (ks < invarianceTol || isnan(ks)) {
    ks = 0.5 * std::abs(invar._1);// also 0.5*abs(invar_xx)
  }

  // to avoid increasing tau by more than ks increasing by 0.05%, use a separate
  // stress tensor and always increase the separate stress tensor using the
  // original stress tensor, only changing ks for each iteration notice that
  // through all this process, only the diagonals are really increased by a
  // value of 0.05% of the subfilter tke ks start by initializing the separate
  // stress tensor
  mat3sym tau_new;
  tau_new._11 = tau._11 + 2.0 / 3.0 * ks;
  tau_new._12 = tau._12;
  tau_new._13 = tau._13;
  tau_new._22 = tau._22 + 2.0 / 3.0 * ks;
  tau_new._23 = tau._23;
  tau_new._33 = tau._33 + 2.0 / 3.0 * ks;

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
    tau_new._11 = tau._11 + 2.0 / 3.0 * ks;
    tau_new._22 = tau._22 + 2.0 / 3.0 * ks;
    tau_new._33 = tau._33 + 2.0 / 3.0 * ks;

    calcInvariants(tau_new, invar);
  }

  if (iter == 999) {
    //std::cout << "WARNING (Plume::makeRealizable): unable to make stress "
    //             "tensor realizble.";
  }

  // now set the output actual stress tensor using the separate temporary stress
  // tensor
  tau = tau_new;
}


__device__ bool invert3(mat3 &A)
{

  // calculate the determinant
  float det = A._11 * (A._22 * A._33 - A._23 * A._32)
              - A._12 * (A._21 * A._33 - A._23 * A._31)
              + A._13 * (A._21 * A._32 - A._22 * A._31);

  // check for near zero value determinants
  if (std::abs(det) < 1e-6) {
    det = 10e10;
    A = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    return false;
  } else {

    // calculate the inverse (cannot be done implace)
    mat3 Ainv;
    Ainv._11 = (A._22 * A._33 - A._23 * A._32) / det;
    Ainv._12 = -(A._12 * A._33 - A._13 * A._32) / det;
    Ainv._13 = (A._12 * A._23 - A._22 * A._13) / det;
    Ainv._21 = -(A._21 * A._33 - A._23 * A._31) / det;
    Ainv._22 = (A._11 * A._33 - A._13 * A._31) / det;
    Ainv._23 = -(A._11 * A._23 - A._13 * A._21) / det;
    Ainv._31 = (A._21 * A._32 - A._31 * A._22) / det;
    Ainv._32 = -(A._11 * A._32 - A._12 * A._31) / det;
    Ainv._33 = (A._11 * A._22 - A._12 * A._21) / det;

    // set the input reference A matrix
    A = Ainv;

    return true;
  }
}

__device__ void matmult(const mat3 &A, const vec3 &b, vec3 &x)
{
  // now calculate the Ax=b x value from the input inverse A matrix and b matrix
  x._1 = b._1 * A._11 + b._2 * A._12 + b._3 * A._13;
  x._2 = b._1 * A._21 + b._2 * A._22 + b._3 * A._23;
  x._3 = b._1 * A._31 + b._2 * A._32 + b._3 * A._33;
}
