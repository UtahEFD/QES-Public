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

#include "test_PlumeGeneralData.h"

typedef struct
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
} mat3;
typedef struct
{
  float _11;
  float _12;
  float _13;
  float _22;
  float _23;
  float _33;
} mat3sym;


typedef struct
{
  float _1;
  float _2;
  float _3;
} vec3;

__device__ void calcInvariants(const float &txx,
                               const float &txy,
                               const float &txz,
                               const float &tyy,
                               const float &tyz,
                               const float &tzz,
                               float &invar_xx,
                               float &invar_yy,
                               float &invar_zz)
{
  // since the x doesn't depend on itself, can just set the output without doing
  // any temporary variables (copied from Bailey's code)
  invar_xx = txx + tyy + tzz;
  invar_yy = txx * tyy + txx * tzz + tyy * tzz - txy * txy - txz * txz - tyz * tyz;
  invar_zz = txx * (tyy * tzz - tyz * tyz) - txy * (txy * tzz - tyz * txz) + txz * (txy * tyz - tyy * txz);
}

__device__ void makeRealizable(float &txx,
                               float &txy,
                               float &txz,
                               float &tyy,
                               float &tyz,
                               float &tzz)
{
  // first calculate the invariants and see if they are already realizable
  // the calcInvariants function modifies the values directly, so they always
  // need initialized to something before being sent into said function to be
  // calculated
  float invarianceTol = 1.0e-4;

  float invar_xx = 0.0;
  float invar_yy = 0.0;
  float invar_zz = 0.0;
  calcInvariants(txx, txy, txz, tyy, tyz, tzz, invar_xx, invar_yy, invar_zz);

  if (invar_xx > invarianceTol && invar_yy > invarianceTol && invar_zz > invarianceTol) {
    return;// tau is already realizable
  }

  // since tau is not already realizable, need to make it realizeable
  // start by making a guess of ks, the subfilter scale tke
  // I keep wondering if we can use the input Turb->tke for this or if we should
  // leave it as is
  float b = 4.0 / 3.0 * (txx + tyy + tzz);// also 4.0/3.0*invar_xx
  float c = txx * tyy + txx * tzz + tyy * tzz - txy * txy - txz * txz - tyz * tyz;// also invar_yy
  float ks = 1.01 * (-b + std::sqrt(b * b - 16.0 / 3.0 * c)) / (8.0 / 3.0);

  // if the initial guess is bad, use the straight up invar_xx value
  if (ks < invarianceTol || isnan(ks)) {
    ks = 0.5 * std::abs(txx + tyy + tzz);// also 0.5*abs(invar_xx)
  }

  // to avoid increasing tau by more than ks increasing by 0.05%, use a separate
  // stress tensor and always increase the separate stress tensor using the
  // original stress tensor, only changing ks for each iteration notice that
  // through all this process, only the diagonals are really increased by a
  // value of 0.05% of the subfilter tke ks start by initializing the separate
  // stress tensor
  float txx_new = txx + 2.0 / 3.0 * ks;
  float txy_new = txy;
  float txz_new = txz;
  float tyy_new = tyy + 2.0 / 3.0 * ks;
  float tyz_new = tyz;
  float tzz_new = tzz + 2.0 / 3.0 * ks;

  calcInvariants(txx_new, txy_new, txz_new, tyy_new, tyz_new, tzz_new, invar_xx, invar_yy, invar_zz);

  // now adjust the diagonals by 0.05% of the subfilter tke, which is ks, till
  // tau is realizable or if too many iterations go on, give a warning. I've had
  // trouble with this taking too long
  //  if it isn't realizable, so maybe another approach for when the iterations
  //  are reached might be smart
  int iter = 0;
  while ((invar_xx < invarianceTol || invar_yy < invarianceTol || invar_zz < invarianceTol) && iter < 1000) {
    iter = iter + 1;

    // increase subfilter tke by 5%
    ks = ks * 1.050;

    // note that the right hand side is not tau_new, to force tau to only
    // increase by increasing ks
    txx_new = txx + 2.0 / 3.0 * ks;
    tyy_new = tyy + 2.0 / 3.0 * ks;
    tzz_new = tzz + 2.0 / 3.0 * ks;

    calcInvariants(txx_new, txy_new, txz_new, tyy_new, tyz_new, tzz_new, invar_xx, invar_yy, invar_zz);
  }

  if (iter == 999) {
    // std::cout << "WARNING (Plume::makeRealizable): unable to make stress "
    //              "tensor realizble.";
  }

  // now set the output actual stress tensor using the separate temporary stress
  // tensor
  txx = txx_new;
  txy = txy_new;
  txz = txz_new;
  tyy = tyy_new;
  tyz = tyz_new;
  tzz = tzz_new;
}

__device__ bool invert3(float &A_11,
                        float &A_12,
                        float &A_13,
                        float &A_21,
                        float &A_22,
                        float &A_23,
                        float &A_31,
                        float &A_32,
                        float &A_33)
{
  // note that with Bailey's code, the input A_21, A_31, and A_32 are zeros even
  // though they are used here at least when using this on tau to calculate the
  // inverse stress tensor. This is not true when calculating the inverse A
  // matrix for the Ax=b calculation

  // now calculate the determinant
  float det = A_11 * (A_22 * A_33 - A_23 * A_32) - A_12 * (A_21 * A_33 - A_23 * A_31) + A_13 * (A_21 * A_32 - A_22 * A_31);

  // check for near zero value determinants
  // LA future work: I'm still debating whether this warning needs to be limited
  // by the updateFrequency information
  //  if so, how would we go about limiting that info? Would probably need to
  //  make the loop counter variables actual data members of the class
  if (std::abs(det) < 1e-10) {
    // std::cout << "WARNING (Plume::invert3): matrix nearly singular" <<
    // std::endl; std::cout << "abs(det) = \"" << std::abs(det) << "\",  A_11 =
    // \"" << A_11 << "\", A_12 = \"" << A_12 << "\", A_13 = \""
    //          << A_13 << "\", A_21 = \"" << A_21 << "\", A_22 = \"" << A_22 <<
    //          "\", A_23 = \"" << A_23 << "\", A_31 = \""
    //          << A_31 << "\" A_32 = \"" << A_32 << "\", A_33 = \"" << A_33 <<
    //          "\"" << std::endl;

    det = 10e10;
    A_11 = 0.0;
    A_12 = 0.0;
    A_13 = 0.0;
    A_21 = 0.0;
    A_22 = 0.0;
    A_23 = 0.0;
    A_31 = 0.0;
    A_32 = 0.0;
    A_33 = 0.0;

    return false;

  } else {

    // calculate the inverse. Because the inverted matrix depends on other
    // components of the matrix,
    //  need to make a temporary value till all the inverted parts of the matrix
    //  are set
    float Ainv_11 = (A_22 * A_33 - A_23 * A_32) / det;
    float Ainv_12 = -(A_12 * A_33 - A_13 * A_32) / det;
    float Ainv_13 = (A_12 * A_23 - A_22 * A_13) / det;
    float Ainv_21 = -(A_21 * A_33 - A_23 * A_31) / det;
    float Ainv_22 = (A_11 * A_33 - A_13 * A_31) / det;
    float Ainv_23 = -(A_11 * A_23 - A_13 * A_21) / det;
    float Ainv_31 = (A_21 * A_32 - A_31 * A_22) / det;
    float Ainv_32 = -(A_11 * A_32 - A_12 * A_31) / det;
    float Ainv_33 = (A_11 * A_22 - A_12 * A_21) / det;

    // now set the input reference A matrix to the temporary inverted A matrix
    // values
    A_11 = Ainv_11;
    A_12 = Ainv_12;
    A_13 = Ainv_13;
    A_21 = Ainv_21;
    A_22 = Ainv_22;
    A_23 = Ainv_23;
    A_31 = Ainv_31;
    A_32 = Ainv_32;
    A_33 = Ainv_33;

    return true;
  }
}

__device__ void matmult(const float &A_11, const float &A_12, const float &A_13, const float &A_21, const float &A_22, const float &A_23, const float &A_31, const float &A_32, const float &A_33, const float &b_11, const float &b_21, const float &b_31, float &x_11, float &x_21, float &x_31)
{
  // now calculate the Ax=b x value from the input inverse A matrix and b matrix
  x_11 = b_11 * A_11 + b_21 * A_12 + b_31 * A_13;
  x_21 = b_11 * A_21 + b_21 * A_22 + b_31 * A_23;
  x_31 = b_11 * A_31 + b_21 * A_32 + b_31 * A_33;
}


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
    // std::cout << "WARNING (Plume::makeRealizable): unable to make stress "
    //              "tensor realizble.";
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


__global__ void testCUDA(int length,
                         float *d_A11,
                         float *d_A12,
                         float *d_A13,
                         float *d_A21,
                         float *d_A22,
                         float *d_A23,
                         float *d_A31,
                         float *d_A32,
                         float *d_A33,
                         float *d_b1,
                         float *d_b2,
                         float *d_b3,
                         float *d_x1,
                         float *d_x2,
                         float *d_x3)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < length; it += stride) {
    bool tt = invert3(d_A11[it],
                      d_A12[it],
                      d_A13[it],
                      d_A21[it],
                      d_A22[it],
                      d_A23[it],
                      d_A31[it],
                      d_A32[it],
                      d_A33[it]);
    matmult(d_A11[it],
            d_A12[it],
            d_A13[it],
            d_A21[it],
            d_A22[it],
            d_A23[it],
            d_A31[it],
            d_A32[it],
            d_A33[it],
            d_b1[it],
            d_b2[it],
            d_b3[it],
            d_x1[it],
            d_x2[it],
            d_x3[it]);
  }
  return;
}
__global__ void testCUDA_matmult(int length, mat3 *d_A, vec3 *d_b, vec3 *d_x)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < length; it += stride) {
    bool tt = invert3(d_A[it]);
    matmult(d_A[it], d_b[it], d_x[it]);
  }
  return;
}

__global__ void testCUDA_invar(int length, mat3sym *d_tau, vec3 *d_invar)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int it = index; it < length; it += stride) {
    makeRealizable(10e-4, d_tau[it]);
    calcInvariants(d_tau[it], d_invar[it]);
  }
  return;
}

void test_PlumeGeneralData::testGPU(int length)
{

  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);

  int blockCount = 1;
  cudaDeviceGetAttribute(&blockCount, cudaDevAttrMultiProcessorCount, gpuID);
  std::cout << blockCount << std::endl;

  int threadsPerBlock = 32;
  cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuID);
  std::cout << threadsPerBlock << std::endl;

  int blockSize = 1024;
  dim3 numberOfThreadsPerBlock(blockSize, 1, 1);
  dim3 numberOfBlocks(ceil(length / (float)(blockSize)), 1, 1);

  std::vector<float> A11, A12, A13, A21, A22, A23, A31, A32, A33;
  A11.resize(length, 1.0);
  A12.resize(length, 2.0);
  A13.resize(length, 3.0);
  A21.resize(length, 2.0);
  A22.resize(length, 1.0);
  A23.resize(length, 2.0);
  A31.resize(length, 3.0);
  A32.resize(length, 2.0);
  A33.resize(length, 1.0);

  std::vector<float> b1, b2, b3;
  b1.resize(length, 1.0);
  b2.resize(length, 1.0);
  b3.resize(length, 1.0);

  std::vector<float> x1, x2, x3;
  x1.resize(length, 0.0);
  x2.resize(length, 0.0);
  x3.resize(length, 0.0);

  if (errorCheck == cudaSuccess) {
    // temp

    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    float *d_A11, *d_A12, *d_A13, *d_A21, *d_A22, *d_A23, *d_A31, *d_A32, *d_A33;

    cudaMalloc((void **)&d_A11, length * sizeof(float));
    cudaMalloc((void **)&d_A12, length * sizeof(float));
    cudaMalloc((void **)&d_A13, length * sizeof(float));
    cudaMalloc((void **)&d_A21, length * sizeof(float));
    cudaMalloc((void **)&d_A22, length * sizeof(float));
    cudaMalloc((void **)&d_A23, length * sizeof(float));
    cudaMalloc((void **)&d_A31, length * sizeof(float));
    cudaMalloc((void **)&d_A32, length * sizeof(float));
    cudaMalloc((void **)&d_A33, length * sizeof(float));

    float *d_b1, *d_b2, *d_b3;
    cudaMalloc((void **)&d_b1, length * sizeof(float));
    cudaMalloc((void **)&d_b2, length * sizeof(float));
    cudaMalloc((void **)&d_b3, length * sizeof(float));

    float *d_x1, *d_x2, *d_x3;
    cudaMalloc((void **)&d_x1, length * sizeof(float));
    cudaMalloc((void **)&d_x2, length * sizeof(float));
    cudaMalloc((void **)&d_x3, length * sizeof(float));

    cudaMemcpy(d_A11, A11.data(), length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A12, A12.data(), length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A13, A13.data(), length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A21, A21.data(), length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A22, A22.data(), length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A23, A23.data(), length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A31, A31.data(), length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A32, A32.data(), length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A33, A33.data(), length * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_b1, b1.data(), length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2.data(), length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, b3.data(), length * sizeof(float), cudaMemcpyHostToDevice);

    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    testCUDA<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length,
                                                          d_A11,
                                                          d_A12,
                                                          d_A13,
                                                          d_A21,
                                                          d_A22,
                                                          d_A23,
                                                          d_A31,
                                                          d_A32,
                                                          d_A33,
                                                          d_b1,
                                                          d_b2,
                                                          d_b3,
                                                          d_x1,
                                                          d_x2,
                                                          d_x3);
    cudaDeviceSynchronize();
    auto kernelEndTime = std::chrono::high_resolution_clock::now();


    // cudamemcpy back to host
    /*
      cudaMemcpy(A11.data(), d_A11, length * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(A12.data(), d_A12, length * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(A13.data(), d_A13, length * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(A21.data(), d_A21, length * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(A22.data(), d_A22, length * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(A23.data(), d_A23, length * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(A31.data(), d_A31, length * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(A32.data(), d_A32, length * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(A33.data(), d_A33, length * sizeof(float), cudaMemcpyDeviceToHost);
    */
    cudaMemcpy(x1.data(), d_x1, length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x2.data(), d_x2, length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x3.data(), d_x3, length * sizeof(float), cudaMemcpyDeviceToHost);

    // cudafree
    cudaFree(d_A11);
    cudaFree(d_A12);
    cudaFree(d_A13);
    cudaFree(d_A21);
    cudaFree(d_A22);
    cudaFree(d_A23);
    cudaFree(d_A31);
    cudaFree(d_A32);
    cudaFree(d_A33);

    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_b3);

    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_x3);

    auto gpuEndTime = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "\t kernel  elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "\t GPU  elapsed time: " << gpuElapsed.count() << " s\n";
    /*
    std::cout << A11[0] << " " << A12[0] << " " << A13[0] << std::endl;
    std::cout << A21[0] << " " << A22[0] << " " << A23[0] << std::endl;
    std::cout << A31[0] << " " << A32[0] << " " << A33[0] << std::endl;
    */
    std::cout << x1[0] << " " << x2[0] << " " << x3[0] << std::endl;

  } else {
    printf("CUDA ERROR!\n");
  }
}


void test_PlumeGeneralData::testGPU_struct(int length)
{

  int gpuID = 0;
  cudaError_t errorCheck = cudaGetDevice(&gpuID);

  int blockCount = 1;
  cudaDeviceGetAttribute(&blockCount, cudaDevAttrMultiProcessorCount, gpuID);
  // std::cout << blockCount << std::endl;

  int threadsPerBlock = 32;
  cudaDeviceGetAttribute(&threadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuID);
  // std::cout << threadsPerBlock << std::endl;

  int blockSize = 1024;
  dim3 numberOfThreadsPerBlock(blockSize, 1, 1);
  dim3 numberOfBlocks(ceil(length / (float)(blockSize)), 1, 1);

  mat3 tmp = { 1, 2, 3, 2, 1, 2, 3, 2, 1 };
  std::vector<mat3> A;
  A.resize(length, tmp);

  std::vector<vec3> b;
  b.resize(length, { 1.0, 1.0, 1.0 });

  std::vector<vec3> x;
  x.resize(length, { 0.0, 0.0, 0.0 });

  std::vector<mat3sym> tau;
  // tau.resize(length, { 1, 2, 3, 1, 2, 1 });
  tau.resize(length, { 1, 0, 3, 0, 0, 1 });
  std::vector<vec3> invar;
  invar.resize(length, { 0.0, 0.0, 0.0 });

  if (errorCheck == cudaSuccess) {
    // temp

    auto gpuStartTime = std::chrono::high_resolution_clock::now();

    mat3 *d_A;
    cudaMalloc((void **)&d_A, 9 * length * sizeof(float));
    vec3 *d_b;
    cudaMalloc((void **)&d_b, 3 * length * sizeof(float));
    vec3 *d_x;
    cudaMalloc((void **)&d_x, length * sizeof(vec3));

    cudaMemcpy(d_A, A.data(), 9 * length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), 3 * length * sizeof(float), cudaMemcpyHostToDevice);


    mat3sym *d_tau;
    cudaMalloc((void **)&d_tau, length * sizeof(mat3sym));
    vec3 *d_invar;
    cudaMalloc((void **)&d_invar, length * sizeof(vec3));

    cudaMemcpy(d_tau, tau.data(), length * sizeof(mat3sym), cudaMemcpyHostToDevice);


    // call kernel
    auto kernelStartTime = std::chrono::high_resolution_clock::now();
    testCUDA_matmult<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_A, d_b, d_x);
    testCUDA_invar<<<numberOfBlocks, numberOfThreadsPerBlock>>>(length, d_tau, d_invar);
    cudaDeviceSynchronize();
    auto kernelEndTime = std::chrono::high_resolution_clock::now();


    // cudamemcpy back to host
    cudaMemcpy(A.data(), d_A, 9 * length * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x.data(), d_x, 3 * length * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(tau.data(), d_tau, length * sizeof(mat3sym), cudaMemcpyDeviceToHost);
    cudaMemcpy(invar.data(), d_invar, length * sizeof(vec3), cudaMemcpyDeviceToHost);


    // cudafree
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);

    cudaFree(d_tau);
    cudaFree(d_invar);

    auto gpuEndTime = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double> kernelElapsed = kernelEndTime - kernelStartTime;
    std::cout << "\t kernel  elapsed time: " << kernelElapsed.count() << " s\n";
    std::chrono::duration<double> gpuElapsed = gpuEndTime - gpuStartTime;
    std::cout << "\t GPU  elapsed time: " << gpuElapsed.count() << " s\n";

    std::cout << A[0]._11 << " " << A[0]._12 << " " << A[0]._13 << std::endl;
    std::cout << A[0]._21 << " " << A[0]._22 << " " << A[0]._23 << std::endl;
    std::cout << A[0]._31 << " " << A[0]._32 << " " << A[0]._33 << std::endl;
    std::cout << x[0]._1 << " " << x[0]._2 << " " << x[0]._3 << std::endl;

    std::cout << std::endl;
    std::cout << tau[0]._11 << " " << tau[0]._12 << " " << tau[0]._13 << std::endl;
    std::cout << tau[0]._12 << " " << tau[0]._22 << " " << tau[0]._23 << std::endl;
    std::cout << tau[0]._13 << " " << tau[0]._23 << " " << tau[0]._33 << std::endl;
    std::cout << invar[0]._1 << " " << invar[0]._2 << " " << invar[0]._3 << std::endl;


  } else {
    printf("CUDA ERROR!\n");
  }
}
