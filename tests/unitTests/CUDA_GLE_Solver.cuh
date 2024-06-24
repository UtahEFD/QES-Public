/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
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

/** @file GLE_Solver.cpp
 * @brief
 */

#pragma once

#include "util/VectorMath.h"
#include "util/VectorMath_CUDA.cuh"

// #include "winds/TURBGeneralData.h"

#include "Particle.cuh"
// #include "Interp.h"
// #include "PLUMEGeneralData.h"

// need to add information coming form:  TURBGeneralData *TGD, PLUMEGeneralData *PGD
__device__ void solve(particle_AOS *p, int tid, float par_dt, float invarianceTol, float vel_threshold)
{

  // float txx = 0.0, txy = 0.0, txz = 0.0, tyy = 0.0, tyz = 0.0, tzz = 0.0;
  // float flux_div_x = 0.0, flux_div_y = 0.0, flux_div_z = 0.0;

  /*PGD[tid].interp[tid].interpValues(TGD,
                            p[tid].xPos,
                            p[tid].yPos,
                            p[tid].zPos,
                            txx,
                            txy,
                            txz,
                            tyy,
                            tyz,
                            tzz,
                            flux_div_x,
                            flux_div_y,
                            flux_div_z,
                            p[tid].nuT,
                            p[tid].CoEps);*/

  // mat3sym tau = { txx, txy, txz, tyy, tyz, tzz };
  // vec3 flux_div = { flux_div_x, flux_div_y, flux_div_z };

  // mat3sym tau = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  // vec3 flux_div = { 0.0, 0.0, 0.0 };

  // PGD[tid].interp[tid].interpValues(TGD,p[tid].pos,tau,flux_div,p[tid].nuT,p[tid].CoEps);

  // now need to call makeRealizable on tau
  makeRealizable(invarianceTol, p[tid].tau);
  // mat3sym tau = p[tid].tau;
  // makeRealizable(invarianceTol, tau);

  // now need to calculate the inverse values for tau
  // directly modifies the values of tau
  mat3 L = { p[tid].tau._11, p[tid].tau._12, p[tid].tau._13, p[tid].tau._12, p[tid].tau._22, p[tid].tau._23, p[tid].tau._13, p[tid].tau._23, p[tid].tau._33 };
  // mat3 L = { tau._11, tau._12, tau._13, tau._12, tau._22, tau._23, tau._13, tau._23, tau._33 };

  p[tid].isRogue = !invert(L);
  if (p[tid].isRogue) {
    // int cellIdNew = interp[tid].getCellId(xPos,yPos,zPos);
    // std::cerr << "ERROR in Matrix inversion of stress tensor" << std::endl;
    p[tid].isActive = false;
    return;
  }

  // these are the random numbers for each direction
  /*
  vec3 vRandn = { PGD[tid].threadRNG[omp_get_thread_num()][tid].norRan(),
                  PGD[tid].threadRNG[omp_get_thread_num()][tid].norRan(),
                  PGD[tid].threadRNG[omp_get_thread_num()][tid].norRan() };
  */
  vec3 vRandn = { 0.1f, 0.1f, 0.1f };

  // now calculate a bunch of values for the current particle
  // calculate the time derivative of the stress tensor: (tau_current - tau_old)/dt

  /*
  mat3sym tau_ddt = { (tau._11 - p[tid].tau_old._11) / par_dt,
                      (tau._12 - p[tid].tau_old._12) / par_dt,
                      (tau._13 - p[tid].tau_old._13) / par_dt,
                      (tau._22 - p[tid].tau_old._22) / par_dt,
                      (tau._23 - p[tid].tau_old._23) / par_dt,
                      (tau._33 - p[tid].tau_old._33) / par_dt };
  */
  mat3sym tau_ddt = { (p[tid].tau._11 - p[tid].tau_old._11) / par_dt,
                      (p[tid].tau._12 - p[tid].tau_old._12) / par_dt,
                      (p[tid].tau._13 - p[tid].tau_old._13) / par_dt,
                      (p[tid].tau._22 - p[tid].tau_old._22) / par_dt,
                      (p[tid].tau._23 - p[tid].tau_old._23) / par_dt,
                      (p[tid].tau._33 - p[tid].tau_old._33) / par_dt };
  /*
    mat3sym tau_ddt = { (tau._11 - p[tid].tau._11) / par_dt,
                    (tau._12 - p[tid].tau._12) / par_dt,
                    (tau._13 - p[tid].tau._13) / par_dt,
                    (tau._22 - p[tid].tau._22) / par_dt,
                    (tau._23 - p[tid].tau._23) / par_dt,
                    (tau._33 - p[tid].tau._33) / par_dt };
  */

  // now calculate and set the A and b matrices for an Ax = b
  // A = -I + 0.5*(-CoEps*L + dTdt*L )*par_dt;
  mat3 A = { -1.0f + 0.50f * (-p[tid].CoEps * L._11 + L._11 * tau_ddt._11 + L._12 * tau_ddt._12 + L._13 * tau_ddt._13) * par_dt,
             -0.0f + 0.50f * (-p[tid].CoEps * L._12 + L._12 * tau_ddt._11 + L._22 * tau_ddt._12 + L._23 * tau_ddt._13) * par_dt,
             -0.0f + 0.50f * (-p[tid].CoEps * L._13 + L._13 * tau_ddt._11 + L._23 * tau_ddt._12 + L._33 * tau_ddt._13) * par_dt,
             -0.0f + 0.50f * (-p[tid].CoEps * L._12 + L._11 * tau_ddt._12 + L._12 * tau_ddt._22 + L._13 * tau_ddt._23) * par_dt,
             -1.0f + 0.50f * (-p[tid].CoEps * L._22 + L._12 * tau_ddt._12 + L._22 * tau_ddt._22 + L._23 * tau_ddt._23) * par_dt,
             -0.0f + 0.50f * (-p[tid].CoEps * L._23 + L._13 * tau_ddt._12 + L._23 * tau_ddt._22 + L._33 * tau_ddt._23) * par_dt,
             -0.0f + 0.50f * (-p[tid].CoEps * L._13 + L._11 * tau_ddt._13 + L._12 * tau_ddt._23 + L._13 * tau_ddt._33) * par_dt,
             -0.0f + 0.50f * (-p[tid].CoEps * L._23 + L._12 * tau_ddt._13 + L._22 * tau_ddt._23 + L._23 * tau_ddt._33) * par_dt,
             -1.0f + 0.50f * (-p[tid].CoEps * L._33 + L._13 * tau_ddt._13 + L._23 * tau_ddt._23 + L._33 * tau_ddt._33) * par_dt };

  /*double A_11 = -1.0 + 0.50 * (-p[tid].CoEps * lxx + lxx * dtxxdt + lxy * dtxydt + lxz * dtxzdt) * par_dt;
  double A_12 = 0.50 * (-p[tid].CoEps * lxy + lxy * dtxxdt + lyy * dtxydt + lyz * dtxzdt) * par_dt;
  double A_13 = 0.50 * (-p[tid].CoEps * lxz + lxz * dtxxdt + lyz * dtxydt + lzz * dtxzdt) * par_dt;

  double A_21 = 0.50 * (-p[tid].CoEps * lxy + lxx * dtxydt + lxy * dtyydt + lxz * dtyzdt) * par_dt;
  double A_22 = -1.0 + 0.50 * (-p[tid].CoEps * lyy + lxy * dtxydt + lyy * dtyydt + lyz * dtyzdt) * par_dt;
  double A_23 = 0.50 * (-p[tid].CoEps * lyz + lxz * dtxydt + lyz * dtyydt + lzz * dtyzdt) * par_dt;

  double A_31 = 0.50 * (-p[tid].CoEps * lxz + lxx * dtxzdt + lxy * dtyzdt + lxz * dtzzdt) * par_dt;
  double A_32 = 0.50 * (-p[tid].CoEps * lyz + lxy * dtxzdt + lyy * dtyzdt + lyz * dtzzdt) * par_dt;
  double A_33 = -1.0 + 0.50 * (-p[tid].CoEps * lzz + lxz * dtxzdt + lyz * dtyzdt + lzz * dtzzdt) * par_dt;*/

  // b = -vectFluct - 0.5*vecFluxDiv*par_dt - sqrt(CoEps*par_dt)*vecRandn;
  vec3 b = { -p[tid].velFluct_old._1 - 0.50f * p[tid].fluxDiv._1 * par_dt - std::sqrt(p[tid].CoEps * par_dt) * vRandn._1,
             -p[tid].velFluct_old._2 - 0.50f * p[tid].fluxDiv._2 * par_dt - std::sqrt(p[tid].CoEps * par_dt) * vRandn._2,
             -p[tid].velFluct_old._3 - 0.50f * p[tid].fluxDiv._3 * par_dt - std::sqrt(p[tid].CoEps * par_dt) * vRandn._3 };

  /*double b_11 = -p[tid].uFluct_old - 0.50 * flux_div_x * par_dt - std::sqrt(p[tid].CoEps * par_dt) * xRandn;
  double b_21 = -p[tid].vFluct_old - 0.50 * flux_div_y * par_dt - std::sqrt(p[tid].CoEps * par_dt) * yRandn;
  double b_31 = -p[tid].wFluct_old - 0.50 * flux_div_z * par_dt - std::sqrt(p[tid].CoEps * par_dt) * zRandn;*/

  // now prepare for the Ax=b calculation by calculating the inverted A matrix
  p[tid].isRogue = !invert(A);
  if (p[tid].isRogue) {
    // std::cerr << "ERROR in matrix inversion in Langevin equation" << std::endl;
    p[tid].isActive = false;
    return;
  }

  // now do the Ax=b calculation using the inverted matrix (vecFluct = A*b)
  multiply(A, b, p[tid].velFluct);

  // now check to see if the value is rogue or not
  if (std::abs(p[tid].velFluct._1) >= vel_threshold || isnan(p[tid].velFluct._1)) {
    // std::cerr << "Particle # " << p[tid].particleID << " is rogue, ";
    // std::cerr << "uFluct = " << p[tid].velFluct._1 << ", CoEps = " << p[tid].CoEps << std::endl;
    p[tid].velFluct._1 = 0.0;
    p[tid].isActive = false;
    p[tid].isRogue = true;
  }
  if (std::abs(p[tid].velFluct._2) >= vel_threshold || isnan(p[tid].velFluct._2)) {
    // std::cerr << "Particle # " << p[tid].particleID << " is rogue, ";
    // std::cerr << "vFluct = " << p[tid].velFluct._2 << ", CoEps = " << p[tid].CoEps << std::endl;
    p[tid].velFluct._2 = 0.0;
    p[tid].isActive = false;
    p[tid].isRogue = true;
  }
  if (std::abs(p[tid].velFluct._3) >= vel_threshold || isnan(p[tid].velFluct._3)) {
    // std::cerr << "Particle # " << p[tid].particleID << " is rogue, ";
    // std::cerr << "wFluct = " << p[tid].velFluct._3 << ", CoEps = " << p[tid].CoEps << std::endl;
    p[tid].velFluct._3 = 0.0;
    p[tid].isActive = false;
    p[tid].isRogue = true;
  }

  // now update the old values to be ready for the next particle time iteration
  // the current values are already set for the next iteration by the above calculations
  p[tid].tau_old = p[tid].tau;
  // p[tid].tau_old = tau;
  //  p[tid].tau = tau;
}

__device__ void solve(particle_SOA p, int tid, float par_dt, float invarianceTol, float vel_threshold)
{

  float CoEps = p.CoEps[tid];
  bool isActive;
  bool isRogue;

  // float txx = 0.0, txy = 0.0, txz = 0.0, tyy = 0.0, tyz = 0.0, tzz = 0.0;
  // float flux_div_x = 0.0, flux_div_y = 0.0, flux_div_z = 0.0;

  /*PGD[tid].interp[tid].interpValues(TGD,
                            p[tid].xPos,
                            p[tid].yPos,
                            p[tid].zPos,
                            txx,
                            txy,
                            txz,
                            tyy,
                            tyz,
                            tzz,
                            flux_div_x,
                            flux_div_y,
                            flux_div_z,
                            p[tid].nuT,
                            p[tid].CoEps);*/

  // mat3sym tau = { p.tau[tid]._11, p.tau[tid]._12, p.tau[tid]._13, p.tau[tid]._22, p.tau[tid]._23, p.tau[tid]._33 };
  //  vec3 flux_div = { flux_div_x, flux_div_y, flux_div_z };

  // vec3 flux_div = { 0.0, 0.0, 0.0 };

  // PGD[tid].interp[tid].interpValues(TGD,p[tid].pos,tau,flux_div,p[tid].nuT,p[tid].CoEps);

  // now need to call makeRealizable on tau
  // makeRealizable(invarianceTol, p[tid].tau);
  // mat3sym tau = { p.tau[tid]._11, p.tau[tid]._12, p.tau[tid]._13, p.tau[tid]._22, p.tau[tid]._23, p.tau[tid]._33 };
  // mat3sym tau = { 0.0, 0.0, 0.0, 0.0, 0.0 };
  // mat3sym tau = p.tau[tid];
  // makeRealizable(invarianceTol, tau);
  // p.tau[tid] = tau;

  makeRealizable(invarianceTol, p.tau[tid]);

  // now need to calculate the inverse values for tau
  // directly modifies the values of tau
  mat3 L = { p.tau[tid]._11, p.tau[tid]._12, p.tau[tid]._13, p.tau[tid]._12, p.tau[tid]._22, p.tau[tid]._23, p.tau[tid]._13, p.tau[tid]._23, p.tau[tid]._33 };
  // mat3 L = { tau._11, tau._12, tau._13, tau._12, tau._22, tau._23, tau._13, tau._23, tau._33 };

  isRogue = !invert(L);
  if (isRogue) {
    p.state[tid] = ROGUE;
    return;
  }
  /*if (isRogue) {
    // int cellIdNew = interp[tid].getCellId(xPos,yPos,zPos);
    // std::cerr << "ERROR in Matrix inversion of stress tensor" << std::endl;
    // p[tid].isActive = false;
    }*/

  // these are the random numbers for each direction
  /*
  vec3 vRandn = { PGD[tid].threadRNG[omp_get_thread_num()][tid].norRan(),
                  PGD[tid].threadRNG[omp_get_thread_num()][tid].norRan(),
                  PGD[tid].threadRNG[omp_get_thread_num()][tid].norRan() };
  */
  vec3 vRandn = { 0.1f, 0.1f, 0.1f };

  // now calculate a bunch of values for the current particle
  // calculate the time derivative of the stress tensor: (tau_current - tau_old)/dt


  /*
     mat3sym tau_ddt = { (p[tid].tau._11 - p[tid].tau_old._11) / par_dt,
                      (p[tid].tau._12 - p[tid].tau_old._12) / par_dt,
                      (p[tid].tau._13 - p[tid].tau_old._13) / par_dt,
                      (p[tid].tau._22 - p[tid].tau_old._22) / par_dt,
                      (p[tid].tau._23 - p[tid].tau_old._23) / par_dt,
                      (p[tid].tau._33 - p[tid].tau_old._33) / par_dt };
  */
  /*
    mat3sym tau = p.tau[tid];
  mat3sym tau_old = p.tau_old[tid];
  mat3sym tau_ddt = { (tau._11 - tau_old._11) / par_dt,
                      (tau._12 - tau_old._12) / par_dt,
                      (tau._13 - tau_old._13) / par_dt,
                      (tau._22 - tau_old._22) / par_dt,
                      (tau._23 - tau_old._23) / par_dt,
                      (tau._33 - tau_old._33) / par_dt };
  */
  /*
    mat3sym tau_ddt = { (tau._11 - p.tau_old[tid]._11) / par_dt,
                      (tau._12 - p.tau_old[tid]._12) / par_dt,
                      (tau._13 - p.tau_old[tid]._13) / par_dt,
                      (tau._22 - p.tau_old[tid]._22) / par_dt,
                      (tau._23 - p.tau_old[tid]._23) / par_dt,
                      (tau._33 - p.tau_old[tid]._33) / par_dt };
  */
  mat3sym tau_ddt = { (p.tau[tid]._11 - p.tau_old[tid]._11) / par_dt,
                      (p.tau[tid]._12 - p.tau_old[tid]._12) / par_dt,
                      (p.tau[tid]._13 - p.tau_old[tid]._13) / par_dt,
                      (p.tau[tid]._22 - p.tau_old[tid]._22) / par_dt,
                      (p.tau[tid]._23 - p.tau_old[tid]._23) / par_dt,
                      (p.tau[tid]._33 - p.tau_old[tid]._33) / par_dt };

  // now calculate and set the A and b matrices for an Ax = b
  // A = -I + 0.5*(-CoEps*L + dTdt*L )*par_dt;
  mat3 A = { -1.0f + 0.50f * (-CoEps * L._11 + L._11 * tau_ddt._11 + L._12 * tau_ddt._12 + L._13 * tau_ddt._13) * par_dt,
             -0.0f + 0.50f * (-CoEps * L._12 + L._12 * tau_ddt._11 + L._22 * tau_ddt._12 + L._23 * tau_ddt._13) * par_dt,
             -0.0f + 0.50f * (-CoEps * L._13 + L._13 * tau_ddt._11 + L._23 * tau_ddt._12 + L._33 * tau_ddt._13) * par_dt,
             -0.0f + 0.50f * (-CoEps * L._12 + L._11 * tau_ddt._12 + L._12 * tau_ddt._22 + L._13 * tau_ddt._23) * par_dt,
             -1.0f + 0.50f * (-CoEps * L._22 + L._12 * tau_ddt._12 + L._22 * tau_ddt._22 + L._23 * tau_ddt._23) * par_dt,
             -0.0f + 0.50f * (-CoEps * L._23 + L._13 * tau_ddt._12 + L._23 * tau_ddt._22 + L._33 * tau_ddt._23) * par_dt,
             -0.0f + 0.50f * (-CoEps * L._13 + L._11 * tau_ddt._13 + L._12 * tau_ddt._23 + L._13 * tau_ddt._33) * par_dt,
             -0.0f + 0.50f * (-CoEps * L._23 + L._12 * tau_ddt._13 + L._22 * tau_ddt._23 + L._23 * tau_ddt._33) * par_dt,
             -1.0f + 0.50f * (-CoEps * L._33 + L._13 * tau_ddt._13 + L._23 * tau_ddt._23 + L._33 * tau_ddt._33) * par_dt };

  /*double A_11 = -1.0 + 0.50 * (-p[tid].CoEps * lxx + lxx * dtxxdt + lxy * dtxydt + lxz * dtxzdt) * par_dt;
  double A_12 = 0.50 * (-p[tid].CoEps * lxy + lxy * dtxxdt + lyy * dtxydt + lyz * dtxzdt) * par_dt;
  double A_13 = 0.50 * (-p[tid].CoEps * lxz + lxz * dtxxdt + lyz * dtxydt + lzz * dtxzdt) * par_dt;

  double A_21 = 0.50 * (-p[tid].CoEps * lxy + lxx * dtxydt + lxy * dtyydt + lxz * dtyzdt) * par_dt;
  double A_22 = -1.0 + 0.50 * (-p[tid].CoEps * lyy + lxy * dtxydt + lyy * dtyydt + lyz * dtyzdt) * par_dt;
  double A_23 = 0.50 * (-p[tid].CoEps * lyz + lxz * dtxydt + lyz * dtyydt + lzz * dtyzdt) * par_dt;

  double A_31 = 0.50 * (-p[tid].CoEps * lxz + lxx * dtxzdt + lxy * dtyzdt + lxz * dtzzdt) * par_dt;
  double A_32 = 0.50 * (-p[tid].CoEps * lyz + lxy * dtxzdt + lyy * dtyzdt + lyz * dtzzdt) * par_dt;
  double A_33 = -1.0 + 0.50 * (-p[tid].CoEps * lzz + lxz * dtxzdt + lyz * dtyzdt + lzz * dtzzdt) * par_dt;*/

  // b = -vectFluct - 0.5*vecFluxDiv*par_dt - sqrt(CoEps*par_dt)*vecRandn;
  vec3 b = { -p.velFluct_old[tid]._1 - 0.50f * p.flux_div[tid]._1 * par_dt - std::sqrt(CoEps * par_dt) * vRandn._1,
             -p.velFluct_old[tid]._2 - 0.50f * p.flux_div[tid]._2 * par_dt - std::sqrt(CoEps * par_dt) * vRandn._2,
             -p.velFluct_old[tid]._3 - 0.50f * p.flux_div[tid]._3 * par_dt - std::sqrt(CoEps * par_dt) * vRandn._3 };

  /*double b_11 = -p[tid].uFluct_old - 0.50 * flux_div_x * par_dt - std::sqrt(p[tid].CoEps * par_dt) * xRandn;
  double b_21 = -p[tid].vFluct_old - 0.50 * flux_div_y * par_dt - std::sqrt(p[tid].CoEps * par_dt) * yRandn;
  double b_31 = -p[tid].wFluct_old - 0.50 * flux_div_z * par_dt - std::sqrt(p[tid].CoEps * par_dt) * zRandn;*/

  // now prepare for the Ax=b calculation by calculating the inverted A matrix
  isRogue = !invert(A);
  if (isRogue) {
    p.state[tid] = ROGUE;
    return;
  }
  /*if (isRogue) {
    // std::cerr << "ERROR in matrix inversion in Langevin equation" << std::endl;
    isActive = false;
    }*/

  // now do the Ax=b calculation using the inverted matrix (vecFluct = A*b)
  multiply(A, b, p.velFluct[tid]);

  // now check to see if the value is rogue or not
  if (std::abs(p.velFluct[tid]._1) >= vel_threshold || isnan(p.velFluct[tid]._1)) {
    // std::cerr << "Particle # " << p[tid].particleID << " is rogue, ";
    // std::cerr << "uFluct = " << p[tid].velFluct._1 << ", CoEps = " << p[tid].CoEps << std::endl;
    p.velFluct[tid]._1 = 0.0;
    // isActive = false;
    isRogue = true;
  }
  if (std::abs(p.velFluct[tid]._2) >= vel_threshold || isnan(p.velFluct[tid]._2)) {
    // std::cerr << "Particle # " << p[tid].particleID << " is rogue, ";
    // std::cerr << "vFluct = " << p[tid].velFluct._2 << ", CoEps = " << p[tid].CoEps << std::endl;
    p.velFluct[tid]._2 = 0.0;
    // isActive = false;
    isRogue = true;
  }
  if (std::abs(p.velFluct[tid]._3) >= vel_threshold || isnan(p.velFluct[tid]._3)) {
    // std::cerr << "Particle # " << p[tid].particleID << " is rogue, ";
    // std::cerr << "wFluct = " << p[tid].velFluct._3 << ", CoEps = " << p[tid].CoEps << std::endl;
    p.velFluct[tid]._3 = 0.0;
    // isActive = false;
    isRogue = true;
  }

  if (isRogue)
    p.state[tid] = ROGUE;

  // p.velFluct[tid]._1 = velFluct._1;
  // p.velFluct[tid]._2 = velFluct._2;
  // p.velFluct[tid]._3 = velFluct._3;

  // now update the old values to be ready for the next particle time iteration
  // the current values are already set for the next iteration by the above calculations
  // p[tid].tau_old = p[tid].tau;
  p.tau_old[tid] = p.tau[tid];
}
