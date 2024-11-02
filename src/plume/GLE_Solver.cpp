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

#include "GLE_Solver.h"
#include "PLUMEGeneralData.h"

void GLE_Solver_CPU::solve(Particle *p, double &par_dt, TURBGeneralData *TGD, PLUMEGeneralData *PGD)
{

  double txx = 0.0, txy = 0.0, txz = 0.0, tyy = 0.0, tyz = 0.0, tzz = 0.0;
  double flux_div_x = 0.0, flux_div_y = 0.0, flux_div_z = 0.0;

  PGD->interp->interpValues(TGD,
                            p->xPos,
                            p->yPos,
                            p->zPos,
                            txx,
                            txy,
                            txz,
                            tyy,
                            tyz,
                            tzz,
                            flux_div_x,
                            flux_div_y,
                            flux_div_z,
                            p->nuT,
                            p->CoEps);

  mat3sym tau = { static_cast<float>(txx),
                  static_cast<float>(txy),
                  static_cast<float>(txz),
                  static_cast<float>(tyy),
                  static_cast<float>(tyz),
                  static_cast<float>(tzz) };
  vec3 flux_div = { static_cast<float>(flux_div_x),
                    static_cast<float>(flux_div_y),
                    static_cast<float>(flux_div_z) };

  // mat3sym tau = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  // vec3 flux_div = { 0.0, 0.0, 0.0 };

  // PGD->interp->interpValues(TGD,p->pos,tau,flux_div,p->nuT,p->CoEps);

  // now need to call makeRealizable on tau
  VectorMath::makeRealizable(PGD->invarianceTol, tau);

  // now need to calculate the inverse values for tau
  // directly modifies the values of tau
  mat3 L = { tau._11, tau._12, tau._13, tau._12, tau._22, tau._23, tau._13, tau._23, tau._33 };

  p->isRogue = !VectorMath::invert(L);
  if (p->isRogue) {
    // int cellIdNew = interp->getCellId(xPos,yPos,zPos);
    std::cerr << "ERROR in Matrix inversion of stress tensor" << std::endl;
    p->isActive = false;
  }

  // these are the random numbers for each direction

#ifdef _OPENMP
  vec3 vRandn = { static_cast<float>(PGD->threadRNG[omp_get_thread_num()]->norRan()),
                  static_cast<float>(PGD->threadRNG[omp_get_thread_num()]->norRan()),
                  static_cast<float>(PGD->threadRNG[omp_get_thread_num()]->norRan()) };
#else
  vec3 vRandn = { (float)PGD->RNG->norRan(), (float)PGD->RNG->norRan(), (float)PGD->RNG->norRan() };
#endif

  // now calculate a bunch of values for the current particle
  // calculate the time derivative of the stress tensor: (tau_current - tau_old)/dt
  mat3sym tau_ddt = { static_cast<float>((tau._11 - p->tau._11) / par_dt),
                      static_cast<float>((tau._12 - p->tau._12) / par_dt),
                      static_cast<float>((tau._13 - p->tau._13) / par_dt),
                      static_cast<float>((tau._22 - p->tau._22) / par_dt),
                      static_cast<float>((tau._23 - p->tau._23) / par_dt),
                      static_cast<float>((tau._33 - p->tau._33) / par_dt) };

  // now calculate and set the A and b matrices for an Ax = b
  // A = -I + 0.5*(-CoEps*L + dTdt*L )*par_dt;
  mat3 A = { static_cast<float>(-1.0 + 0.50 * (-p->CoEps * L._11 + L._11 * tau_ddt._11 + L._12 * tau_ddt._12 + L._13 * tau_ddt._13) * par_dt),
             static_cast<float>(-0.0 + 0.50 * (-p->CoEps * L._12 + L._12 * tau_ddt._11 + L._22 * tau_ddt._12 + L._23 * tau_ddt._13) * par_dt),
             static_cast<float>(-0.0 + 0.50 * (-p->CoEps * L._13 + L._13 * tau_ddt._11 + L._23 * tau_ddt._12 + L._33 * tau_ddt._13) * par_dt),
             static_cast<float>(-0.0 + 0.50 * (-p->CoEps * L._12 + L._11 * tau_ddt._12 + L._12 * tau_ddt._22 + L._13 * tau_ddt._23) * par_dt),
             static_cast<float>(-1.0 + 0.50 * (-p->CoEps * L._22 + L._12 * tau_ddt._12 + L._22 * tau_ddt._22 + L._23 * tau_ddt._23) * par_dt),
             static_cast<float>(-0.0 + 0.50 * (-p->CoEps * L._23 + L._13 * tau_ddt._12 + L._23 * tau_ddt._22 + L._33 * tau_ddt._23) * par_dt),
             static_cast<float>(-0.0 + 0.50 * (-p->CoEps * L._13 + L._11 * tau_ddt._13 + L._12 * tau_ddt._23 + L._13 * tau_ddt._33) * par_dt),
             static_cast<float>(-0.0 + 0.50 * (-p->CoEps * L._23 + L._12 * tau_ddt._13 + L._22 * tau_ddt._23 + L._23 * tau_ddt._33) * par_dt),
             static_cast<float>(-1.0 + 0.50 * (-p->CoEps * L._33 + L._13 * tau_ddt._13 + L._23 * tau_ddt._23 + L._33 * tau_ddt._33) * par_dt) };

  /*double A_11 = -1.0 + 0.50 * (-p->CoEps * lxx + lxx * dtxxdt + lxy * dtxydt + lxz * dtxzdt) * par_dt;
  double A_12 = 0.50 * (-p->CoEps * lxy + lxy * dtxxdt + lyy * dtxydt + lyz * dtxzdt) * par_dt;
  double A_13 = 0.50 * (-p->CoEps * lxz + lxz * dtxxdt + lyz * dtxydt + lzz * dtxzdt) * par_dt;

  double A_21 = 0.50 * (-p->CoEps * lxy + lxx * dtxydt + lxy * dtyydt + lxz * dtyzdt) * par_dt;
  double A_22 = -1.0 + 0.50 * (-p->CoEps * lyy + lxy * dtxydt + lyy * dtyydt + lyz * dtyzdt) * par_dt;
  double A_23 = 0.50 * (-p->CoEps * lyz + lxz * dtxydt + lyz * dtyydt + lzz * dtyzdt) * par_dt;

  double A_31 = 0.50 * (-p->CoEps * lxz + lxx * dtxzdt + lxy * dtyzdt + lxz * dtzzdt) * par_dt;
  double A_32 = 0.50 * (-p->CoEps * lyz + lxy * dtxzdt + lyy * dtyzdt + lyz * dtzzdt) * par_dt;
  double A_33 = -1.0 + 0.50 * (-p->CoEps * lzz + lxz * dtxzdt + lyz * dtyzdt + lzz * dtzzdt) * par_dt;*/

  // b = -vectFluct - 0.5*vecFluxDiv*par_dt - sqrt(CoEps*par_dt)*vecRandn;
  vec3 b = { static_cast<float>(-p->velFluct_old._1 - 0.50 * flux_div._1 * par_dt - std::sqrt(p->CoEps * par_dt) * vRandn._1),
             static_cast<float>(-p->velFluct_old._2 - 0.50 * flux_div._2 * par_dt - std::sqrt(p->CoEps * par_dt) * vRandn._2),
             static_cast<float>(-p->velFluct_old._3 - 0.50 * flux_div._3 * par_dt - std::sqrt(p->CoEps * par_dt) * vRandn._3) };

  /*double b_11 = -p->uFluct_old - 0.50 * flux_div_x * par_dt - std::sqrt(p->CoEps * par_dt) * xRandn;
  double b_21 = -p->vFluct_old - 0.50 * flux_div_y * par_dt - std::sqrt(p->CoEps * par_dt) * yRandn;
  double b_31 = -p->wFluct_old - 0.50 * flux_div_z * par_dt - std::sqrt(p->CoEps * par_dt) * zRandn;*/

  // now prepare for the Ax=b calculation by calculating the inverted A matrix
  p->isRogue = !VectorMath::invert(A);
  if (p->isRogue) {
    std::cerr << "ERROR in matrix inversion in Langevin equation" << std::endl;
    p->isActive = false;
  }

  // now do the Ax=b calculation using the inverted matrix (vecFluct = A*b)
  VectorMath::multiply(A, b, p->velFluct);

  // now check to see if the value is rogue or not
  if (std::abs(p->velFluct._1) >= PGD->vel_threshold || isnan(p->velFluct._1)) {
    std::cerr << "Particle # " << p->particleID << " is rogue, ";
    std::cerr << "uFluct = " << p->velFluct._1 << ", CoEps = " << p->CoEps << std::endl;
    p->velFluct._1 = 0.0;
    p->isActive = false;
    p->isRogue = true;
  }
  if (std::abs(p->velFluct._2) >= PGD->vel_threshold || isnan(p->velFluct._2)) {
    std::cerr << "Particle # " << p->particleID << " is rogue, ";
    std::cerr << "vFluct = " << p->velFluct._2 << ", CoEps = " << p->CoEps << std::endl;
    p->velFluct._2 = 0.0;
    p->isActive = false;
    p->isRogue = true;
  }
  if (std::abs(p->velFluct._3) >= PGD->vel_threshold || isnan(p->velFluct._3)) {
    std::cerr << "Particle # " << p->particleID << " is rogue, ";
    std::cerr << "wFluct = " << p->velFluct._3 << ", CoEps = " << p->CoEps << std::endl;
    p->velFluct._3 = 0.0;
    p->isActive = false;
    p->isRogue = true;
  }

  // now update the old values to be ready for the next particle time iteration
  // the current values are already set for the next iteration by the above calculations
  /*p->txx = txx;
  p->txy = txy;
  p->txz = txz;
  p->tyy = tyy;
  p->tyz = tyz;
  p->tzz = tzz;*/

  p->tau = tau;
}
