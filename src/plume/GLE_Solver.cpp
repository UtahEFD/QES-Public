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

void GLE_Solver_CPU::solve(Particle *p, float &dt, TURBGeneralData *TGD, PLUMEGeneralData *PGD)
{
  bool isRogue = false;

  double txx = 0.0, txy = 0.0, txz = 0.0, tyy = 0.0, tyz = 0.0, tzz = 0.0;
  double flux_div_x = 0.0, flux_div_y = 0.0, flux_div_z = 0.0;

  mat3sym tau = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  vec3 flux_div = { 0.0, 0.0, 0.0 };

  PGD->interp->interpTurbValues(TGD, p->pos, tau, flux_div, p->nuT, p->CoEps);
  /*
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
                    static_cast<float>(flux_div_z) };*/

  // now need to call makeRealizable on tau
  VectorMath::makeRealizable(PGD->invarianceTol, tau);

  // now need to calculate the inverse values for tau
  // directly modifies the values of tau
  mat3 L = { tau._11, tau._12, tau._13, tau._12, tau._22, tau._23, tau._13, tau._23, tau._33 };

  isRogue = !VectorMath::invert(L);
  if (isRogue) {
    // int cellIdNew = interp->getCellId(xPos,yPos,zPos);
    std::cerr << "ERROR in Matrix inversion of stress tensor" << std::endl;
    // isActive = false;
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
  mat3sym tau_ddt = { (tau._11 - p->tau._11) / dt,
                      (tau._12 - p->tau._12) / dt,
                      (tau._13 - p->tau._13) / dt,
                      (tau._22 - p->tau._22) / dt,
                      (tau._23 - p->tau._23) / dt,
                      (tau._33 - p->tau._33) / dt };

  // now calculate and set the A and b matrices for an Ax = b
  // A = -I + 0.5*(-CoEps*L + dTdt*L )*par_dt;
  mat3 A = { -1.0f + 0.5f * (-p->CoEps * L._11 + L._11 * tau_ddt._11 + L._12 * tau_ddt._12 + L._13 * tau_ddt._13) * dt,
             -0.0f + 0.5f * (-p->CoEps * L._12 + L._12 * tau_ddt._11 + L._22 * tau_ddt._12 + L._23 * tau_ddt._13) * dt,
             -0.0f + 0.5f * (-p->CoEps * L._13 + L._13 * tau_ddt._11 + L._23 * tau_ddt._12 + L._33 * tau_ddt._13) * dt,
             -0.0f + 0.5f * (-p->CoEps * L._12 + L._11 * tau_ddt._12 + L._12 * tau_ddt._22 + L._13 * tau_ddt._23) * dt,
             -1.0f + 0.5f * (-p->CoEps * L._22 + L._12 * tau_ddt._12 + L._22 * tau_ddt._22 + L._23 * tau_ddt._23) * dt,
             -0.0f + 0.5f * (-p->CoEps * L._23 + L._13 * tau_ddt._12 + L._23 * tau_ddt._22 + L._33 * tau_ddt._23) * dt,
             -0.0f + 0.5f * (-p->CoEps * L._13 + L._11 * tau_ddt._13 + L._12 * tau_ddt._23 + L._13 * tau_ddt._33) * dt,
             -0.0f + 0.5f * (-p->CoEps * L._23 + L._12 * tau_ddt._13 + L._22 * tau_ddt._23 + L._23 * tau_ddt._33) * dt,
             -1.0f + 0.5f * (-p->CoEps * L._33 + L._13 * tau_ddt._13 + L._23 * tau_ddt._23 + L._33 * tau_ddt._33) * dt };

  // b = -vectFluct - 0.5*vecFluxDiv*par_dt - sqrt(CoEps*par_dt)*vecRandn;
  vec3 b = { -p->velFluct_old._1 - 0.5f * flux_div._1 * dt - sqrtf(p->CoEps * dt) * vRandn._1,
             -p->velFluct_old._2 - 0.5f * flux_div._2 * dt - sqrtf(p->CoEps * dt) * vRandn._2,
             -p->velFluct_old._3 - 0.5f * flux_div._3 * dt - sqrtf(p->CoEps * dt) * vRandn._3 };

  // now prepare for the Ax=b calculation by calculating the inverted A matrix
  isRogue = !VectorMath::invert(A);
  if (isRogue) {
    std::cerr << "ERROR in matrix inversion in Langevin equation" << std::endl;
    // isActive = false;
  }

  // now do the Ax=b calculation using the inverted matrix (vecFluct = A*b)
  VectorMath::multiply(A, b, p->velFluct);

  // now check to see if the value is rogue or not
  if (std::abs(p->velFluct._1) >= PGD->vel_threshold || isnan(p->velFluct._1)) {
    std::cerr << "Particle # " << p->ID << " is rogue, ";
    std::cerr << "uFluct = " << p->velFluct._1 << ", CoEps = " << p->CoEps << std::endl;
    p->velFluct._1 = 0.0;
    // isActive = false;
    isRogue = true;
  }
  if (std::abs(p->velFluct._2) >= PGD->vel_threshold || isnan(p->velFluct._2)) {
    std::cerr << "Particle # " << p->ID << " is rogue, ";
    std::cerr << "vFluct = " << p->velFluct._2 << ", CoEps = " << p->CoEps << std::endl;
    p->velFluct._2 = 0.0;
    // isActive = false;
    isRogue = true;
  }
  if (std::abs(p->velFluct._3) >= PGD->vel_threshold || isnan(p->velFluct._3)) {
    std::cerr << "Particle # " << p->ID << " is rogue, ";
    std::cerr << "wFluct = " << p->velFluct._3 << ", CoEps = " << p->CoEps << std::endl;
    p->velFluct._3 = 0.0;
    // isActive = false;
    isRogue = true;
  }

  // now update the old values to be ready for the next particle time iteration
  // the current values are already set for the next iteration by the above calculations
  p->tau = tau;

  if (isRogue) p->state = ROGUE;
}

void GLE_Solver_CPU::solve(float &dt,
                           ParticleControl &p_control,
                           ParticleCore &p_core,
                           ParticleLSDM &p_lsdm,
                           TURBGeneralData *TGD,
                           PLUMEGeneralData *PGD)
{
  bool isRogue = false;

  double txx = 0.0, txy = 0.0, txz = 0.0, tyy = 0.0, tyz = 0.0, tzz = 0.0;
  double flux_div_x = 0.0, flux_div_y = 0.0, flux_div_z = 0.0;

  mat3sym tau = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  vec3 flux_div = { 0.0, 0.0, 0.0 };

  PGD->interp->interpTurbValues(TGD, p_core.pos, tau, flux_div, p_lsdm.nuT, p_lsdm.CoEps);

  // now need to call makeRealizable on tau
  mat3sym tau_db = tau;
  VectorMath::makeRealizable(PGD->invarianceTol, tau);

  // now need to calculate the inverse values for tau
  // directly modifies the values of tau
  mat3 L = { tau._11, tau._12, tau._13, tau._12, tau._22, tau._23, tau._13, tau._23, tau._33 };

  isRogue = !VectorMath::invert(L);
  if (isRogue) {
    // int cellIdNew = interp->getCellId(xPos,yPos,zPos);
    std::cerr << "ERROR in Matrix inversion of stress tensor" << std::endl;
    std::cerr << "particle position: "
              << p_core.pos._1 << " " << p_core.pos._2 << " " << p_core.pos._3 << " "
              << tau._11 << " " << tau._12 << " " << tau._13 << " "
              << tau._22 << " " << tau._23 << " " << tau._33 << std::endl;
  }

  // these are the random numbers for each direction

#ifdef _OPENMP
  vec3 vRandn = { static_cast<float>(PGD->threadRNG[omp_get_thread_num()]->norRan()),
                  static_cast<float>(PGD->threadRNG[omp_get_thread_num()]->norRan()),
                  static_cast<float>(PGD->threadRNG[omp_get_thread_num()]->norRan()) };
#else
  vec3 vRandn = { (float)PGD->RNG->norRan(),
                  (float)PGD->RNG->norRan(),
                  (float)PGD->RNG->norRan() };
#endif

  // now calculate a bunch of values for the current particle
  // calculate the time derivative of the stress tensor: (tau_current - tau_old)/dt
  mat3sym tau_ddt = { (tau._11 - p_lsdm.tau._11) / dt,
                      (tau._12 - p_lsdm.tau._12) / dt,
                      (tau._13 - p_lsdm.tau._13) / dt,
                      (tau._22 - p_lsdm.tau._22) / dt,
                      (tau._23 - p_lsdm.tau._23) / dt,
                      (tau._33 - p_lsdm.tau._33) / dt };

  // now calculate and set the A and b matrices for an Ax = b
  // A = -I + 0.5*(-CoEps*L + dTdt*L )*par_dt;
  mat3 A = { -1.0f + 0.5f * (-p_lsdm.CoEps * L._11 + L._11 * tau_ddt._11 + L._12 * tau_ddt._12 + L._13 * tau_ddt._13) * dt,
             -0.0f + 0.5f * (-p_lsdm.CoEps * L._12 + L._12 * tau_ddt._11 + L._22 * tau_ddt._12 + L._23 * tau_ddt._13) * dt,
             -0.0f + 0.5f * (-p_lsdm.CoEps * L._13 + L._13 * tau_ddt._11 + L._23 * tau_ddt._12 + L._33 * tau_ddt._13) * dt,
             -0.0f + 0.5f * (-p_lsdm.CoEps * L._12 + L._11 * tau_ddt._12 + L._12 * tau_ddt._22 + L._13 * tau_ddt._23) * dt,
             -1.0f + 0.5f * (-p_lsdm.CoEps * L._22 + L._12 * tau_ddt._12 + L._22 * tau_ddt._22 + L._23 * tau_ddt._23) * dt,
             -0.0f + 0.5f * (-p_lsdm.CoEps * L._23 + L._13 * tau_ddt._12 + L._23 * tau_ddt._22 + L._33 * tau_ddt._23) * dt,
             -0.0f + 0.5f * (-p_lsdm.CoEps * L._13 + L._11 * tau_ddt._13 + L._12 * tau_ddt._23 + L._13 * tau_ddt._33) * dt,
             -0.0f + 0.5f * (-p_lsdm.CoEps * L._23 + L._12 * tau_ddt._13 + L._22 * tau_ddt._23 + L._23 * tau_ddt._33) * dt,
             -1.0f + 0.5f * (-p_lsdm.CoEps * L._33 + L._13 * tau_ddt._13 + L._23 * tau_ddt._23 + L._33 * tau_ddt._33) * dt };

  // b = -vectFluct - 0.5*vecFluxDiv*par_dt - sqrt(CoEps*par_dt)*vecRandn;
  vec3 b = { -p_lsdm.velFluct_old._1 - 0.5f * flux_div._1 * dt - sqrtf(p_lsdm.CoEps * dt) * vRandn._1,
             -p_lsdm.velFluct_old._2 - 0.5f * flux_div._2 * dt - sqrtf(p_lsdm.CoEps * dt) * vRandn._2,
             -p_lsdm.velFluct_old._3 - 0.5f * flux_div._3 * dt - sqrtf(p_lsdm.CoEps * dt) * vRandn._3 };

  // now prepare for the Ax=b calculation by calculating the inverted A matrix
  isRogue = !VectorMath::invert(A);
  if (isRogue) {
    std::cerr << "ERROR in matrix inversion in Langevin equation" << std::endl;
    // isActive = false;
  }

  // now do the Ax=b calculation using the inverted matrix (vecFluct = A*b)
  VectorMath::multiply(A, b, p_lsdm.velFluct);

  // now check to see if the value is rogue or not
  if (std::abs(p_lsdm.velFluct._1) >= PGD->vel_threshold || isnan(p_lsdm.velFluct._1)) {
    // std::cerr << "Particle # " << p->ID << " is rogue, ";
    std::cerr << "uFluct = " << p_lsdm.velFluct._1 << ", CoEps = " << p_lsdm.CoEps << std::endl;
    p_lsdm.velFluct._1 = 0.0;
    // isActive = false;
    isRogue = true;
  }
  if (std::abs(p_lsdm.velFluct._2) >= PGD->vel_threshold || isnan(p_lsdm.velFluct._2)) {
    // std::cerr << "Particle # " << p->ID << " is rogue, ";
    std::cerr << "vFluct = " << p_lsdm.velFluct._2 << ", CoEps = " << p_lsdm.CoEps << std::endl;
    p_lsdm.velFluct._2 = 0.0;
    // isActive = false;
    isRogue = true;
  }
  if (std::abs(p_lsdm.velFluct._3) >= PGD->vel_threshold || isnan(p_lsdm.velFluct._3)) {
    // std::cerr << "Particle # " << p->ID << " is rogue, ";
    std::cerr << "wFluct = " << p_lsdm.velFluct._3 << ", CoEps = " << p_lsdm.CoEps << std::endl;
    p_lsdm.velFluct._3 = 0.0;
    // isActive = false;
    isRogue = true;
  }

  // now update the old values to be ready for the next particle time iteration
  // the current values are already set for the next iteration by the above calculations
  p_lsdm.tau = tau;

  if (isRogue) {
    p_control.state = ROGUE;
    /*std::cerr << "Particle: " << (isRogue ? "ROGUE" : "ACTIVE") << "\n"
              << dt << "\n"
              << p_core.pos._1 << " " << p_core.pos._2 << " " << p_core.pos._3 << "\n"
              << p_lsdm.velMean._1 << " " << p_lsdm.velMean._2 << " " << p_lsdm.velMean._3 << "\n"
              << p_lsdm.velFluct_old._1 << " " << p_lsdm.velFluct_old._2 << " " << p_lsdm.velFluct_old._3 << "\n"
              << b._1 << " " << b._2 << " " << b._3 << "\n"
              << tau._11 << " " << tau._12 << " " << tau._13 << " "
              << tau._22 << " " << tau._23 << " " << tau._33 << "\n"
              << tau_ddt._11 << " " << tau_ddt._12 << " " << tau_ddt._13 << " "
              << tau_ddt._22 << " " << tau_ddt._23 << " " << tau_ddt._33 << "\n"
              << flux_div._1 << " " << flux_div._2 << " " << flux_div._3 << std::endl;*/
  }
}
