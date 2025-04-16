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

/** @file GLE_Solver.cpp
 * @brief
 */

#include "GLE_Solver.h"
#include "PLUMEGeneralData.h"

void GLE_Solver_CPU::solve(float &dt,
                           ParticleCore &p_core,
                           ParticleLSDM &p_lsdm,
                           ParticleState &state,
                           TURBGeneralData *TGD,
                           PLUMEGeneralData *PGD)
{
  mat3sym tau = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  vec3 flux_div = { 0.0, 0.0, 0.0 };

  PGD->interp->interpTurbValues(TGD, p_core.pos, tau, flux_div, p_lsdm.nuT, p_lsdm.CoEps);

  // now need to call makeRealizable on tau
  VectorMath::makeRealizable(PGD->invarianceTol, tau);

  // now need to calculate the inverse values for tau
  mat3 L = { tau._11, tau._12, tau._13, tau._12, tau._22, tau._23, tau._13, tau._23, tau._33 };

  bool isRogue = !VectorMath::invert(L);
  if (isRogue) {
    // std::cerr << "ERROR in Matrix inversion of stress tensor" << std::endl;
    p_lsdm.velFluct = { 0.0, 0.0, 0.0 };
    state = ROGUE;
    return;
  }

  // these are the random numbers for each direction
#ifdef _OPENMP
  vec3 vRandn = { PGD->threadRNG[omp_get_thread_num()]->norRan(),
                  PGD->threadRNG[omp_get_thread_num()]->norRan(),
                  PGD->threadRNG[omp_get_thread_num()]->norRan() };
#else
  vec3 vRandn = { PGD->RNG->norRan(), PGD->RNG->norRan(), PGD->RNG->norRan() };
#endif

  // calculate the time derivative of the stress tensor: (tau_current - tau_old)/dt
  mat3sym tau_ddt = { (tau._11 - p_lsdm.tau._11) / dt,
                      (tau._12 - p_lsdm.tau._12) / dt,
                      (tau._13 - p_lsdm.tau._13) / dt,
                      (tau._22 - p_lsdm.tau._22) / dt,
                      (tau._23 - p_lsdm.tau._23) / dt,
                      (tau._33 - p_lsdm.tau._33) / dt };

  // calculate and set the A and b matrices for 3D GLE
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

  // now prepare to solve Ax=b by calculating the inverted A matrix
  isRogue = !VectorMath::invert(A);
  if (isRogue) {
    // std::cerr << "ERROR in matrix inversion in Langevin equation" << std::endl;
    p_lsdm.velFluct = { 0.0, 0.0, 0.0 };
    state = ROGUE;
    return;
  }

  // now calculate new fluctuation vector: vecFluct = A*b (A is the inverse here)
  VectorMath::multiply(A, b, p_lsdm.velFluct);

  // now check to see if the value is rogue or not
  if (std::abs(p_lsdm.velFluct._1) >= PGD->vel_threshold || isnan(p_lsdm.velFluct._1)) {
    // std::cerr << "Particle # " << p->ID << " is rogue, ";
    // std::cerr << "uFluct = " << p_lsdm.velFluct._1 << ", CoEps = " << p_lsdm.CoEps << std::endl;
    // p_lsdm.velFluct = { 0.0, 0.0, 0.0 };
    isRogue = true;
  }
  if (std::abs(p_lsdm.velFluct._2) >= PGD->vel_threshold || isnan(p_lsdm.velFluct._2)) {
    // std::cerr << "Particle # " << p->ID << " is rogue, ";
    // std::cerr << "vFluct = " << p_lsdm.velFluct._2 << ", CoEps = " << p_lsdm.CoEps << std::endl;
    // p_lsdm.velFluct = { 0.0, 0.0, 0.0 };
    isRogue = true;
  }
  if (std::abs(p_lsdm.velFluct._3) >= PGD->vel_threshold || isnan(p_lsdm.velFluct._3)) {
    // std::cerr << "Particle # " << p->ID << " is rogue, ";
    // std::cerr << "wFluct = " << p_lsdm.velFluct._3 << ", CoEps = " << p_lsdm.CoEps << std::endl;
    // p_lsdm.velFluct = { 0.0, 0.0, 0.0 };
    isRogue = true;
  }

  // now update the old values to be ready for the next particle time iteration
  p_lsdm.tau = tau;

  if (isRogue) {
    p_lsdm.velFluct = { 0.0, 0.0, 0.0 };
    state = ROGUE;
  }
}
