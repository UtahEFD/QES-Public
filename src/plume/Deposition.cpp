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

/** @file Deposition.cpp
 * @brief
 */

#include "Deposition.h"

Deposition::Deposition(const WINDSGeneralData *WGD)
  : x(WGD->domain.x), y(WGD->domain.y), z(WGD->domain.z), z_face(WGD->domain.z_face)
{
  /* use constructor to make copies.
  x.resize(WGD->nx - 1);
  for (auto k = 0u; k < x.size(); ++k) {
    x[k] = WGD->x[k];
  }
  y.resize(WGD->ny - 1);
  for (auto k = 0u; k < y.size(); ++k) {
    y[k] = WGD->y[k];
  }
  z.resize(WGD->nz - 1);
  for (auto k = 0u; k < z.size(); ++k) {
    z[k] = WGD->z[k];
  }
  z.resize(WGD->z_face);
  for (auto k = 0u; k < z_face.size(); ++k) {
    z_face[k] = WGD->z_face[k];
  }
 */

  // need to pass this as an argument. should it be a parameter?
  boxSizeZ = 1;

#ifdef _OPENMP
  thread_depcvol.resize(WGD->domain.numCellCentered());
  for (auto &it : thread_depcvol) {
    it.resize(omp_get_num_threads(), 0.0);
  }
#else
  depcvol.resize(WGD->domain.numCellCentered(), 0.0);
#endif

  nbrFace = WGD->wall_below_indices.size()
            + WGD->wall_above_indices.size()
            + WGD->wall_back_indices.size()
            + WGD->wall_front_indices.size()
            + WGD->wall_left_indices.size()
            + WGD->wall_right_indices.size();
}

void Deposition::deposit(Particle *p,
                         const double &disX,
                         const double &disY,
                         const double &disZ,
                         const double &uTot,
                         const double &vTot,
                         const double &wTot,
                         const double &vs,
                         WINDSGeneralData *WGD,
                         TURBGeneralData *TGD,
                         Interp *interp)
{

  double rhoAir = 1.225;// in kg m^-3
  double nuAir = 1.506E-5;// in m^2 s^-1

  if (p->isActive) {

    // Particle position and attributes
    double xPos_old = p->xPos - disX;
    double yPos_old = p->yPos - disY;
    double zPos_old = p->zPos - disZ;

    int cellId_old = interp->getCellId(xPos_old, yPos_old, zPos_old);
    int cellId = interp->getCellId(p->xPos, p->yPos, p->zPos);
    Vector3Int cellIdx = interp->getCellIndex(cellId);
    // int i = cellIdx[0];
    // int j = cellIdx[1];
    int k = cellIdx[2];

    // distance travelled through veg
    double partDist = sqrt(pow(disX, 2)
                           + pow(disY, 2)
                           + pow(disZ, 2));
    // particle speed [m/s]
    double MTot = sqrt(pow(uTot, 2)
                       + pow(vTot, 2)
                       + pow(wTot, 2));


    // Radial distance-based mass decay from data
    /*if (false) {

      // Calculate distance (in x-y plane) from source
      double distFromSource = pow(pow(p->xPos - p->xPos_init, 2)
                                    + pow(p->yPos - p->yPos_init, 2),
                                  0.5);
      // Take deposited mass away from particle
      double P_r = exp(-p->decayConst * distFromSource);// undeposited fraction of mass
      p->m = p->m_o * P_r;
      // p->m_kg = p->m_kg_o * P_r;
    }*/


    // Apply deposition model depending on particle location
    if (WGD->isCanopy(cellId_old)) {// if particle was in veg cell last time step

      // Calculate deposition fraction
      double elementDiameter = 100.0e-3;// temporarily hard-coded [m]
      double leafAreaDensitydep = 5.57;// LAD, temporarily hard-coded [m^-1]
      double Cc = 1.0;// Cunningham correction factor, temporarily hard-coded, only important for <10um particles
      double parRMS = 1.0 / sqrt(3.0) * sqrt(p->txx + p->tyy + p->tzz);// RMS of velocity fluctuations the particle is experiencing [m/s]
      double taylorMicroscale = sqrt((15.0 * nuAir * 5.0 * pow(parRMS, 2.0)) / p->CoEps);
      double Stk = (p->rho * pow((1.0E-6) * p->d, 2.0) * MTot * Cc) / (18.0 * rhoAir * nuAir * elementDiameter);// classical Stokes number
      double ReLambda = parRMS * taylorMicroscale / nuAir;// Taylor microscale Reynolds number
      double depEff = 1.0 - 1.0 / (p->c1 * pow(pow(ReLambda, 0.3) * Stk, p->c2) + 1.0);// deposition efficiency (E in Bailey 2018 Eq. 13)
      double ReLeaf = elementDiameter * MTot / nuAir;// leaf Reynolds number

      // Temporary fix to address limitations of Price 2017 model (their correlation is only valid for 400 < ReLeaf < 6000)
      if (ReLeaf > 6000.0) {
        ReLeaf = 6000.0;
      }
      if (ReLeaf < 400.0) {
        ReLeaf = 400.0;
      }

      double gam = -6.5e-5 * ReLeaf + 0.43;// non-impaction surface weighting factor
      // double gam = 0.1; // temporarily set to 1 because of problems (gam is becoming negative, but should be positive and ~0.1)

      double adjLAD = leafAreaDensitydep * (1.0 + gam);// LAD adjusted to include non-impaction surface

      double P_v = exp(-depEff * adjLAD * partDist * 0.7);// the undeposited mass fraction. The /2 comes from Ross' G function, assuming uniform leaf orientation distribution

      // add deposition amount to the buffer (for parallelization)
#ifdef _OPENMP
      thread_depcvol[cellId_old][omp_get_thread_num()] += (1.0 - P_v) * p->m;
#else
      depcvol[cellId_old] += (1.0 - P_v) * p->m;
#endif

      // Take deposited mass away from particle
      p->m *= P_v;
      // p->m_kg *= P_v;
    } else if (WGD->isTerrain(WGD->domain.cellAdd(cellId, 0, 0, -1))) {// Ground deposition
      double dt = partDist / MTot;
      double ustarDep = pow(pow(p->txz, 2.0) + pow(p->tyz, 2.0), 0.25);
      double Sc = nuAir / p->nuT;
      double ra = 1.0 / (0.4 * ustarDep) * log(((10000.0 * ustarDep * boxSizeZ) / (2.0 * nuAir) + 1.0 / Sc) / ((100.0 * ustarDep / nuAir) + 1.0 / Sc));
      double Cc = 1.0;// Cunningham correction factor
      double Stk_ground = (vs * pow(ustarDep, 2.0)) / (9.81 * nuAir);
      double rb = 1.0 / (ustarDep * (pow(Sc, -2.0 / 3.0) + pow(10.0, -3.0 / Stk_ground)));
      double vd = 1.0 / (ra + rb + ra * rb * vs) + vs;
      double dz_g = WGD->domain.z_face[k] - WGD->domain.z_face[k - 1];

      double P_g = exp(-vd * dt / dz_g);

      // add deposition amount to the buffer (for parallelization)
#ifdef _OPENMP
      thread_depcvol[cellId_old][omp_get_thread_num()] += (1.0 - P_g) * p->m;
#else
      depcvol[cellId_old] += (1.0 - P_g) * p->m;
#endif

      // Take deposited mass away from particle
      p->m *= P_g;
      // p->m_kg *= P_g;
    } else {
      return;
    }

    // If particle mass drops below mass of a single particle, set it to zero and inactivate it
    double oneParMass = p->rho * (1.0 / 6.0) * M_PI * pow((1.0E-6) * p->d, 3.0);
    if (p->m / 1000.0 < oneParMass) {
      // p->m_kg = 0.0;
      p->m = 0.0;
      p->isActive = false;
    }

  }// if ( isActive == true )
}
