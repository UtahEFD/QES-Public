/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
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

/** @file DepositParticle.cpp */

#define _USE_MATH_DEFINES
#include <math.h>
#include "Plume.hpp"

void Plume::depositParticle(const double &xPos, const double &yPos, const double &zPos, const double &disX, const double &disY, const double &disZ, const double &uTot, const double &vTot, const double &wTot, const double &txx, const double &tyy, const double &tzz, const double &txz, const double &txy, const double &tyz, const double &vs, const double &CoEps, const double &boxSizeZ, const double &nuT, Particle *par_ptr, WINDSGeneralData *WGD, TURBGeneralData *TGD)
{

  double rhoAir = 1.225;// in kg m^-3
  double nuAir = 1.506E-5;// in m^2 s^-1

  if (par_ptr->isActive == true) {

    // Particle position and attributes
    double xPos_old = xPos - disX;
    double yPos_old = yPos - disY;
    double zPos_old = zPos - disZ;

    int cellId_old = interp->getCellId(xPos_old, yPos_old, zPos_old);
    int cellId = interp->getCellId(xPos, yPos, zPos);
    Vector3Int cellIdx = interp->getCellIndex(cellId);
    int i = cellIdx[0], j = cellIdx[1], k = cellIdx[2];

    double partDist = sqrt(pow(disX, 2) + pow(disY, 2) + pow(disZ, 2));// distance travelled through veg
    double MTot = sqrt(pow(uTot, 2) + pow(vTot, 2) + pow(wTot, 2));// particle speed [m/s]


    // Radial distance-based mass decay from data
    if (false) {

      // Calculate distance (in x-y plane) from source
      double distFromSource = pow(pow(xPos - par_ptr->xPos_init, 2)
                                    + pow(yPos - par_ptr->yPos_init, 2),
                                  0.5);
      // Take deposited mass away from particle
      double P_r = exp(-par_ptr->decayConst * distFromSource);// undeposited fraction of mass
      par_ptr->m = par_ptr->m_o * P_r;
      // par_ptr->m_kg = par_ptr->m_kg_o * P_r;
    }


    // Apply deposition model depending on particle location
    if (WGD->isCanopy(cellId_old)) {// if particle was in veg cell last time step

      // Calculate deposition fraction
      double elementDiameter = 100.0e-3;// temporarily hard-coded [m]
      double leafAreaDensitydep = 5.57;// LAD, temporarily hard-coded [m^-1]
      double Cc = 1.0;// Cunningham correction factor, temporarily hard-coded, only important for <10um particles
      double parRMS = 1.0 / sqrt(3.0) * sqrt(txx + tyy + tzz);// RMS of velocity fluctuations the particle is experiencing [m/s]
      double taylorMicroscale = sqrt((15.0 * nuAir * 5.0 * pow(parRMS, 2.0)) / CoEps);
      double Stk = (par_ptr->rho * pow((1.0E-6) * par_ptr->d, 2.0) * MTot * Cc) / (18.0 * rhoAir * nuAir * elementDiameter);// classical Stokes number
      double ReLambda = parRMS * taylorMicroscale / nuAir;// Taylor microscale Reynolds number
      double depEff = 1.0 - 1.0 / (par_ptr->c1 * pow(pow(ReLambda, 0.3) * Stk, par_ptr->c2) + 1.0);// deposition efficiency (E in Bailey 2018 Eq. 13)
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
      par_ptr->dep_buffer_flag = true;
      par_ptr->dep_buffer_cell.push_back(cellId_old);
      par_ptr->dep_buffer_val.push_back((1.0 - P_v) * par_ptr->m);
      // deposition->depcvol[cellId_old] += (1.0 - P_v) * par_ptr->m;

      // Take deposited mass away from particle
      par_ptr->m *= P_v;
      // par_ptr->m_kg *= P_v;

    } else if (WGD->isTerrain(cellId - (WGD->nx - 1) * (WGD->ny - 1))) {// Ground deposition
      double dt = partDist / MTot;
      double ustarDep = pow(pow(txz, 2.0) + pow(tyz, 2.0), 0.25);
      double Sc = nuAir / nuT;
      double ra = 1.0 / (0.4 * ustarDep) * log(((10000.0 * ustarDep * boxSizeZ) / (2.0 * nuAir) + 1.0 / Sc) / ((100.0 * ustarDep / nuAir) + 1.0 / Sc));
      double Cc = 1.0;// Cunningham correction factor
      double Stk_ground = (vs * pow(ustarDep, 2.0)) / (9.81 * nuAir);
      double rb = 1.0 / (ustarDep * (pow(Sc, -2.0 / 3.0) + pow(10.0, -3.0 / Stk_ground)));
      double vd = 1.0 / (ra + rb + ra * rb * vs) + vs;
      double dz_g = WGD->z_face[k] - WGD->z_face[k - 1];

      double P_g = exp(-vd * dt / dz_g);

      // add deposition amount to the buffer (for parallelization)
      par_ptr->dep_buffer_flag = true;
      par_ptr->dep_buffer_cell.push_back(cellId_old);
      par_ptr->dep_buffer_val.push_back((1.0 - P_g) * par_ptr->m);
      // deposition->depcvol[cellId] += (1.0 - P_g) * par_ptr->m;

      // Take deposited mass away from particle
      par_ptr->m *= P_g;
      // par_ptr->m_kg *= P_g;

    } else {
      return;
    }

    // If particle mass drops below mass of a single particle, set it to zero and inactivate it
    double oneParMass = par_ptr->rho * (1.0 / 6.0) * M_PI * pow((1.0E-6) * par_ptr->d, 3.0);
    if (par_ptr->m / 1000.0 < oneParMass) {
      // par_ptr->m_kg = 0.0;
      par_ptr->m = 0.0;
      par_ptr->isActive = false;
    }

  }// if ( isActive == true )
}
