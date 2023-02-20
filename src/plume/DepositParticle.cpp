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

void Plume::depositParticle(double xPos, double yPos, double zPos, double disX, double disY, double disZ, double uTot, double vTot, double wTot, double txx, double tyy, double tzz, double CoEps, std::list<Particle *>::iterator parItr, WINDSGeneralData *WGD, TURBGeneralData *TGD)
{

  double rhoAir = 1.225;// in kg m^-3
  double nuAir = 1.506E-5;// in m^2 s^-1

  if ((*parItr)->isActive == true) {

    //    std::cout << "Deposition code running (particle not necessarily depositing) \n";

    // Determine if particle was in veg cell last time step
    double xPos_old = xPos - disX;
    double yPos_old = yPos - disY;
    double zPos_old = zPos - disZ;

    int cellId_old = interp->getCellId(xPos_old, yPos_old, zPos_old);

    // If so, calculate deposition fraction
    if (WGD->icellflag[cellId_old] == 20 || WGD->icellflag[cellId_old] == 22 || WGD->icellflag[cellId_old] == 24 || WGD->icellflag[cellId_old] == 28) {

      double elementDiameter = 100e-3;// temporarily hard-coded [m]
      double leafAreaDensitydep = 5.57;// LAD, temporarily hard-coded [m^-1]
      double Cc = 1;// Cunningham correction factor, temporarily hard-coded, only important for <10um particles

      double vegDistance = sqrt(pow(disX, 2) + pow(disY, 2) + pow(disZ, 2));// distance travelled through veg

      double MTot = sqrt(pow(uTot, 2) + pow(vTot, 2) + pow(wTot, 2));// particle speed [m/s]

      double parRMS = 1 / sqrt(3) * sqrt(txx + tyy + tzz);// RMS of velocity fluctuations the particle is experiencing [m/s]

      double taylorMicroscale = sqrt((15 * nuAir * 5 * pow(parRMS, 2)) / CoEps);

      double Stk = ((*parItr)->rho * pow((*parItr)->d_m, 2) * MTot * Cc) / (18 * rhoAir * nuAir * elementDiameter);// classical Stokes number

      double ReLambda = parRMS * taylorMicroscale / nuAir;// Taylor microscale Reynolds number

      double depEff = 1 - 1 / (2.049 * pow(pow(ReLambda, 0.3) * Stk, 1.19) + 1);// deposition efficiency (E in Bailey 2018 Eq. 13)

      double ReLeaf = elementDiameter * MTot / nuAir;// leaf Reynolds number

      // Temporary fix to address limitations of Price 2017 model (correlation only valid for 400 < ReLeaf < 6000)
      if (ReLeaf > 6000.0) {
        ReLeaf = 6000.0;
      }
      if (ReLeaf < 400.0) {
        ReLeaf = 400.0;
      }

      double gam = -6.5e-5 * ReLeaf + 0.43;// non-impaction surface weighting factor
      // double gam = 0.1; // temporarily set to 1 because of problems (gam is becoming negative, but should be positive and ~0.1)

      double adjLAD = leafAreaDensitydep * (1 + gam);// LAD adjusted to include non-impaction surface

      (*parItr)->wdepos *= exp(-depEff * adjLAD * vegDistance / 2);// the /2 comes from Ross' G function, assuming uniform leaf orientation distribution

      // add deposition amount to the buffer (for parallelization)
      (*parItr)->dep_buffer_flag = true;
      (*parItr)->dep_buffer_cell.push_back(cellId_old);
      (*parItr)->dep_buffer_val.push_back((1 - (*parItr)->wdepos) * (*parItr)->m);
      // deposition->depcvol[cellId_old] += (1 - (*parItr)->wdepos) * (*parItr)->m;

      // Take deposited mass away from particle
      (*parItr)->m *= (*parItr)->wdepos;
      (*parItr)->m_kg *= (*parItr)->wdepos;

      //      std::cout << "DEPOSIT CHECKPOINT 0" << std::endl;
      // Add deposited mass to deposition bins (bins are on QES-Winds grid)
      //      std::cout << "size of depcvol = " << WGD->depcvol.size() << std::endl;
      // std::cout << "Mass being added: " << (1 - (*parItr)->wdepos) * (*parItr)->m << std::endl;

      //     std::cout << "DEPOSIT CHECKPOINT 1" << std::endl;

      // std::cout << "particle in homog. veg., mass: " << (*parItr)->m  << " wdepos = " << (*parItr)->wdepos << " depEff = " << depEff << " adjLAD = " << adjLAD << " vegDistance = " << vegDistance << " gam = " << gam << " ReLeaf = " << ReLeaf << " MTot = " << MTot << std::endl;
    } else {
      return;
    }

    // If particle mass drops below mass of a single particle, set it to zero and inactivate it
    double oneParMass = (*parItr)->rho * (1 / 6) * M_PI * pow((*parItr)->d_m, 3);
    if ((*parItr)->m_kg < oneParMass) {
      (*parItr)->m_kg = 0.0;
      (*parItr)->m = 0.0;
      (*parItr)->isActive = false;
    }

  }// if ( isActive == true )

  return;
}
