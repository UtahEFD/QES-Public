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

/** @file AdvectParticle.cpp */

#include "Plume.hpp"

void Plume::advectParticle(double timeRemainder, Particle *p, double boxSizeZ, WINDSGeneralData *WGD, TURBGeneralData *TGD)
{
  /*
   * this function is advencing the particle -> status is returned in:
   * - parItr->isRogue
   * - parItr->isActive
   * this function take in a particle pointer
   * and does not do any manipulation on particleList
   */

  double rhoAir = 1.225;// in kg m^-3
  double nuAir = 1.506E-5;// in m^2 s^-1

  // settling velocity
  double vs = 0;

  if (p->d > 0.0 && p->rho > rhoAir) {
    //  dimensionless grain diameter
    double dstar = (1.0E-6) * p->d * pow(9.81 / pow(nuAir, 2.0) * (p->rho / rhoAir - 1.), 1.0 / 3.0);
    // drag coefficent
    double Cd = (432.0 / pow(dstar, 3.0)) * pow(1.0 + 0.022 * pow(dstar, 3.0), 0.54)
                + 0.47 * (1.0 - exp(-0.15 * pow(dstar, 0.45)));
    // dimensionless settling velociy
    double wstar = pow((4.0 * dstar) / (3.0 * Cd), 0.5);
    // settling velocity
    vs = wstar * pow(9.81 * nuAir * (p->rho / rhoAir - 1.0), 1.0 / 3.0);
  }

  // time to do a particle timestep loop. start the time remainder as the simulation timestep.
  // at each particle timestep loop iteration the time remainder gets closer and closer to zero.
  // the particle timestep for a given particle timestep loop is either the time remainder or the value calculated
  // from the Courant Number, whichever is smaller.
  // particles can go inactive too, so need to use that as a condition to quit early too
  // LA important note: can't use the simulation timestep for the timestep remainder, the last simulation timestep
  //  is potentially smaller than the simulation timestep. So need to use the simTimes.at(nSimTimes-1)-simTimes.at(nSimTimes-2)
  //  for the last simulation timestep. The problem is that simTimes.at(nSimTimes-1) is greater than simTimes.at(nSimTimes-2) + sim_dt.
  // FMargairaz -> need clean-up the comment

  while (p->isActive && timeRemainder > 0.0) {

    /*
      now get the Lagrangian values for the current iteration from the Interperian grid
      will need to use the interp3D function
    */

    interp->interpValues(WGD, p->xPos, p->yPos, p->zPos, p->uMean, p->vMean, p->wMean);

    // adjusting mean vertical velocity for settling velocity
    p->wMean -= vs;

    // now calculate the particle timestep using the courant number, the velocity fluctuation from the last time,
    // and the grid sizes. Uses timeRemainder as the timestep if it is smaller than the one calculated from the Courant number
    int cellId = interp->getCellId(p->xPos, p->yPos, p->zPos);

    double dWall = WGD->mixingLengths[cellId];
    double par_dt = calcCourantTimestep(dWall,
                                        std::abs(p->uMean) + std::abs(p->uFluct),
                                        std::abs(p->vMean) + std::abs(p->vFluct),
                                        std::abs(p->wMean) + std::abs(p->wFluct),
                                        timeRemainder);

    // std::cout << "par_dt = " << par_dt << std::endl;
    //  update the par_time, useful for debugging
    // par_time = par_time + par_dt;

    GLE_solver(p, par_dt, TGD);

    if (p->isRogue) {
      p->isActive = false;
      break;
    }
    // Pete: Do you need this???
    // ONLY if this should never happen....
    //    assert( isRogue == false );

    // now update the particle position for this iteration
    double disX = (p->uMean + p->uFluct) * par_dt;
    double disY = (p->vMean + p->vFluct) * par_dt;
    double disZ = (p->wMean + p->wFluct) * par_dt;

    p->xPos = p->xPos + disX;
    p->yPos = p->yPos + disY;
    p->zPos = p->zPos + disZ;

    double uTot = p->uMean + p->uFluct;
    double vTot = p->vMean + p->vFluct;
    double wTot = p->wMean + p->wFluct;

    // Deposit mass (vegetation only right now)
    if (p->depFlag && p->isActive) {
      depositParticle(p, disX, disY, disZ, uTot, vTot, wTot, vs, boxSizeZ, WGD, TGD);
    }

    // check and do wall (building and terrain) reflection (based in the method)
    if (p->isActive) {
      p->isActive = wallReflect->reflect(WGD, this, p->xPos, p->yPos, p->zPos, disX, disY, disZ, p->uFluct, p->vFluct, p->wFluct);
    }

    // now apply boundary conditions
    if (p->isActive) p->isActive = domainBC_x->enforce(p->xPos, p->uFluct);
    if (p->isActive) p->isActive = domainBC_y->enforce(p->yPos, p->vFluct);
    if (p->isActive) p->isActive = domainBC_z->enforce(p->zPos, p->wFluct);

    // now update the old values to be ready for the next particle time iteration
    // the current values are already set for the next iteration by the above calculations
    // !!! this is extremely important for the next iteration to work accurately
    p->delta_uFluct = p->uFluct - p->uFluct_old;
    p->delta_vFluct = p->vFluct - p->vFluct_old;
    p->delta_wFluct = p->wFluct - p->wFluct_old;

    p->uFluct_old = p->uFluct;
    p->vFluct_old = p->vFluct;
    p->wFluct_old = p->wFluct;

    // now set the time remainder for the next loop
    // if the par_dt calculated from the Courant Number is greater than the timeRemainder,
    // the function for calculating par_dt will use the timeRemainder for the output par_dt
    // so this should result in a timeRemainder of exactly zero, no need for a tol.
    timeRemainder = timeRemainder - par_dt;

  }// while( isActive == true && timeRemainder > 0.0 )
}

void Plume::GLE_solver(Particle *p, double &par_dt, TURBGeneralData *TGD)
{
  double txx = 0.0, txy = 0.0, txz = 0.0, tyy = 0.0, tyz = 0.0, tzz = 0.0;
  double flux_div_x = 0.0, flux_div_y = 0.0, flux_div_z = 0.0;

  interp->interpValues(TGD, p->xPos, p->yPos, p->zPos, txx, txy, txz, tyy, tyz, tzz, flux_div_x, flux_div_y, flux_div_z, p->nuT, p->CoEps);

  // now need to call makeRealizable on tau
  makeRealizable(txx, txy, txz, tyy, tyz, tzz);

  // now need to calculate the inverse values for tau
  // directly modifies the values of tau
  double lxx = txx;
  double lxy = txy;
  double lxz = txz;
  double lyx = txy;
  double lyy = tyy;
  double lyz = tyz;
  double lzx = txz;
  double lzy = tyz;
  double lzz = tzz;
  p->isRogue = !invert3(lxx, lxy, lxz, lyx, lyy, lyz, lzx, lzy, lzz);
  if (p->isRogue) {
    // int cellIdNew = interp->getCellId(xPos,yPos,zPos);
    std::cerr << "ERROR in Matrix inversion of stress tensor" << std::endl;
    p->isActive = false;
  }

// these are the random numbers for each direction
#ifdef _OPENMP
  double xRandn = threadRNG[omp_get_thread_num()]->norRan();
  double yRandn = threadRNG[omp_get_thread_num()]->norRan();
  double zRandn = threadRNG[omp_get_thread_num()]->norRan();
#else
  double xRandn = RNG->norRan();
  double yRandn = RNG->norRan();
  double zRandn = RNG->norRan();
#endif

  // now calculate a bunch of values for the current particle
  // calculate the time derivative of the stress tensor: (tau_current - tau_old)/dt
  double dtxxdt = (txx - p->txx) / par_dt;
  double dtxydt = (txy - p->txy) / par_dt;
  double dtxzdt = (txz - p->txz) / par_dt;
  double dtyydt = (tyy - p->tyy) / par_dt;
  double dtyzdt = (tyz - p->tyz) / par_dt;
  double dtzzdt = (tzz - p->tzz) / par_dt;

  // now calculate and set the A and b matrices for an Ax = b
  // A = -I + 0.5*(-CoEps*L + dTdt*L )*par_dt;
  double A_11 = -1.0 + 0.50 * (-p->CoEps * lxx + lxx * dtxxdt + lxy * dtxydt + lxz * dtxzdt) * par_dt;
  double A_12 = 0.50 * (-p->CoEps * lxy + lxy * dtxxdt + lyy * dtxydt + lyz * dtxzdt) * par_dt;
  double A_13 = 0.50 * (-p->CoEps * lxz + lxz * dtxxdt + lyz * dtxydt + lzz * dtxzdt) * par_dt;

  double A_21 = 0.50 * (-p->CoEps * lxy + lxx * dtxydt + lxy * dtyydt + lxz * dtyzdt) * par_dt;
  double A_22 = -1.0 + 0.50 * (-p->CoEps * lyy + lxy * dtxydt + lyy * dtyydt + lyz * dtyzdt) * par_dt;
  double A_23 = 0.50 * (-p->CoEps * lyz + lxz * dtxydt + lyz * dtyydt + lzz * dtyzdt) * par_dt;

  double A_31 = 0.50 * (-p->CoEps * lxz + lxx * dtxzdt + lxy * dtyzdt + lxz * dtzzdt) * par_dt;
  double A_32 = 0.50 * (-p->CoEps * lyz + lxy * dtxzdt + lyy * dtyzdt + lyz * dtzzdt) * par_dt;
  double A_33 = -1.0 + 0.50 * (-p->CoEps * lzz + lxz * dtxzdt + lyz * dtyzdt + lzz * dtzzdt) * par_dt;

  // b = -vectFluct - 0.5*vecFluxDiv*par_dt - sqrt(CoEps*par_dt)*vecRandn;
  double b_11 = -p->uFluct_old - 0.50 * flux_div_x * par_dt - std::sqrt(p->CoEps * par_dt) * xRandn;
  double b_21 = -p->vFluct_old - 0.50 * flux_div_y * par_dt - std::sqrt(p->CoEps * par_dt) * yRandn;
  double b_31 = -p->wFluct_old - 0.50 * flux_div_z * par_dt - std::sqrt(p->CoEps * par_dt) * zRandn;

  // now prepare for the Ax=b calculation by calculating the inverted A matrix
  p->isRogue = !invert3(A_11, A_12, A_13, A_21, A_22, A_23, A_31, A_32, A_33);
  if (p->isRogue) {
    std::cerr << "ERROR in matrix inversion in Langevin equation" << std::endl;
    p->isActive = false;
  }
  // now do the Ax=b calculation using the inverted matrix (vecFluct = A*b)
  matmult(A_11, A_12, A_13, A_21, A_22, A_23, A_31, A_32, A_33, b_11, b_21, b_31, p->uFluct, p->vFluct, p->wFluct);

  // now check to see if the value is rogue or not
  if (std::abs(p->uFluct) >= vel_threshold || isnan(p->uFluct)) {
    std::cerr << "Particle # " << p->particleID << " is rogue, ";
    std::cerr << "uFluct = " << p->uFluct << ", CoEps = " << p->CoEps << std::endl;
    p->uFluct = 0.0;
    p->isActive = false;
    p->isRogue = true;
  }
  if (std::abs(p->vFluct) >= vel_threshold || isnan(p->vFluct)) {
    std::cerr << "Particle # " << p->particleID << " is rogue, ";
    std::cerr << "vFluct = " << p->vFluct << ", CoEps = " << p->CoEps << std::endl;
    p->vFluct = 0.0;
    p->isActive = false;
    p->isRogue = true;
  }
  if (std::abs(p->wFluct) >= vel_threshold || isnan(p->wFluct)) {
    std::cerr << "Particle # " << p->particleID << " is rogue, ";
    std::cerr << "wFluct = " << p->wFluct << ", CoEps = " << p->CoEps << std::endl;
    p->wFluct = 0.0;
    p->isActive = false;
    p->isRogue = true;
  }

  // now update the old values to be ready for the next particle time iteration
  // the current values are already set for the next iteration by the above calculations
  p->txx = txx;
  p->txy = txy;
  p->txz = txz;
  p->tyy = tyy;
  p->tyz = tyz;
  p->tzz = tzz;
}