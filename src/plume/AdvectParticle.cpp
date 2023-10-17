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

void Plume::advectParticle(double timeRemainder, Particle *par_ptr, double boxSizeZ, WINDSGeneralData *WGD, TURBGeneralData *TGD)
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


  if (par_ptr->d > 0.0 && par_ptr->rho > rhoAir) {
    //  dimensionless grain diameter
    double dstar = (1.0E-6)*par_ptr->d * pow(9.81 / pow(nuAir, 2.0) * (par_ptr->rho / rhoAir - 1.), 1.0 / 3.0);
    // drag coefficent
    double Cd = (432.0 / pow(dstar, 3.0)) * pow(1.0 + 0.022 * pow(dstar, 3.0), 0.54)
                + 0.47 * (1.0 - exp(-0.15 * pow(dstar, 0.45)));
    // dimensionless settling velociy
    double wstar = pow((4.0 * dstar) / (3.0 * Cd), 0.5);
    // settling velocity
    vs = wstar * pow(9.81 * nuAir * (par_ptr->rho / rhoAir - 1.0), 1.0 / 3.0);
  }


  //  get the current isRogue and isActive information
  bool isRogue = par_ptr->isRogue;
  bool isActive = par_ptr->isActive;

  // getting the current position for where the particle is at for a given time
  // if it is the first time a particle is ever released, then the value is already set at the initial value
  // LA notes: technically this value is the old position to be overwritten with the new position.
  //  I've been tempted for a while to store both. Might have to for correctly implementing reflective building BCs
  double xPos = par_ptr->xPos;
  double yPos = par_ptr->yPos;
  double zPos = par_ptr->zPos;

  double disX = 0.0;
  double disY = 0.0;
  double disZ = 0.0;

  double uMean = 0.0;
  double vMean = 0.0;
  double wMean = 0.0;

  double uTot = 0.0;
  double vTot = 0.0;
  double wTot = 0.0;

  double flux_div_x = 0.0;
  double flux_div_y = 0.0;
  double flux_div_z = 0.0;

  double nuT = 0.0;
  // size_t cellIdx_old = interp->getCellId(xPos,yPos,zPos);

  // getting the initial position, for use in setting finished particles
  // double xPos_init = par_ptr->xPos_init;
  // double yPos_init = par_ptr->yPos_init;
  // double zPos_init = par_ptr->zPos_init;

  // grab the velFluct values.
  // LA notes: hmm, Bailey's code just starts out setting these values to zero,
  //  so the velFluct values are actually the old velFluct, that will be overwritten during the solver.
  //  velFluct_old and velFluct are probably identical and kind of redundant in this implementation
  //  but it shouldn't hurt anything for now, even if it is redundant
  //  besides, it will probably change a bit if we decide to change what is outputted on a regular, and on a debug basis
  double uFluct = par_ptr->uFluct;
  double vFluct = par_ptr->vFluct;
  double wFluct = par_ptr->wFluct;

  // get all other values for the particle
  // in this case this, all the old velocity fluctuations and old stress tensor values for the particle
  // LA note: also need to keep track of a delta_velFluct,
  //  but since delta_velFluct is never used, just set later on, it doesn't need grabbed as a value till later
  double uFluct_old = par_ptr->uFluct_old;
  double vFluct_old = par_ptr->vFluct_old;
  double wFluct_old = par_ptr->wFluct_old;

  double txx_old = par_ptr->txx_old;
  double txy_old = par_ptr->txy_old;
  double txz_old = par_ptr->txz_old;
  double tyy_old = par_ptr->tyy_old;
  double tyz_old = par_ptr->tyz_old;
  double tzz_old = par_ptr->tzz_old;


  // need to avoid current tau values going out of scope now that I've added the particle timestep loop
  // so initialize their values to the tau_old values. They will be overwritten with the Interperian grid value
  // at each iteration in the particle timestep loop
  double txx = txx_old;
  double txy = txy_old;
  double txz = txz_old;
  double tyy = tyy_old;
  double tyz = tyz_old;
  double tzz = tzz_old;

  // need to get the delta velFluct values right by doing the calculation inside the particle loop
  // these values go out of scope unless initialized here. So initialize them to zero (velFluct - velFluct_old = 0 right now)
  // they will be overwritten with the actual values in the particle timestep loop
  double delta_uFluct = 0.0;
  double delta_vFluct = 0.0;
  double delta_wFluct = 0.0;

  double CoEps = 1e-6;

  // time to do a particle timestep loop. start the time remainder as the simulation timestep.
  // at each particle timestep loop iteration the time remainder gets closer and closer to zero.
  // the particle timestep for a given particle timestep loop is either the time remainder or the value calculated
  // from the Courant Number, whichever is smaller.
  // particles can go inactive too, so need to use that as a condition to quit early too
  // LA important note: can't use the simulation timestep for the timestep remainder, the last simulation timestep
  //  is potentially smaller than the simulation timestep. So need to use the simTimes.at(nSimTimes-1)-simTimes.at(nSimTimes-2)
  //  for the last simulation timestep. The problem is that simTimes.at(nSimTimes-1) is greater than simTimes.at(nSimTimes-2) + sim_dt.
  // FMargairaz -> need clean-up the comment

  while (isActive && timeRemainder > 0.0) {

    /*
      now get the Lagrangian values for the current iteration from the Interperian grid
      will need to use the interp3D function
    */

    interp->interpValues(xPos, yPos, zPos, WGD, uMean, vMean, wMean, TGD, txx, txy, txz, tyy, tyz, tzz, flux_div_x, flux_div_y, flux_div_z, nuT, CoEps);

    // now need to call makeRealizable on tau
    makeRealizable(txx, txy, txz, tyy, tyz, tzz);

    // adjusting mean vertical velocity for settling velocity
    wMean -= vs;


    // now calculate the particle timestep using the courant number, the velocity fluctuation from the last time,
    // and the grid sizes. Uses timeRemainder as the timestep if it is smaller than the one calculated from the Courant number

    int cellId = interp->getCellId(xPos, yPos, zPos);

    double dWall = WGD->mixingLengths[cellId];
    double par_dt = calcCourantTimestep(dWall,
                                        std::abs(uMean) + std::abs(uFluct),
                                        std::abs(vMean) + std::abs(vFluct),
                                        std::abs(wMean) + std::abs(wFluct),
                                        timeRemainder);

    // std::cout << "par_dt = " << par_dt << std::endl;
    //  update the par_time, useful for debugging
    // par_time = par_time + par_dt;

    // now need to calculate the inverse values for tau
    // directly modifies the values of tau
    // LA warn: I just noticed that Bailey's code always leaves the last three components alone,
    //  never filled with the symmetrical tensor values. This seems fine for makeRealizable,
    //  but I wonder if it messes with the invert3 stuff since those values are used even though they are empty in his code
    //  going to send in 9 terms anyways to try to follow Bailey's method for now
    double lxx = txx;
    double lxy = txy;
    double lxz = txz;
    double lyx = txy;
    // double lyx = 0.0;
    double lyy = tyy;
    double lyz = tyz;
    double lzx = txz;
    double lzy = tyz;
    // double lzx = 0.0;
    // double lzy = 0.0;
    double lzz = tzz;
    isRogue = !invert3(lxx, lxy, lxz, lyx, lyy, lyz, lzx, lzy, lzz);
    if (isRogue) {
      // int cellIdNew = interp->getCellId(xPos,yPos,zPos);
      std::cerr << "ERROR in Matrix inversion of stress tensor" << std::endl;
      isActive = false;
      break;
    }

// these are the random numbers for each direction
// LA note: should be randn() matlab equivalent, which is a normally distributed random number
// LA future work: it is possible the rogue particles are caused by the random number generator stuff.
//  Need to look into it at some time.
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
    double dtxxdt = (txx - txx_old) / par_dt;
    double dtxydt = (txy - txy_old) / par_dt;
    double dtxzdt = (txz - txz_old) / par_dt;
    double dtyydt = (tyy - tyy_old) / par_dt;
    double dtyzdt = (tyz - tyz_old) / par_dt;
    double dtzzdt = (tzz - tzz_old) / par_dt;


    // now calculate and set the A and b matrices for an Ax = b
    // A = -I + 0.5*(-CoEps*L + dTdt*L )*par_dt;
    double A_11 = -1.0 + 0.50 * (-CoEps * lxx + lxx * dtxxdt + lxy * dtxydt + lxz * dtxzdt) * par_dt;
    double A_12 = 0.50 * (-CoEps * lxy + lxy * dtxxdt + lyy * dtxydt + lyz * dtxzdt) * par_dt;
    double A_13 = 0.50 * (-CoEps * lxz + lxz * dtxxdt + lyz * dtxydt + lzz * dtxzdt) * par_dt;

    double A_21 = 0.50 * (-CoEps * lxy + lxx * dtxydt + lxy * dtyydt + lxz * dtyzdt) * par_dt;
    double A_22 = -1.0 + 0.50 * (-CoEps * lyy + lxy * dtxydt + lyy * dtyydt + lyz * dtyzdt) * par_dt;
    double A_23 = 0.50 * (-CoEps * lyz + lxz * dtxydt + lyz * dtyydt + lzz * dtyzdt) * par_dt;

    double A_31 = 0.50 * (-CoEps * lxz + lxx * dtxzdt + lxy * dtyzdt + lxz * dtzzdt) * par_dt;
    double A_32 = 0.50 * (-CoEps * lyz + lxy * dtxzdt + lyy * dtyzdt + lyz * dtzzdt) * par_dt;
    double A_33 = -1.0 + 0.50 * (-CoEps * lzz + lxz * dtxzdt + lyz * dtyzdt + lzz * dtzzdt) * par_dt;

    // b = -vectFluct - 0.5*vecFluxDiv*par_dt - sqrt(CoEps*par_dt)*vecRandn;
    double b_11 = -uFluct_old - 0.50 * flux_div_x * par_dt - std::sqrt(CoEps * par_dt) * xRandn;
    double b_21 = -vFluct_old - 0.50 * flux_div_y * par_dt - std::sqrt(CoEps * par_dt) * yRandn;
    double b_31 = -wFluct_old - 0.50 * flux_div_z * par_dt - std::sqrt(CoEps * par_dt) * zRandn;

    // now prepare for the Ax=b calculation by calculating the inverted A matrix
    isRogue = !invert3(A_11, A_12, A_13, A_21, A_22, A_23, A_31, A_32, A_33);
    if (isRogue) {
      std::cerr << "ERROR in matrix inversion in Langevin equation" << std::endl;
      isActive = false;
      break;
    }
    // now do the Ax=b calculation using the inverted matrix (vecFluct = A*b)
    matmult(A_11, A_12, A_13, A_21, A_22, A_23, A_31, A_32, A_33, b_11, b_21, b_31, uFluct, vFluct, wFluct);


    // now check to see if the value is rogue or not
    if (std::abs(uFluct) >= vel_threshold || isnan(uFluct)) {
      std::cerr << "Particle # " << par_ptr->particleID << " is rogue, ";
      std::cerr << "uFluct = " << uFluct << ", CoEps = " << CoEps << std::endl;
      uFluct = 0.0;
      isActive = false;
      isRogue = true;
      break;
    }
    if (std::abs(vFluct) >= vel_threshold || isnan(vFluct)) {
      std::cerr << "Particle # " << par_ptr->particleID << " is rogue, ";
      std::cerr << "vFluct = " << vFluct << ", CoEps = " << CoEps << std::endl;
      vFluct = 0.0;
      isActive = false;
      isRogue = true;
      break;
    }
    if (std::abs(wFluct) >= vel_threshold || isnan(wFluct)) {
      std::cerr << "Particle # " << par_ptr->particleID << " is rogue, ";
      std::cerr << "wFluct = " << wFluct << ", CoEps = " << CoEps << std::endl;
      wFluct = 0.0;
      isActive = false;
      isRogue = true;
      break;
    }

    if (isRogue) {
      isActive = false;
      break;
    }
    // Pete: Do you need this???
    // ONLY if this should never happen....
    //    assert( isRogue == false );

    // now update the particle position for this iteration
    disX = (uMean + uFluct) * par_dt;
    disY = (vMean + vFluct) * par_dt;
    disZ = (wMean + wFluct) * par_dt;

    xPos = xPos + disX;
    yPos = yPos + disY;
    zPos = zPos + disZ;

    uTot = uMean + uFluct;
    vTot = vMean + vFluct;
    wTot = wMean + wFluct;

    // Deposit mass (vegetation only right now)
    if (par_ptr->depFlag && isActive) {
      depositParticle(xPos, yPos, zPos, disX, disY, disZ, uTot, vTot, wTot, txx, tyy, tzz, txz, txy, tyz, vs, CoEps, boxSizeZ, nuT, par_ptr, WGD, TGD);
    }

    // check and do wall (building and terrain) reflection (based in the method)
    if (isActive) {
      isActive = wallReflect->reflect(WGD, this, xPos, yPos, zPos, disX, disY, disZ, uFluct, vFluct, wFluct);
    }

    // now apply boundary conditions
    if (isActive) isActive = domainBC_x->enforce(xPos, uFluct);
    if (isActive) isActive = domainBC_y->enforce(yPos, vFluct);
    if (isActive) isActive = domainBC_z->enforce(zPos, wFluct);

    // now update the old values to be ready for the next particle time iteration
    // the current values are already set for the next iteration by the above calculations
    // note: it may look strange to set the old values to the current values, then to use these
    //  old values when setting the storage values, but that is what the old code was technically doing
    //  we are already done using the old _old values by this point and need to use the current ones
    // but we do need to set the delta velFluct values before setting the velFluct_old values to the current velFluct values
    // !!! this is extremely important for the next iteration to work accurately
    delta_uFluct = uFluct - uFluct_old;
    delta_vFluct = vFluct - vFluct_old;
    delta_wFluct = wFluct - wFluct_old;
    uFluct_old = uFluct;
    vFluct_old = vFluct;
    wFluct_old = wFluct;

    txx_old = txx;
    txy_old = txy;
    txz_old = txz;
    tyy_old = tyy;
    tyz_old = tyz;
    tzz_old = tzz;

    // cellIdx_old=cellIdx;

    // now set the time remainder for the next loop
    // if the par_dt calculated from the Courant Number is greater than the timeRemainder,
    // the function for calculating par_dt will use the timeRemainder for the output par_dt
    // so this should result in a timeRemainder of exactly zero, no need for a tol.
    timeRemainder = timeRemainder - par_dt;

  }// while( isActive == true && timeRemainder > 0.0 )

  // now update the old values and current values in the dispersion storage to be ready for the next iteration
  // also throw in the already calculated velFluct increment
  // notice that the values from the particle timestep loop are used directly here,
  //  just need to put the existing vals into storage
  // !!! this is extremely important for output and the next iteration to work correctly
  par_ptr->xPos = xPos;
  par_ptr->yPos = yPos;
  par_ptr->zPos = zPos;

  par_ptr->disX = disX;
  par_ptr->disY = disY;
  par_ptr->disZ = disZ;

  // par_ptr->uTot = uTot;
  // par_ptr->vTot = vTot;
  // par_ptr->wTot = wTot;

  par_ptr->CoEps = CoEps;

  par_ptr->uMean = uMean;
  par_ptr->vMean = vMean;
  par_ptr->wMean = wMean;

  par_ptr->uFluct = uFluct;
  par_ptr->vFluct = vFluct;
  par_ptr->wFluct = wFluct;

  // these are the current velFluct values by this point
  par_ptr->uFluct_old = uFluct_old;
  par_ptr->vFluct_old = vFluct_old;
  par_ptr->wFluct_old = wFluct_old;

  par_ptr->delta_uFluct = delta_uFluct;
  par_ptr->delta_vFluct = delta_vFluct;
  par_ptr->delta_wFluct = delta_wFluct;

  par_ptr->txx_old = txx_old;
  par_ptr->txy_old = txy_old;
  par_ptr->txz_old = txz_old;
  par_ptr->tyy_old = tyy_old;
  par_ptr->tyz_old = tyz_old;
  par_ptr->tzz_old = tzz_old;

  par_ptr->isRogue = isRogue;
  par_ptr->isActive = isActive;
}
