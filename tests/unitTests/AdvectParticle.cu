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

/** @file AdvectParticle.cu */

#include "util/vectorMath_CUDA.h"
#include "particle.cuh"

__device__ void advectParticle(float timeRemainder, particle &p)
{

  float rhoAir = 1.225;// in kg m^-3
  float nuAir = 1.506E-5;// in m^2 s^-1

  // set settling velocity
  //(*parItr)->setSettlingVelocity(rhoAir, nuAir);

  // get the current isRogue and isActive information
  bool isRogue = false;//(*parItr)->isRogue;
  bool isActive = true;//(*parItr)->isActive;

  // getting the current position for where the particle is at for a given time
  // if it is the first time a particle is ever released, then the value is already set at the initial value
  // LA notes: technically this value is the old position to be overwritten with the new position.
  //  I've been tempted for a while to store both. Might have to for correctly implementing reflective building BCs
  // vec3 pPos = { 0.0, 0.0, 0.0 };

  vec3 uMean = { 1.0, 2.0, 3.0 };

  vec3 flux_div = { 0.0, 0.0, 0.0 };

  // size_t cellIdx_old = interp->getCellId(xPos,yPos,zPos);

  // grab the velFluct values.
  // LA notes: hmm, Bailey's code just starts out setting these values to zero,
  //  so the velFluct values are actually the old velFluct, that will be overwritten during the solver.
  //  velFluct_old and velFluct are probably identical and kind of redundant in this implementation
  //  but it shouldn't hurt anything for now, even if it is redundant
  //  besides, it will probably change a bit if we decide to change what is outputted on a regular, and on a debug basis
  vec3 uFluct = { 0.0, 0.0, 0.0 };//(*parItr)->uFluct;

  // get all other values for the particle
  // in this case this, all the old velocity fluctuations and old stress tensor values for the particle
  // LA note: also need to keep track of a delta_velFluct,
  //  but since delta_velFluct is never used, just set later on, it doesn't need grabbed as a value till later
  vec3 uFluct_old = { 0.0, 0.0, 0.0 };//(*parItr)->uFluct_old;

  mat3sym tau_old = { 1, 0, 0, 1, 0, 1 };//(*parItr)->txx_old;

  // need to avoid current tau values going out of scope now that I've added the particle timestep loop
  // so initialize their values to the tau_old values. They will be overwritten with the Interperian grid value
  // at each iteration in the particle timestep loop
  mat3sym tau = tau_old;

  // need to get the delta velFluct values right by doing the calculation inside the particle loop
  // these values go out of scope unless initialized here. So initialize them to zero (velFluct - velFluct_old = 0 right now)
  // they will be overwritten with the actual values in the particle timestep loop
  vec3 uFluct_delta = { 0.0, 0.0, 0.0 };

  float CoEps = 1e-6;

  // time to do a particle timestep loop. start the time remainder as the simulation timestep.
  // at each particle timestep loop iteration the time remainder gets closer and closer to zero.
  // the particle timestep for a given particle timestep loop is either the time remainder or the value calculated
  // from the Courant Number, whichever is smaller.
  // particles can go inactive too, so need to use that as a condition to quit early too
  // LA important note: can't use the simulation timestep for the timestep remainder, the last simulation timestep
  //  is potentially smaller than the simulation timestep. So need to use the simTimes.at(nSimTimes-1)-simTimes.at(nSimTimes-2)
  //  for the last simulation timestep. The problem is that simTimes.at(nSimTimes-1) is greater than simTimes.at(nSimTimes-2) + sim_dt.
  // FMargairaz -> need clean-up the comment

  while (isActive == true && timeRemainder > 0.0) {

    /*
      now get the Lagrangian values for the current iteration from the Interperian grid
      will need to use the interp3D function
    */

    // interp->interpValues(xPos, yPos, zPos, WGD, uMean, vMean, wMean, TGD, txx, txy, txz, tyy, tyz, tzz, flux_div_x, flux_div_y, flux_div_z, CoEps);

    // now need to call makeRealizable on tau
    makeRealizable(10e-4, p.tau);

    // adjusting mean vertical velocity for settling velocity
    // wMean -= (*parItr)->vs;


    // now calculate the particle timestep using the courant number, the velocity fluctuation from the last time,
    // and the grid sizes. Uses timeRemainder as the timestep if it is smaller than the one calculated from the Courant number

    // int cellId = interp->getCellId(xPos, yPos, zPos);
    // double dWall = WGD->mixingLengths[cellId];
    // double par_dt = calcCourantTimestep(dWall,
    //                                     std::abs(uMean) + std::abs(uFluct),
    //                                     std::abs(vMean) + std::abs(vFluct),
    //                                     std::abs(wMean) + std::abs(wFluct),
    //                                     timeRemainder);
    float par_dt = 0.1;

    // update the par_time, useful for debugging
    // par_time = par_time + par_dt;

    // now need to calculate the inverse values for tau
    // directly modifies the values of tau
    // LA warn: I just noticed that Bailey's code always leaves the last three components alone,
    //  never filled with the symmetrical tensor values. This seems fine for makeRealizable,
    //  but I wonder if it messes with the invert3 stuff since those values are used even though they are empty in his code
    //  going to send in 9 terms anyways to try to follow Bailey's method for now

    mat3 L;
    L._11 = p.tau._11;
    L._12 = p.tau._12;
    L._13 = p.tau._13;
    L._21 = p.tau._12;
    L._22 = p.tau._22;
    L._23 = p.tau._23;
    L._31 = p.tau._13;
    L._32 = p.tau._23;
    L._33 = p.tau._33;

    isRogue = !invert3(L);
    if (isRogue == true) {
      // int cellIdNew = interp->getCellId(xPos,yPos,zPos);
      // std::cerr << "ERROR in Matrix inversion of stress tensor" << std::endl;
      isActive = false;
      break;
    }

    // these are the random numbers for each direction
    // LA note: should be randn() matlab equivalent, which is a normally distributed random number
    // LA future work: it is possible the rogue particles are caused by the random number generator stuff.
    //  Need to look into it at some time.
    // double xRandn = random::norRan();
    // double yRandn = random::norRan();
    // double zRandn = random::norRan();
    vec3 vRandn = { 0.0, 0.0, 0.0 };

    // now calculate a bunch of values for the current particle
    // calculate the time derivative of the stress tensor: (tau_current - tau_old)/dt
    mat3sym tau_ddt;
    tau_ddt._11 = (p.tau._11 - tau_old._11) / par_dt;
    tau_ddt._12 = (p.tau._12 - tau_old._12) / par_dt;
    tau_ddt._13 = (p.tau._13 - tau_old._13) / par_dt;
    tau_ddt._22 = (p.tau._22 - tau_old._22) / par_dt;
    tau_ddt._23 = (p.tau._23 - tau_old._23) / par_dt;
    tau_ddt._33 = (p.tau._33 - tau_old._33) / par_dt;


    // now calculate and set the A and b matrices for an Ax = b
    // A = -I + 0.5*(-CoEps*L + dTdt*L )*par_dt;
    mat3 A;
    A._11 = -1.0 + 0.50 * (-CoEps * L._11 + L._11 * tau_ddt._11 + L._12 * tau_ddt._12 + L._13 * tau_ddt._13) * par_dt;
    A._12 = -0.0 + 0.50 * (-CoEps * L._12 + L._12 * tau_ddt._11 + L._22 * tau_ddt._12 + L._23 * tau_ddt._13) * par_dt;
    A._13 = -0.0 + 0.50 * (-CoEps * L._13 + L._13 * tau_ddt._11 + L._23 * tau_ddt._12 + L._33 * tau_ddt._13) * par_dt;

    A._21 = -0.0 + 0.50 * (-CoEps * L._12 + L._11 * tau_ddt._12 + L._12 * tau_ddt._22 + L._13 * tau_ddt._23) * par_dt;
    A._22 = -1.0 + 0.50 * (-CoEps * L._22 + L._12 * tau_ddt._12 + L._22 * tau_ddt._22 + L._23 * tau_ddt._23) * par_dt;
    A._23 = -0.0 + 0.50 * (-CoEps * L._23 + L._13 * tau_ddt._12 + L._23 * tau_ddt._22 + L._33 * tau_ddt._23) * par_dt;

    A._31 = -0.0 + 0.50 * (-CoEps * L._13 + L._11 * tau_ddt._13 + L._12 * tau_ddt._23 + L._13 * tau_ddt._33) * par_dt;
    A._32 = -0.0 + 0.50 * (-CoEps * L._23 + L._12 * tau_ddt._13 + L._22 * tau_ddt._23 + L._23 * tau_ddt._33) * par_dt;
    A._33 = -1.0 + 0.50 * (-CoEps * L._33 + L._13 * tau_ddt._13 + L._23 * tau_ddt._23 + L._33 * tau_ddt._33) * par_dt;


    // b = -vectFluct - 0.5*vecFluxDiv*par_dt - sqrt(CoEps*par_dt)*vecRandn;
    vec3 b;
    b._1 = -uFluct_old._1 - 0.50 * flux_div._1 * par_dt - std::sqrt(CoEps * par_dt) * vRandn._1;
    b._2 = -uFluct_old._2 - 0.50 * flux_div._2 * par_dt - std::sqrt(CoEps * par_dt) * vRandn._2;
    b._3 = -uFluct_old._3 - 0.50 * flux_div._3 * par_dt - std::sqrt(CoEps * par_dt) * vRandn._3;

    // now prepare for the Ax=b calculation by calculating the inverted A matrix
    isRogue = !invert3(A);
    if (isRogue == true) {
      // std::cerr << "ERROR in matrix inversion in Langevin equation" << std::endl;
      isActive = false;
      break;
    }
    // now do the Ax=b calculation using the inverted matrix (vecFluct = A*b)
    matmult(A, b, uFluct);


    // now check to see if the value is rogue or not
    /*
  if (std::abs(uFluct) >= vel_threshold || isnan(uFluct)) {
    std::cerr << "Particle # " << (*parItr)->particleID << " is rogue, ";
    std::cerr << "responsible uFluct was \"" << uFluct << "\"" << std::endl;
    uFluct = 0.0;
    isActive = false;
    isRogue = true;
    break;
  }
  if (std::abs(vFluct) >= vel_threshold || isnan(vFluct)) {
    std::cerr << "Particle # " << (*parItr)->particleID << " is rogue, ";
    std::cerr << "responsible vFluct was \"" << vFluct << "\"" << std::endl;
    vFluct = 0.0;
    isActive = false;
    isRogue = true;
    break;
  }
  if (std::abs(wFluct) >= vel_threshold || isnan(wFluct)) {
    std::cerr << "Particle # " << (*parItr)->particleID << " is rogue, ";
    std::cerr << "responsible wFluct was \"" << wFluct << "\"" << std::endl;
    wFluct = 0.0;
    isActive = false;
    isRogue = true;
    break;
  }
  */

    if (isRogue == true) {
      isActive = false;
      break;
    }
    // Pete: Do you need this???
    // ONLY if this should never happen....
    //    assert( isRogue == false );


    pPos._1 = pPos._1 + (uMean._1 + uFluct._1) * par_dt;
    pPos._2 = pPos._2 + (uMean._2 + uFluct._2) * par_dt;
    pPos._3 = pPos._3 + (uMean._3 + uFluct._3) * par_dt;
    // now update the particle position for this iteration
    // double disX = (uMean + uFluct) * par_dt;
    // double disY = (vMean + vFluct) * par_dt;
    // double disZ = (wMean + wFluct) * par_dt;

    // xPos = xPos + disX;
    // yPos = yPos + disY;
    // zPos = zPos + disZ;

    /*
  // check and do wall (building and terrain) reflection (based in the method)
  if (isActive == true) {
    isActive = (this->*wallReflection)(WGD, xPos, yPos, zPos, disX, disY, disZ, uFluct, vFluct, wFluct, uFluct_old, vFluct_old, wFluct_old);
  }

  // now apply boundary conditions
  if (isActive == true) isActive = (this->*enforceWallBCs_x)(xPos, uFluct, uFluct_old, domainXstart, domainXend);
  if (isActive == true) isActive = (this->*enforceWallBCs_y)(yPos, vFluct, vFluct_old, domainYstart, domainYend);
  if (isActive == true) isActive = (this->*enforceWallBCs_z)(zPos, wFluct, wFluct_old, domainZstart, domainZend);
  */

    // now update the old values to be ready for the next particle time iteration
    // the current values are already set for the next iteration by the above calculations
    // note: it may look strange to set the old values to the current values, then to use these
    //  old values when setting the storage values, but that is what the old code was technically doing
    //  we are already done using the old _old values by this point and need to use the current ones
    // but we do need to set the delta velFluct values before setting the velFluct_old values to the current velFluct values
    // !!! this is extremely important for the next iteration to work accurately
    uFluct_delta._1 = uFluct._1 - uFluct_old._1;
    uFluct_delta._1 = uFluct._1 - uFluct_old._1;
    uFluct_delta._1 = uFluct._1 - uFluct_old._1;

    tau_old = tau;

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
  /*
    (*parItr)->xPos = xPos;
    (*parItr)->yPos = yPos;
    (*parItr)->zPos = zPos;

    (*parItr)->uMean = uMean;
    (*parItr)->vMean = vMean;
    (*parItr)->wMean = wMean;

    (*parItr)->uFluct = uFluct;
    (*parItr)->vFluct = vFluct;
    (*parItr)->wFluct = wFluct;

    // these are the current velFluct values by this point
    (*parItr)->uFluct_old = uFluct_old;
    (*parItr)->vFluct_old = vFluct_old;
    (*parItr)->wFluct_old = wFluct_old;

    (*parItr)->delta_uFluct = delta_uFluct;
    (*parItr)->delta_vFluct = delta_vFluct;
    (*parItr)->delta_wFluct = delta_wFluct;

    (*parItr)->txx_old = txx_old;
    (*parItr)->txy_old = txy_old;
    (*parItr)->txz_old = txz_old;
    (*parItr)->tyy_old = tyy_old;
    (*parItr)->tyz_old = tyz_old;
    (*parItr)->tzz_old = tzz_old;

    (*parItr)->isRogue = isRogue;
    (*parItr)->isActive = isActive;
  */

  return;
}
