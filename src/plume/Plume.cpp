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

/** @file Plume.cpp */

#include "Plume.hpp"

Plume::Plume(WINDSGeneralData *WGD, TURBGeneralData *TGD)
  : particleList(0), allSources(0)
{
  // copy debug information
  doParticleDataOutput = false;// arguments->doParticleDataOutput;
  outputSimInfoFile = false;// arguments->doSimInfoFileOutput;
  outputFolder = "";// arguments->outputFolder;
  caseBaseName = "";// arguments->caseBaseName;
  debug = false;// arguments->debug;

  verbose = false;// arguments->verbose;

  // make local copies of the QES-Winds nVals for each dimension
  nx = WGD->nx;
  ny = WGD->ny;
  nz = WGD->nz;

  dx = WGD->dx;
  dy = WGD->dy;
  dz = WGD->dz;
  dxy = WGD->dxy;
}

Plume::Plume(PlumeInputData *PID, WINDSGeneralData *WGD, TURBGeneralData *TGD)
  : particleList(0), allSources(0)
{

  std::cout << "[Plume] \t Setting up simulation details " << std::endl;

  // copy debug information
  doParticleDataOutput = false;// arguments->doParticleDataOutput;
  outputSimInfoFile = false;// arguments->doSimInfoFileOutput;
  outputFolder = "";// arguments->outputFolder;
  caseBaseName = "";// arguments->caseBaseName;
  debug = false;// arguments->debug;

  verbose = false;// arguments->verbose;

  // make local copies of the QES-Winds nVals for each dimension
  nx = WGD->nx;
  ny = WGD->ny;
  nz = WGD->nz;

  dx = WGD->dx;
  dy = WGD->dy;
  dz = WGD->dz;
  dxy = WGD->dxy;

  // Create instance of Interpolation class
  std::cout << "[Plume] \t Interpolation Method set to: "
            << PID->plumeParams->interpMethod << std::endl;
  if (PID->plumeParams->interpMethod == "analyticalPowerLaw") {
    interp = new InterpPowerLaw(WGD, TGD, debug);
  } else if (PID->plumeParams->interpMethod == "nearestCell") {
    interp = new InterpNearestCell(WGD, TGD, debug);
  } else if (PID->plumeParams->interpMethod == "triLinear") {
    interp = new InterpTriLinear(WGD, TGD, debug);
  } else {
    std::cerr << "[ERROR] unknown interpolation method" << std::endl;
    exit(EXIT_FAILURE);
  }

#ifdef _OPENMP
  // if using openmp the RNG is not thread safe, use an array of RNG (one per thread)
#pragma omp parallel default(none) shared(threadRNG)
  {
#pragma omp master
    {
      threadRNG.resize(omp_get_num_threads(), nullptr);
    }
#pragma omp barrier
    int tID = omp_get_thread_num();
    threadRNG[tID] = new Random(long(time(nullptr)) ^ tID);
  }
#else
  RNG = RandomSingleton::getInstance();
#endif

  // get the domain start and end values, needed for wall boundary condition
  // application
  domainXstart = interp->xStart;
  domainXend = interp->xEnd;
  domainYstart = interp->yStart;
  domainYend = interp->yEnd;
  domainZstart = interp->zStart;
  domainZend = interp->zEnd;

  // Need dz for ground deposition
  double lBndz = PID->colParams->boxBoundsZ1;
  double uBndz = PID->colParams->boxBoundsZ2;
  int nBoxesZ = PID->colParams->nBoxesZ;
  boxSizeZ = (uBndz - lBndz) / (nBoxesZ);

  // make copies of important dispersion time variables
  sim_dt = PID->plumeParams->timeStep;

  // time variables
  simTimeStart = WGD->timestamp[0];
  simTimeCurr = simTimeStart;
  simTimeIdx = 0;

  // other important time variables not from dispersion
  CourantNum = PID->plumeParams->CourantNum;

  // set additional values from the input
  invarianceTol = PID->plumeParams->invarianceTol;
  updateFrequency_timeLoop = PID->plumeParams->updateFrequency_timeLoop;
  updateFrequency_particleLoop = PID->plumeParams->updateFrequency_particleLoop;

  // set the isRogueCount and isNotActiveCount to zero
  isRogueCount = 0;
  isNotActiveCount = 0;
  // set the particle counter to zero
  nParsReleased = 0;

  // get sources from input data and add them to the allSources vector
  // this also calls the many check and calc functions for all the input sources
  // !!! note that these check and calc functions have to be called here
  //  because each source requires extra data not found in the individual source
  //  data
  // !!! totalParsToRelease needs calculated very carefully here using
  // information from each of the sources
  if (PID->sourceParams) {
    getInputSources(PID);
  } else {
    std::cout << "[WARNING]\t no source parameters" << std::endl;
  }


  /* setup boundary condition functions */

  // now get the input boundary condition types from the inputs
  std::string xBCtype = PID->BCs->xBCtype;
  std::string yBCtype = PID->BCs->yBCtype;
  std::string zBCtype = PID->BCs->zBCtype;

  // now set the boundary condition function for the plume runs,
  // and check to make sure the input BCtypes are legitimate
  setBCfunctions(xBCtype, yBCtype, zBCtype);

  // now set the wall reflection function
  if (PID->BCs->wallReflection == "doNothing") {
    wallReflect = new WallReflection_DoNothing();
  } else if (PID->BCs->wallReflection == "setInactive") {
    wallReflect = new WallReflection_SetToInactive();
  } else if (PID->BCs->wallReflection == "stairstepReflection") {
    wallReflect = new WallReflection_StairStep();
  } else if (PID->BCs->wallReflection == "meshReflection") {
    if (WGD->mesh) {
      wallReflect = new WallReflection_TriMesh();
    } else {
      wallReflect = new WallReflection_StairStep();
    }
  } else {
    // this should not happend
    std::cerr << "[ERROR] unknown wall reflection setting" << std::endl;
    exit(EXIT_FAILURE);
  }

  deposition = new Deposition(WGD);
}

void Plume::run(QEStime loopTimeEnd, WINDSGeneralData *WGD, TURBGeneralData *TGD, std::vector<QESNetCDFOutput *> outputVec)
{
  auto startTimeAdvec = std::chrono::high_resolution_clock::now();

  // get the threshold velocity fluctuation to define rogue particles
  vel_threshold = 10.0 * getMaxVariance(TGD);

  // //////////////////////////////////////////
  // TIME Stepping Loop
  // for every simulation time step
  // //////////////////////////////////////////

  if (debug) {
    // start recording the amount of time it takes to perform the simulation
    // time integration loop
    timers.startNewTimer("simulation time integration loop");

    // start additional timers that need to be reset at different times during
    // the following loops LA future work: probably should wrap all of these in
    // a debug if statement
    timers.startNewTimer("advection loop");
    timers.startNewTimer("particle iteration");
  }

  QEStime nextUpdate = simTimeCurr + updateFrequency_timeLoop;
  float simTime = simTimeCurr - simTimeStart;

  std::cout << "[Plume] \t Advecting particles from " << simTimeCurr << " to " << loopTimeEnd << "\n";
  std::cout << "\t\t total run time = " << loopTimeEnd - simTimeCurr << " s \n";
  std::cout << "\t\t simulation time = " << simTime << " s (iteration = " << simTimeIdx << "). \n";
  std::cout << "\t\t Particles: Released = " << nParsReleased << " "
            << "Active = " << particleList.size() << std::endl;

  // LA note: that this loop goes from 0 to nTimes-2, not nTimes-1. This is
  // because
  //  a given time iteration is calculating where particles for the current time
  //  end up for the next time so in essence each time iteration is calculating
  //  stuff for one timestep ahead of the loop. This also comes out in making
  //  sure the numPar to release makes sense. A list of times from 0 to 10 with
  //  timestep of 1 means that nTimes is 11. So if the loop went from times 0 to
  //  10 in that case, if 10 pars were released each time, 110 particles, not
  //  100 particles would end up being released.
  // LA note on debug timers: because the loop is doing stuff for the next time,
  // and particles start getting released at time zero,
  //  this means that the updateFrequency needs to match with tStep+1, not
  //  tStep. At the same time, the current time output to consol output and to
  //  function calls need to also be set to tStep+1.
  // FMargairaz -> need clean-up
  while (simTimeCurr < loopTimeEnd) {
    // need to release new particles -> add new particles to the number to move
    int nParsToRelease = generateParticleList(simTime, WGD, TGD);
    if (debug) {
      std::cout << "Time = " << simTime << " s (iteration = " << simTimeIdx
                << "). Finished emitting particles "
                << "from " << allSources.size() << " sources. "
                << "Particles: New released = " << nParsToRelease << " "
                << "Total released = " << nParsReleased << "." << std::endl;
    }

    // number of active particle at the current time step.
    // list is scrubbed at the end of each time step (if flag turned true)
    bool needToScrub = false;

    // Move each particle for every simulation time step
    // Advection Loop

    // start recording the amount of time it takes to advect each set of
    // particles for a given simulation timestep,
    //  but only output the result when updateFrequency allows
    // LA future work: would love to put this into a debug if statement wrapper
    /* if( debug == true ) {
       if( (simTimeIdx+1) % updateFrequency_timeLoop == 0 || simTimeIdx == 0 ||
       simTimeIdx == nSimTimes-2 ) { timers.resetStoredTimer("advection loop");
       }
       }*/

    // the current time, updated in this loop with each new par_dt.
    // Will end at simTimes.at(simTimeIdx+1) at the end of this particle loop
    double timeRemainder = loopTimeEnd - simTimeCurr;
    if (timeRemainder > sim_dt) {
      timeRemainder = sim_dt;
    }

    // This the main loop over all active particles
    // All the particle here are active => no need to check
    //  for (auto parItr = particleList.begin(); parItr != particleList.end(); parItr++) {

    auto startTime = std::chrono::high_resolution_clock::now();
    // FM: openmp parallelization of the advection loop
#ifdef _OPENMP
    std::vector<Particle *> tmp(particleList.begin(), particleList.end());
#pragma omp parallel for default(none) shared(WGD, TGD, tmp, timeRemainder)
    for (auto k = 0u; k < tmp.size(); ++k) {
      // call to the main particle adection function (in separate file: AdvectParticle.cpp)
      advectParticle(timeRemainder, tmp[k], boxSizeZ, WGD, TGD);
    }//  END OF OPENMP WORK SHARE
#else
    for (auto &parItr : particleList) {
      //  call to the main particle adection function (in separate file: AdvectParticle.cpp)
      advectParticle(timeRemainder, parItr, boxSizeZ, WGD, TGD);
    }// end of loop
#endif

    //  flush deposition buffer
    for (auto &parItr : particleList) {
      if (parItr->dep_buffer_flag) {
        for (auto n = 0u; n < parItr->dep_buffer_cell.size(); ++n) {
          deposition->depcvol[parItr->dep_buffer_cell[n]] += parItr->dep_buffer_val[n];
        }
        parItr->dep_buffer_flag = false;
        parItr->dep_buffer_cell.clear();
        parItr->dep_buffer_val.clear();
      }
    }

    for (auto &parItr : particleList) {
      // now update the isRogueCount and isNotActiveCount
      if (parItr->isRogue) {
        isRogueCount = isRogueCount + 1;
      }
      if (!parItr->isActive) {
        isNotActiveCount = isNotActiveCount + 1;
        needToScrub = true;
      }
    }// end of loop for (parItr == particleList.begin(); parItr !=
    // particleList.end() ; parItr++ )

    // incrementation of time and timestep
    simTimeIdx++;
    simTimeCurr += timeRemainder;
    simTime = simTimeCurr - simTimeStart;

    // netcdf output for a given simulation timestep
    // note that the first time is already output, so this is the time the loop
    // iteration
    //  is calculating, not the input time to the loop iteration
    for (auto &id_out : outputVec) {
      id_out->save(simTimeCurr);
    }

    // For all particles that need to be removed from the particle
    // advection, remove them now
    if (needToScrub) {
      scrubParticleList();
    }
    // output the time, isRogueCount, and isNotActiveCount information for all
    // simulations, but only when the updateFrequency allows
    if (simTimeCurr >= nextUpdate || (simTimeCurr == loopTimeEnd)) {
      if (verbose) {
        std::cout << "Time = " << simTimeCurr << " (sim time = " << simTime << " s, iteration = " << simTimeIdx << "). "
                  << "Particles: Released = " << nParsReleased << " "
                  << "Active = " << particleList.size() << " "
                  << "Rogue = " << isRogueCount << "." << std::endl;
      } else {
        std::cout << "Time = " << simTimeCurr << " (sim time = " << simTime << " s, iteration = " << simTimeIdx << "). "
                  << "Particles: Released = " << nParsReleased << " "
                  << "Active = " << particleList.size() << "." << std::endl;
      }
      nextUpdate += (float)updateFrequency_timeLoop;
      // output advection loop runtime if in debug mode
      if (debug) {
        timers.printStoredTime("advection loop");
      }
    }

  }// end of time loop

  std::cout << "[Plume] \t End of particles advection at Time = " << simTimeCurr
            << " s (iteration = " << simTimeIdx << "). \n";
  std::cout << "\t\t Particles: Released = " << nParsReleased << " "
            << "Active = " << particleList.size() << "." << std::endl;

  // DEBUG - get the amount of time it takes to perform the simulation time
  // integration loop
  if (debug) {
    std::cout << "finished time integration loop" << std::endl;
    // Print out elapsed execution time
    timers.printStoredTime("simulation time integration loop");
  }

  auto endTimerAdvec = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> Elapsed = endTimerAdvec - startTimeAdvec;

  std::cout << "[QES-Plume]\t Advection.\n";
  std::cout << "\t\t elapsed time: " << Elapsed.count() << " s" << std::endl;
}

double Plume::calcCourantTimestep(const double &u,
                                  const double &v,
                                  const double &w,
                                  const double &timeRemainder)
{
  // set the output dt_par val to the timeRemainder
  // then if any of the Courant number values end up smaller, use that value
  // instead
  double dt_par = timeRemainder;

  // LA-note: what to do if the velocity fluctuation is zero?
  //  I forced them to zero to check dt_x, dt_y, and dt_z would get values of
  //  "inf". It ends up keeping dt_par as the timeRemainder
  double dt_x = CourantNum * dx / std::abs(u);
  double dt_y = CourantNum * dy / std::abs(v);
  double dt_z = CourantNum * dz / std::abs(w);

  // now find which dt is the smallest one of the Courant Number ones, or the
  // timeRemainder if any dt is smaller than the already chosen output value set
  // that dt to the output dt value
  if (dt_x < dt_par) {
    dt_par = dt_x;
  }
  if (dt_y < dt_par) {
    dt_par = dt_y;
  }
  if (dt_z < dt_par) {
    dt_par = dt_z;
  }

  return dt_par;
}

double Plume::calcCourantTimestep(const double &d,
                                  const double &u,
                                  const double &v,
                                  const double &w,
                                  const double &timeRemainder)
{
  // if the Courant Number is set to 0.0, we want to exit using the
  // timeRemainder (first time through that is the simTime)
  if (CourantNum == 0.0) {
    return timeRemainder;
  }

  double min_ds = std::min(dxy, dz);
  // double max_u = std::max({ u, v, w });
  double max_u = sqrt(u * u + v * v + w * w);
  double CN = 0.0;

  /*
    if (d > 6.0 * min_ds) {
    //CN = 1.0;
    return timeRemainder;
  } else if (d > 4.0 * min_ds) {
    CN = 0.5;
  } else if (d > 2.0 * min_ds) {
    CN = 1.0 / 3.0;
  } else {
    CN = 0.2;
  }
  */

  if (d > 6.0 * max_u * sim_dt) {
    // CN = 1.0;
    return timeRemainder;
  } else if (d > 3.0 * max_u * sim_dt) {
    CN = std::min(2.0 * CourantNum, 1.0);
  } else {
    CN = CourantNum;
  }

  return std::min(timeRemainder, CN * min_ds / max_u);
}

void Plume::getInputSources(PlumeInputData *PID)
{
  int numSources_Input = PID->sourceParams->sources.size();

  if (numSources_Input == 0) {
    std::cerr << "[ERROR]\t Plume::getInputSources: \n\t\t there are no sources in the input file!" << std::endl;
    exit(1);
  }

  // start at zero particles to release and increment as the number per source
  // totalParsToRelease = 0;

  for (auto s : PID->sourceParams->sources) {
    // now do anything that is needed to the source via the pointer
    s->checkReleaseInfo(PID->plumeParams->timeStep, PID->plumeParams->simDur);
    s->checkPosInfo(domainXstart, domainXend, domainYstart, domainYend, domainZstart, domainZend);

    // now determine the number of particles to release for the source and update the overall count
    totalParsToRelease += s->getNumParticles();

    // add source into the vector of sources
    allSources.push_back(new Source((int)allSources.size(), s));
  }
}

void Plume::addSources(std::vector<Source *> &newSources)
{
  allSources.insert(allSources.end(), newSources.begin(), newSources.end());
}


int Plume::generateParticleList(float currentTime, WINDSGeneralData *WGD, TURBGeneralData *TGD)
{

  // Add new particles now
  // - walk over all sources and add the emitted particles from

  std::list<Particle *> nextSetOfParticles;
  int numNewParticles = 0;
  for (auto source : allSources) {
    numNewParticles += source->emitParticles((float)sim_dt, currentTime, nextSetOfParticles);
  }

  setParticleVals(WGD, TGD, nextSetOfParticles);

  // append all the new particles on to the big particle
  // advection list
  particleList.insert(particleList.end(), nextSetOfParticles.begin(), nextSetOfParticles.end());

  // now calculate the number of particles to release for this timestep
  return numNewParticles;
}

void Plume::scrubParticleList()
{
  for (auto parItr = particleList.begin(); parItr != particleList.end();) {
    if (!(*parItr)->isActive) {
      delete *parItr;
      parItr = particleList.erase(parItr);
    } else {
      ++parItr;
    }
  }
}

void Plume::setParticleVals(WINDSGeneralData *WGD, TURBGeneralData *TGD, std::list<Particle *> newParticles)
{
  // at this time, should be a list of each and every particle that exists at
  // the given time particles and sources can potentially be added to the list
  // elsewhere
  // for (auto parItr = newParticles.begin(); parItr != newParticles.end(); parItr++) {
  std::vector<Particle *> tmp(newParticles.begin(), newParticles.end());
  for (auto &parItr : tmp) {
    // set particle ID (use global particle counter)
    parItr->particleID = nParsReleased;
    nParsReleased++;
  }

  // #pragma omp parallel for default(none) shared(WGD, TGD, tmp)
  // for (auto parItr = tmp.begin(); parItr != tmp.end(); parItr++) {
  //  set particle ID (use global particle counter)
  //(*parItr)->particleID = nParsReleased;
  //
#pragma omp parallel for default(none) shared(WGD, TGD, tmp)
  for (auto k = 0u; k < tmp.size(); ++k) {
    // set particle ID (use global particle counter)
    Particle *par_ptr = tmp[k];
    // nParsReleased++;

    // set the positions to be used by the simulation to the initial positions
    par_ptr->xPos = par_ptr->xPos_init;
    par_ptr->yPos = par_ptr->yPos_init;
    par_ptr->zPos = par_ptr->zPos_init;

    // get the sigma values from the QES grid for the particle value
    double sig_x, sig_y, sig_z;
    // get the tau values from the QES grid for the particle value
    double txx, txy, txz, tyy, tyz, tzz;

    interp->interpInitialValues(par_ptr->xPos,
                                par_ptr->yPos,
                                par_ptr->zPos,
                                TGD,
                                sig_x,
                                sig_y,
                                sig_z,
                                txx,
                                txy,
                                txz,
                                tyy,
                                tyz,
                                tzz);

    // now set the initial velocity fluctuations for the particle
    // The  sqrt of the variance is to match Bailey's code
    // normally distributed random number
#ifdef _OPENMP
    par_ptr->uFluct = sig_x * threadRNG[omp_get_thread_num()]->norRan();
    par_ptr->vFluct = sig_y * threadRNG[omp_get_thread_num()]->norRan();
    par_ptr->wFluct = sig_z * threadRNG[omp_get_thread_num()]->norRan();
#else
    par_ptr->uFluct = sig_x * RNG->norRan();
    par_ptr->vFluct = sig_y * RNG->norRan();
    par_ptr->wFluct = sig_z * RNG->norRan();
#endif


    // set the initial values for the old velFluct values
    par_ptr->uFluct_old = par_ptr->uFluct;
    par_ptr->vFluct_old = par_ptr->vFluct;
    par_ptr->wFluct_old = par_ptr->wFluct;

    // now need to call makeRealizable on tau
    makeRealizable(txx, txy, txz, tyy, tyz, tzz);

    // set tau_old to the interpolated values for each position
    par_ptr->txx_old = txx;
    par_ptr->txy_old = txy;
    par_ptr->txz_old = txz;
    par_ptr->tyy_old = tyy;
    par_ptr->tyz_old = tyz;
    par_ptr->tzz_old = tzz;

    // set delta_velFluct values to zero for now
    par_ptr->delta_uFluct = 0.0;
    par_ptr->delta_vFluct = 0.0;
    par_ptr->delta_wFluct = 0.0;

    // set isRogue to false and isActive to true for each particle
    // isActive = true as particle relased is active immediately
    par_ptr->isRogue = false;
    par_ptr->isActive = true;

    int cellIdNew = interp->getCellId(par_ptr->xPos, par_ptr->yPos, par_ptr->zPos);
    if ((WGD->icellflag[cellIdNew] == 0) && (WGD->icellflag[cellIdNew] == 2)) {
      // std::cerr << "WARNING invalid initial position" << std::endl;
      par_ptr->isActive = false;
    }

    double det = txx * (tyy * tzz - tyz * tyz) - txy * (txy * tzz - tyz * txz) + txz * (txy * tyz - tyy * txz);
    if (std::abs(det) < 1e-10) {
      // std::cerr << "WARNING invalid position stress" << std::endl;
      par_ptr->isActive = false;
    }
  }
}

double Plume::getMaxVariance(const TURBGeneralData *TGD)
{
  // set thoe initial maximum value to a very small number. The idea is to go through each value of the data,
  // setting the current value to the max value each time the current value is bigger than the old maximum value
  double maximumVal = -10e-10;

  // go through each vector to find the maximum value
  // each one could potentially be different sizes if the grid is not 3D
  for (float it : TGD->txx) {
    if (std::sqrt(std::abs(it)) > maximumVal) {
      maximumVal = std::sqrt(std::abs(it));
    }
  }
  for (float it : TGD->tyy) {
    if (std::sqrt(std::abs(it)) > maximumVal) {
      maximumVal = std::sqrt(std::abs(it));
    }
  }
  for (float it : TGD->tzz) {
    if (std::sqrt(std::abs(it)) > maximumVal) {
      maximumVal = std::sqrt(std::abs(it));
    }
  }

  return maximumVal;
}

void Plume::calcInvariants(const double &txx,
                           const double &txy,
                           const double &txz,
                           const double &tyy,
                           const double &tyz,
                           const double &tzz,
                           double &invar_xx,
                           double &invar_yy,
                           double &invar_zz)
{
  // since the x doesn't depend on itself, can just set the output without doing
  // any temporary variables (copied from Bailey's code)
  invar_xx = txx + tyy + tzz;
  invar_yy = txx * tyy + txx * tzz + tyy * tzz - txy * txy - txz * txz - tyz * tyz;
  invar_zz = txx * (tyy * tzz - tyz * tyz) - txy * (txy * tzz - tyz * txz) + txz * (txy * tyz - tyy * txz);
}

void Plume::makeRealizable(double &txx,
                           double &txy,
                           double &txz,
                           double &tyy,
                           double &tyz,
                           double &tzz)
{
  // first calculate the invariants and see if they are already realizable
  // the calcInvariants function modifies the values directly, so they always
  // need initialized to something before being sent into said function to be
  // calculated
  double invar_xx = 0.0;
  double invar_yy = 0.0;
  double invar_zz = 0.0;
  calcInvariants(txx, txy, txz, tyy, tyz, tzz, invar_xx, invar_yy, invar_zz);

  if (invar_xx > invarianceTol && invar_yy > invarianceTol && invar_zz > invarianceTol) {
    return;// tau is already realizable
  }

  // since tau is not already realizable, need to make it realizeable
  // start by making a guess of ks, the subfilter scale tke
  // I keep wondering if we can use the input Turb->tke for this or if we should
  // leave it as is
  double b = 4.0 / 3.0 * (txx + tyy + tzz);// also 4.0/3.0*invar_xx
  double c = txx * tyy + txx * tzz + tyy * tzz - txy * txy - txz * txz - tyz * tyz;// also invar_yy
  double ks = 1.01 * (-b + std::sqrt(b * b - 16.0 / 3.0 * c)) / (8.0 / 3.0);

  // if the initial guess is bad, use the straight up invar_xx value
  if (ks < invarianceTol || isnan(ks)) {
    ks = 0.5 * std::abs(txx + tyy + tzz);// also 0.5*abs(invar_xx)
  }

  // to avoid increasing tau by more than ks increasing by 0.05%, use a separate
  // stress tensor and always increase the separate stress tensor using the
  // original stress tensor, only changing ks for each iteration notice that
  // through all this process, only the diagonals are really increased by a
  // value of 0.05% of the subfilter tke ks start by initializing the separate
  // stress tensor
  double txx_new = txx + 2.0 / 3.0 * ks;
  double txy_new = txy;
  double txz_new = txz;
  double tyy_new = tyy + 2.0 / 3.0 * ks;
  double tyz_new = tyz;
  double tzz_new = tzz + 2.0 / 3.0 * ks;

  calcInvariants(txx_new, txy_new, txz_new, tyy_new, tyz_new, tzz_new, invar_xx, invar_yy, invar_zz);

  // now adjust the diagonals by 0.05% of the subfilter tke, which is ks, till
  // tau is realizable or if too many iterations go on, give a warning. I've had
  // trouble with this taking too long
  //  if it isn't realizable, so maybe another approach for when the iterations
  //  are reached might be smart
  int iter = 0;
  while ((invar_xx < invarianceTol || invar_yy < invarianceTol || invar_zz < invarianceTol) && iter < 1000) {
    iter = iter + 1;

    // increase subfilter tke by 5%
    ks = ks * 1.050;

    // note that the right hand side is not tau_new, to force tau to only
    // increase by increasing ks
    txx_new = txx + 2.0 / 3.0 * ks;
    tyy_new = tyy + 2.0 / 3.0 * ks;
    tzz_new = tzz + 2.0 / 3.0 * ks;

    calcInvariants(txx_new, txy_new, txz_new, tyy_new, tyz_new, tzz_new, invar_xx, invar_yy, invar_zz);
  }

  if (iter == 999) {
    std::cout << "WARNING (Plume::makeRealizable): unable to make stress "
                 "tensor realizble.";
  }

  // now set the output actual stress tensor using the separate temporary stress
  // tensor
  txx = txx_new;
  txy = txy_new;
  txz = txz_new;
  tyy = tyy_new;
  tyz = tyz_new;
  tzz = tzz_new;
}

bool Plume::invert3(double &A_11,
                    double &A_12,
                    double &A_13,
                    double &A_21,
                    double &A_22,
                    double &A_23,
                    double &A_31,
                    double &A_32,
                    double &A_33)
{
  // note that with Bailey's code, the input A_21, A_31, and A_32 are zeros even
  // though they are used here at least when using this on tau to calculate the
  // inverse stress tensor. This is not true when calculating the inverse A
  // matrix for the Ax=b calculation

  // now calculate the determinant
  double det = A_11 * (A_22 * A_33 - A_23 * A_32) - A_12 * (A_21 * A_33 - A_23 * A_31) + A_13 * (A_21 * A_32 - A_22 * A_31);

  // check for near zero value determinants
  // LA future work: I'm still debating whether this warning needs to be limited
  // by the updateFrequency information
  //  if so, how would we go about limiting that info? Would probably need to
  //  make the loop counter variables actual data members of the class
  if (std::abs(det) < 1e-10) {
    std::cerr << "WARNING (Plume::invert3): matrix nearly singular" << std::endl;
    std::cerr << "abs(det) = \"" << std::abs(det) << "\",  A_11 =  \"" << A_11 << "\", A_12 = \"" << A_12 << "\", A_13 = \""
              << A_13 << "\", A_21 = \"" << A_21 << "\", A_22 = \"" << A_22 << "\", A_23 = \"" << A_23 << "\", A_31 = \""
              << A_31 << "\" A_32 = \"" << A_32 << "\", A_33 = \"" << A_33 << "\"" << std::endl;

    det = 10e10;
    A_11 = 0.0;
    A_12 = 0.0;
    A_13 = 0.0;
    A_21 = 0.0;
    A_22 = 0.0;
    A_23 = 0.0;
    A_31 = 0.0;
    A_32 = 0.0;
    A_33 = 0.0;

    return false;

  } else {

    // calculate the inverse. Because the inverted matrix depends on other
    // components of the matrix,
    //  need to make a temporary value till all the inverted parts of the matrix
    //  are set
    double Ainv_11 = (A_22 * A_33 - A_23 * A_32) / det;
    double Ainv_12 = -(A_12 * A_33 - A_13 * A_32) / det;
    double Ainv_13 = (A_12 * A_23 - A_22 * A_13) / det;
    double Ainv_21 = -(A_21 * A_33 - A_23 * A_31) / det;
    double Ainv_22 = (A_11 * A_33 - A_13 * A_31) / det;
    double Ainv_23 = -(A_11 * A_23 - A_13 * A_21) / det;
    double Ainv_31 = (A_21 * A_32 - A_31 * A_22) / det;
    double Ainv_32 = -(A_11 * A_32 - A_12 * A_31) / det;
    double Ainv_33 = (A_11 * A_22 - A_12 * A_21) / det;

    // now set the input reference A matrix to the temporary inverted A matrix
    // values
    A_11 = Ainv_11;
    A_12 = Ainv_12;
    A_13 = Ainv_13;
    A_21 = Ainv_21;
    A_22 = Ainv_22;
    A_23 = Ainv_23;
    A_31 = Ainv_31;
    A_32 = Ainv_32;
    A_33 = Ainv_33;

    return true;
  }
}

void Plume::matmult(const double &A_11,
                    const double &A_12,
                    const double &A_13,
                    const double &A_21,
                    const double &A_22,
                    const double &A_23,
                    const double &A_31,
                    const double &A_32,
                    const double &A_33,
                    const double &b_11,
                    const double &b_21,
                    const double &b_31,
                    double &x_11,
                    double &x_21,
                    double &x_31)
{
  // since the x doesn't depend on itself, can just set the output without doing
  // any temporary variables

  // now calculate the Ax=b x value from the input inverse A matrix and b matrix
  x_11 = b_11 * A_11 + b_21 * A_12 + b_31 * A_13;
  x_21 = b_11 * A_21 + b_21 * A_22 + b_31 * A_23;
  x_31 = b_11 * A_31 + b_21 * A_32 + b_31 * A_33;
}

void Plume::setBCfunctions(const std::string &xBCtype,
                           const std::string &yBCtype,
                           const std::string &zBCtype)
{
  // the idea is to use the string input BCtype to determine which boundary
  // condition function to use later in the program, and to have a function
  // pointer point to the required function. I learned about pointer functions
  // from this website:
  // https://www.learncpp.com/cpp-tutorial/78-function-pointers/

  // output some debug information
  if (debug) {
    std::cout << "xBCtype = \"" << xBCtype << "\"" << std::endl;
    std::cout << "yBCtype = \"" << yBCtype << "\"" << std::endl;
    std::cout << "zBCtype = \"" << zBCtype << "\"" << std::endl;
  }

  if (xBCtype == "exiting") {
    domainBC_x = new DomainBC_exiting(domainXstart, domainXend);
  } else if (xBCtype == "periodic") {
    domainBC_x = new DomainBC_periodic(domainXstart, domainXend);
  } else if (xBCtype == "reflection") {
    domainBC_x = new DomainBC_reflection(domainXstart, domainXend);
  } else {
    std::cerr << "[ERROR]\tPlume::setBCfunctions() input xBCtype = "
              << xBCtype
              << " has not been implemented in code! Available xBCtypes are "
              << "'exiting', 'periodic', 'reflection'" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (yBCtype == "exiting") {
    domainBC_y = new DomainBC_exiting(domainYstart, domainYend);
  } else if (yBCtype == "periodic") {
    domainBC_y = new DomainBC_periodic(domainYstart, domainYend);
  } else if (yBCtype == "reflection") {
    domainBC_y = new DomainBC_reflection(domainYstart, domainYend);
  } else {
    std::cerr << "[ERROR]\tPlume::setBCfunctions() input yBCtype = "
              << yBCtype
              << " has not been implemented in code! Available yBCtypes are "
              << "'exiting', 'periodic', 'reflection'" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (zBCtype == "exiting") {
    domainBC_z = new DomainBC_exiting(domainZstart, domainZend);
  } else if (zBCtype == "periodic") {
    domainBC_z = new DomainBC_periodic(domainZstart, domainZend);
  } else if (zBCtype == "reflection") {
    domainBC_z = new DomainBC_reflection(domainZstart, domainZend);
  } else {
    std::cerr << "[ERROR]\tPlume::setBCfunctions() input zBCtype = "
              << zBCtype
              << " has not been implemented in code! Available zBCtypes are "
              << "'exiting', 'periodic', 'reflection'" << std::endl;
    exit(EXIT_FAILURE);
  }
}
