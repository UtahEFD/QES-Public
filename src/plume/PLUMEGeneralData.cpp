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

/** @file PLUMEGeneralData.cpp */

#include "PLUMEGeneralData.h"
#include <queue>

PLUMEGeneralData::PLUMEGeneralData(WINDSGeneralData *WGD, TURBGeneralData *TGD)
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

PLUMEGeneralData::PLUMEGeneralData(PlumeInputData *PID, WINDSGeneralData *WGD, TURBGeneralData *TGD)
{
  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "[QES-Plume]\t Initialization of plume model...\n";

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
  std::cout << "[QES-Plume]\t Interpolation Method set to: "
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

  // get the domain start and end values, needed for wall boundary condition
  // application
  domainXstart = interp->xStart;
  domainXend = interp->xEnd;
  domainYstart = interp->yStart;
  domainYend = interp->yEnd;
  domainZstart = interp->zStart;
  domainZend = interp->zEnd;

  // now set the wall reflection function
  std::cout << "[QES-Plume]\t Wall Reflection Method set to: "
            << PID->BCs->wallReflection << std::endl;
  if (PID->BCs->wallReflection == "doNothing") {
    wallReflect = new WallReflection_DoNothing(interp);
  } else if (PID->BCs->wallReflection == "setInactive") {
    wallReflect = new WallReflection_SetToInactive(interp);
  } else if (PID->BCs->wallReflection == "stairstepReflection") {
    wallReflect = new WallReflection_StairStep(interp);
  } else if (PID->BCs->wallReflection == "meshReflection") {
    if (WGD->mesh) {
      wallReflect = new WallReflection_TriMesh(interp);
    } else {
      wallReflect = new WallReflection_StairStep(interp);
    }
  } else {
    // this should not happend
    std::cerr << "[ERROR] unknown wall reflection setting" << std::endl;
    exit(EXIT_FAILURE);
  }

  /* setup boundary condition functions */
  // now get the input boundary condition types from the inputs
  std::string xBCtype = PID->BCs->xBCtype;
  std::string yBCtype = PID->BCs->yBCtype;
  std::string zBCtype = PID->BCs->zBCtype;

  // now set the boundary condition function for the plume runs,
  // and check to make sure the input BCtypes are legitimate
  setBCfunctions(xBCtype, yBCtype, zBCtype);

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
  isReleasedCount = 0;
  isActiveCount = 0;
  // isNotActiveCount = 0;

  //  set the particle counter to zero
  // nParsReleased = 0;

  std::cout << "[QES-Plume]\t Initializing Particle Models: " << std::endl;
  for (auto p : PID->particleParams->particles) {
    for (auto s : p->sources) {
      //  now do anything that is needed to the source via the pointer
      s->checkReleaseInfo(PID->plumeParams->timeStep, PID->plumeParams->simDur);
      s->checkPosInfo(domainXstart, domainXend, domainYstart, domainYend, domainZstart, domainZend);
    }
    models[p->tag] = p->create();
    models[p->tag]->initialize(PID, WGD, TGD, this);
  }

  /* FM - NOTE ON MODEL INITIALIZATION
   * new model can also be added without XML interface with the correct constructor
   * for example: models[tag] = new Particle_Model(...);
   * can use the AddSource visitor to add sources to the model.
   * std::vector<TracerParticle_Source *> new_sources;
   * models[tag]->accept(new AddSource(new_sources));
   */
}

void PLUMEGeneralData::run(QEStime loopTimeEnd,
                           WINDSGeneralData *WGD,
                           TURBGeneralData *TGD,
                           std::vector<QESNetCDFOutput *> outputVec)
{
  auto startTimeAdvec = std::chrono::high_resolution_clock::now();

  std::cout << "-------------------------------------------------------------------" << std::endl;

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

  updateCounts();

  std::cout << "[QES-Plume]\t Advecting particles from " << simTimeCurr << " to " << loopTimeEnd << ".\n"
            << "\t\t Total run time = " << loopTimeEnd - simTimeCurr << " s "
            << "(sim time = " << simTime << " s, iteration = " << simTimeIdx << "). \n";
  std::cout << "\t\t Particles: Released = " << isReleasedCount << " "
            << "Active = " << isActiveCount << "." << std::endl;

  // --------------------------------------------------------
  // MAIN TIME LOOP
  // --------------------------------------------------------
  while (simTimeCurr < loopTimeEnd) {

    auto startTime = std::chrono::high_resolution_clock::now();

    // the current time, updated in this loop with each new par_dt.
    double timeRemainder = loopTimeEnd - simTimeCurr;
    if (timeRemainder > sim_dt) {
      timeRemainder = sim_dt;
    }

    // This the main loop over all active particles
    for (const auto &pm : models) {
      pm.second->generateParticleList(simTimeCurr, timeRemainder, WGD, TGD, this);
      pm.second->advect(timeRemainder, WGD, TGD, this);
    }

    // incrementation of time and timestep
    simTimeIdx++;
    simTimeCurr += timeRemainder;
    simTime = simTimeCurr - simTimeStart;

    for (const auto &pm : models) {
      pm.second->process(simTimeCurr, timeRemainder, WGD, TGD, this);
    }

    // netcdf output for a given simulation timestep
    for (auto &id_out : outputVec) {
      id_out->save(simTimeCurr);
    }

    // output the time, isRogueCount, and isNotActiveCount information for all
    // simulations, but only when the updateFrequency allows
    if (simTimeCurr >= nextUpdate || (simTimeCurr == loopTimeEnd)) {
      if (verbose) {
        updateCounts();
        std::cout << "[QES-Plume]\t Time = " << simTimeCurr
                  << " (t = " << simTime << " s, iter = " << simTimeIdx << "). "
                  << "Particles: released = " << isReleasedCount << " "
                  << "active = " << isActiveCount << " "
                  << "rogue = " << isRogueCount << "." << std::endl;
      } else {
        updateCounts();
        std::cout << "[QES-Plume]\t Time = " << simTimeCurr
                  << " (t = " << simTime << " s, iter = " << simTimeIdx << "). "
                  << "Particles: released = " << isReleasedCount << " "
                  << "active = " << isActiveCount << "." << std::endl;
      }
      nextUpdate += (float)updateFrequency_timeLoop;
      // output advection loop runtime if in debug mode
      if (debug) {
        timers.printStoredTime("advection loop");
      }
    }
  }
  // --------------------------------------------------------
  // END TIME LOOP
  // --------------------------------------------------------

  updateCounts();

  std::cout << "[QES-Plume]\t End of particles advection at Time = " << simTimeCurr
            << " s (iteration = " << simTimeIdx << "). \n";
  std::cout << "\t\t Particles: Released = " << isReleasedCount << " "
            << "Active = " << isActiveCount << "." << std::endl;

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

void PLUMEGeneralData::applyBC(Particle *p)
{
  // now apply boundary conditions
  if (p->isActive) p->isActive = domainBC_x->enforce(p->xPos, p->uFluct);
  if (p->isActive) p->isActive = domainBC_y->enforce(p->yPos, p->vFluct);
  if (p->isActive) p->isActive = domainBC_z->enforce(p->zPos, p->wFluct);
}


double PLUMEGeneralData::calcCourantTimestep(const double &u,
                                             const double &v,
                                             const double &w,
                                             const double &timeRemainder)
{
  // set the output dt_par val to the timeRemainder
  // then if any of the Courant number values end up smaller, use that value instead
  double dt_par = timeRemainder;

  // if a velocity fluctuation is zero, it returns dt_par
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

double PLUMEGeneralData::calcCourantTimestep(const double &d,
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


void PLUMEGeneralData::GLE_solver(Particle *p, double &par_dt, TURBGeneralData *TGD)
{
  double txx = 0.0, txy = 0.0, txz = 0.0, tyy = 0.0, tyz = 0.0, tzz = 0.0;
  double flux_div_x = 0.0, flux_div_y = 0.0, flux_div_z = 0.0;

  interp->interpValues(TGD,
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

  // now need to call makeRealizable on tau
  makeRealizable(txx, txy, txz, tyy, tyz, tzz, invarianceTol);

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
  matmult(A_11,
          A_12,
          A_13,
          A_21,
          A_22,
          A_23,
          A_31,
          A_32,
          A_33,
          b_11,
          b_21,
          b_31,
          p->uFluct,
          p->vFluct,
          p->wFluct);

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

void PLUMEGeneralData::setParticle(WINDSGeneralData *WGD, TURBGeneralData *TGD, Particle *par_ptr)
{

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
  makeRealizable(txx, txy, txz, tyy, tyz, tzz, invarianceTol);

  // set tau_old to the interpolated values for each position
  par_ptr->txx = txx;
  par_ptr->txy = txy;
  par_ptr->txz = txz;
  par_ptr->tyy = tyy;
  par_ptr->tyz = tyz;
  par_ptr->tzz = tzz;

  // set delta_velFluct values to zero for now
  par_ptr->delta_uFluct = 0.0;
  par_ptr->delta_vFluct = 0.0;
  par_ptr->delta_wFluct = 0.0;

  // set isRogue to false and isActive to true for each particle
  // isActive = true as particle relased is active immediately
  par_ptr->isRogue = false;
  par_ptr->isActive = true;

  int cellIdNew = interp->getCellId(par_ptr->xPos, par_ptr->yPos, par_ptr->zPos);
  if ((WGD->icellflag[cellIdNew] == 0) || (WGD->icellflag[cellIdNew] == 2)) {
    // std::cerr << "WARNING invalid initial position" << std::endl;
    par_ptr->isActive = false;
  }

  double det = txx * (tyy * tzz - tyz * tyz) - txy * (txy * tzz - tyz * txz) + txz * (txy * tyz - tyy * txz);
  if (std::abs(det) < 1e-10) {
    // std::cerr << "WARNING invalid position stress" << std::endl;
    par_ptr->isActive = false;
  }
}

double PLUMEGeneralData::getMaxVariance(const TURBGeneralData *TGD)
{
  // set the initial maximum value to a very small number. The idea is to go through each value of the data,
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

void PLUMEGeneralData::calcInvariants(const double &txx,
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

void PLUMEGeneralData::makeRealizable(double &txx,
                                      double &txy,
                                      double &txz,
                                      double &tyy,
                                      double &tyz,
                                      double &tzz,
                                      const double &tol)
{
  // first calculate the invariants and see if they are already realizable
  // the calcInvariants function modifies the values directly, so they always
  // need initialized to something before being sent into said function to be
  // calculated
  double invar_xx = 0.0;
  double invar_yy = 0.0;
  double invar_zz = 0.0;
  calcInvariants(txx, txy, txz, tyy, tyz, tzz, invar_xx, invar_yy, invar_zz);

  if (invar_xx > tol && invar_yy > tol && invar_zz > tol) {
    return;// tau is already realizable
  }

  // since tau is not already realizable, need to make it realizeable
  // start by making a guess of ks, the subfilter scale tke
  double b = 4.0 / 3.0 * (txx + tyy + tzz);// also 4.0/3.0*invar_xx
  double c = txx * tyy + txx * tzz + tyy * tzz - txy * txy - txz * txz - tyz * tyz;// also invar_yy
  double ks = 1.01 * (-b + std::sqrt(b * b - 16.0 / 3.0 * c)) / (8.0 / 3.0);

  // if the initial guess is bad, use the straight up invar_xx value
  if (ks < tol || isnan(ks)) {
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

  calcInvariants(txx_new,
                 txy_new,
                 txz_new,
                 tyy_new,
                 tyz_new,
                 tzz_new,
                 invar_xx,
                 invar_yy,
                 invar_zz);

  // now adjust the diagonals by 0.05% of the subfilter tke, which is ks, till
  // tau is realizable or if too many iterations go on, give a warning. I've had
  // trouble with this taking too long
  //  if it isn't realizable, so maybe another approach for when the iterations
  //  are reached might be smart
  int iter = 0;
  while ((invar_xx < tol || invar_yy < tol || invar_zz < tol) && iter < 1000) {
    iter = iter + 1;

    // increase subfilter tke by 5%
    ks = ks * 1.050;

    // note that the right hand side is not tau_new, to force tau to only
    // increase by increasing ks
    txx_new = txx + 2.0 / 3.0 * ks;
    tyy_new = tyy + 2.0 / 3.0 * ks;
    tzz_new = tzz + 2.0 / 3.0 * ks;

    calcInvariants(txx_new,
                   txy_new,
                   txz_new,
                   tyy_new,
                   tyz_new,
                   tzz_new,
                   invar_xx,
                   invar_yy,
                   invar_zz);
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

bool PLUMEGeneralData::invert3(double &A_11,
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
  double det = A_11 * (A_22 * A_33 - A_23 * A_32)
               - A_12 * (A_21 * A_33 - A_23 * A_31)
               + A_13 * (A_21 * A_32 - A_22 * A_31);

  // check for near zero value determinants
  if (std::abs(det) < 1e-10) {
    std::cerr << "WARNING (Plume::invert3): matrix nearly singular" << std::endl;
    std::cerr << "abs(det) = \"" << std::abs(det)
              << "\",  A_11 =  \"" << A_11 << "\", A_12 = \"" << A_12 << "\", A_13 = \"" << A_13
              << "\", A_21 = \"" << A_21 << "\", A_22 = \"" << A_22 << "\", A_23 = \"" << A_23
              << "\", A_31 = \"" << A_31 << "\" A_32 = \"" << A_32 << "\", A_33 = \"" << A_33 << "\""
              << std::endl;

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

void PLUMEGeneralData::matmult(const double &A_11,
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

void PLUMEGeneralData::setBCfunctions(const std::string &xBCtype,
                                      const std::string &yBCtype,
                                      const std::string &zBCtype)
{
  // the idea is to use the string input BCtype to determine which boundary
  // condition function to use later in the program

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

void PLUMEGeneralData::updateCounts()
{
  isRogueCount = 0;
  isActiveCount = 0;
  isReleasedCount = 0;
  for (const auto &pm : models) {
    isRogueCount += pm.second->get_nbr_rogue();
    isActiveCount += pm.second->get_nbr_active();
    isReleasedCount += pm.second->get_nbr_inserted();
  }
}

void PLUMEGeneralData::showCurrentStatus()
{
  updateCounts();
  std::cout << "----------------------------------------------------------------- \n";
  std::cout << "[QES-Plume]\t End run particle summary \n";
  std::cout << "\t\t Current simulation time: " << simTimeCurr << "\n";
  std::cout << "\t\t Simulation run time: " << simTimeCurr - simTimeStart << "\n";
  std::cout << "\t\t Total number of particles released: " << isReleasedCount << "\n";
  std::cout << "\t\t Current number of particles in simulation: " << isActiveCount << "\n";
  std::cout << "\t\t Number of rogue particles: " << isRogueCount << "\n";
  for (const auto &pm : models) {
    std::cout << "\t\t Name: " << pm.first << " with " << pm.second->get_nbr_active() << " particles \n";
  }
  // std::cout << "Number of deleted particles: " << isNotActiveCount << "\n";
  std::cout << "----------------------------------------------------------------- \n"
            << std::flush;
}
