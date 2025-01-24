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

/** @file PLUMEGeneralData.cpp */

#include "PLUMEGeneralData.h"
#include <queue>

PLUMEGeneralData::PLUMEGeneralData(const PlumeParameters &PP,
                                   WINDSGeneralData *WGD,
                                   TURBGeneralData *TGD)
  : plumeParameters(PP)
{
  // copy debug information
  debug = false;// arguments->debug;
  verbose = false;// arguments->verbose;

  // make local copies of the QES-Winds nVals for each dimension
  std::tie(nx, ny, nz) = WGD->domain.getDomainCellNum();
  std::tie(dx, dy, dz) = WGD->domain.getDomainSize();
  dxy = WGD->domain.dxy();
}

PLUMEGeneralData::PLUMEGeneralData(const PlumeParameters &PP,
                                   PlumeInputData *PID,
                                   WINDSGeneralData *WGD,
                                   TURBGeneralData *TGD)
  : plumeParameters(PP)
{
  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "[QES-Plume]\t Initialization of plume model...\n";

  // copy debug information
  debug = false;// arguments->debug;
  verbose = false;// arguments->verbose;

  // make local copies of the QES-Winds nVals for each dimension
  std::tie(nx, ny, nz) = WGD->domain.getDomainCellNum();
  std::tie(dx, dy, dz) = WGD->domain.getDomainSize();
  dxy = WGD->domain.dxy();

  // Create instance of Interpolation class
  std::cout << "[QES-Plume]\t Interpolation Method set to: "
            << PID->plumeParams->interpMethod << std::endl;
  if (PID->plumeParams->interpMethod == "analyticalPowerLaw") {
    interp = new InterpPowerLaw(WGD->domain, debug);
  } else if (PID->plumeParams->interpMethod == "nearestCell") {
    interp = new InterpNearestCell(WGD->domain, debug);
  } else if (PID->plumeParams->interpMethod == "triLinear") {
    interp = new InterpTriLinear(WGD->domain, debug);
  } else {
    std::cerr << "[ERROR] unknown interpolation method" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get the domain start and end values, needed for wall boundary condition
  // application
  interp->getDomainBounds(domainXstart, domainXend, domainYstart, domainYend, domainZstart, domainZend);
  /*domainXstart = interp->xStart;
  domainXend = interp->xEnd;
  domainYstart = interp->yStart;
  domainYend = interp->yEnd;
  domainZstart = interp->zStart;
  domainZend = interp->zEnd;*/

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
      std::cerr << "[WARRNING] mesh wall reflection method not available" << std::endl;
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

  // set GLE Solver
  GLE_solver = new GLE_Solver_CPU();

  // Need dz for ground deposition
  float lBndz = PID->colParams->boxBoundsZ1;
  float uBndz = PID->colParams->boxBoundsZ2;
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
  QESDataTransport data;
  data.put("WGD", WGD);
  data.put("TGD", TGD);
  data.put("PGD", this);

  for (auto p : PID->particleParams->particles) {
    models[p->tag] = p->create(data);

    QESFileOutput_Interface *outfile;
    if (plumeParameters.plumeOutput) {
      outfile = new QESNetCDFOutput_v2(plumeParameters.outputFileBasename + "_" + p->tag + "_plumeOut.nc");
    } else {
      outfile = new QESNullOutput(plumeParameters.outputFileBasename + "_" + p->tag + "_plumeOut.nc");
    }
    outfile->setStartTime(simTimeStart);

    auto *stats = new StatisticsDirector(simTimeStart + PID->colParams->averagingStartTime,
                                         PID->colParams->averagingPeriod,
                                         outfile);
    if (PID->colParams) {
      stats->attach("concentration",
                    new Concentration(PID->colParams,
                                      models[p->tag]->particles_control,
                                      models[p->tag]->particles_core));
    }
    models[p->tag]->setStats(stats);
  }

  /* FM - NOTE ON MODEL INITIALIZATION
   * new model can also be added without XML interface with the correct constructor
   * for example: models[tag] = new Particle_Model(...);
   * can use the AddSource visitor to add sources to the model.
   * std::vector<TracerParticle_Source *> new_sources;
   * models[tag]->accept(new AddSource(new_sources));
   */

  // output for particle data (using visitor)
  if (plumeParameters.particleOutput) {
    particleOutput = new ParticleOutput(PID->partOutParams, this);
  }
}

PLUMEGeneralData::~PLUMEGeneralData()
{
  delete interp;
  delete wallReflect;

  delete domainBC_x;
  delete domainBC_y;
  delete domainBC_z;

  for (const auto &p : models)
    delete p.second;

#ifdef _OPENMP
  for (auto p : threadRNG)
    delete p;
#else
  delete RNG;
#endif
}

void PLUMEGeneralData::run(QEStime loopTimeEnd,
                           WINDSGeneralData *WGD,
                           TURBGeneralData *TGD)
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
    float timeRemainder = loopTimeEnd - simTimeCurr;
    if (timeRemainder > sim_dt) {
      timeRemainder = sim_dt;
    }

    // This the main loop over all active particles
    for (const auto &[key, pm] : models) {
      pm->generateParticleList(simTimeCurr, timeRemainder, WGD, TGD, this);
      pm->advect(timeRemainder, WGD, TGD, this);
    }

    // incrementation of time and timestep
    simTimeIdx++;
    simTimeCurr += timeRemainder;
    simTime = simTimeCurr - simTimeStart;

    // process particle information of output
    for (const auto &[key, pm] : models) {
      pm->process(simTimeCurr, timeRemainder, WGD, TGD, this);
    }

    // output for particle data (using visitor)
    if (particleOutput) {
      particleOutput->save(simTimeCurr, this);
    }

    // output the time, isRogueCount, and isNotActiveCount information for all
    // simulations, but only when the updateFrequency allows
    if (simTimeCurr >= nextUpdate || (simTimeCurr == loopTimeEnd)) {
      printProgress(simTime);
      nextUpdate += (float)updateFrequency_timeLoop;
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
  if (p->state == ACTIVE) domainBC_x->enforce(p->pos._1, p->velFluct._1, p->state);
  if (p->state == ACTIVE) domainBC_y->enforce(p->pos._2, p->velFluct._2, p->state);
  if (p->state == ACTIVE) domainBC_z->enforce(p->pos._3, p->velFluct._3, p->state);
}

void PLUMEGeneralData::applyBC(vec3 &pos, vec3 &velFluct, ParticleState &state)
{
  // now apply boundary conditions
  if (state == ACTIVE) domainBC_x->enforce(pos._1, velFluct._1, state);
  if (state == ACTIVE) domainBC_y->enforce(pos._2, velFluct._2, state);
  if (state == ACTIVE) domainBC_z->enforce(pos._3, velFluct._3, state);
}


float PLUMEGeneralData::calcCourantTimestep(const float &u,
                                            const float &v,
                                            const float &w,
                                            const float &timeRemainder)
{
  // set the output dt_par val to the timeRemainder
  // then if any of the Courant number values end up smaller, use that value instead
  float dt_par = timeRemainder;

  // if a velocity fluctuation is zero, it returns dt_par
  float dt_x = CourantNum * dx / std::abs(u);
  float dt_y = CourantNum * dy / std::abs(v);
  float dt_z = CourantNum * dz / std::abs(w);

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

float PLUMEGeneralData::calcCourantTimestep(const float &d,
                                            const vec3 &vel,
                                            const float &timeRemainder)
{
  // if the Courant Number is set to 0.0, we want to exit using the
  // timeRemainder (first time through that is the simTime)
  if (CourantNum == 0.0) {
    return timeRemainder;
  }

  float min_ds = std::min(dxy, dz);
  // float max_u = std::max({ u, v, w });
  float max_u = VectorMath::length(vel);
  float CN = 0.0;

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

/*void PLUMEGeneralData::GLE_solver_func(Particle *p, float &par_dt, TURBGeneralData *TGD)
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
*/

void PLUMEGeneralData::initializeParticleValues(Particle *par_ptr,
                                                WINDSGeneralData *WGD,
                                                TURBGeneralData *TGD)
{
  // set the positions to be used by the simulation to the initial positions
  par_ptr->pos = par_ptr->pos_init;


  // get the sigma values from the QES grid for the particle value
  vec3 sig;
  // get the tau values from the QES grid for the particle value
  mat3sym tau;

  interp->interpTurbInitialValues(TGD, par_ptr->pos, tau, sig);

  // now set the initial velocity fluctuations for the particle
  // The  sqrt of the variance is to match Bailey's code
  // normally distributed random number
#ifdef _OPENMP
  par_ptr->velFluct._1 = sig._1 * threadRNG[omp_get_thread_num()]->norRan();
  par_ptr->velFluct._2 = sig._2 * threadRNG[omp_get_thread_num()]->norRan();
  par_ptr->velFluct._3 = sig._3 * threadRNG[omp_get_thread_num()]->norRan();
#else
  par_ptr->velFluct._1 = sig._1 * RNG->norRan();
  par_ptr->velFluct._2 = sig._2 * RNG->norRan();
  par_ptr->velFluct._3 = sig._3 * RNG->norRan();
#endif


  // set the initial values for the old velFluct values
  par_ptr->velFluct_old = par_ptr->velFluct;

  // now need to call makeRealizable on tau
  VectorMath::makeRealizable(invarianceTol, tau);

  // set tau_old to the interpolated values for each position
  par_ptr->tau = tau;

  // set delta_velFluct values to zero for now
  par_ptr->delta_velFluct = { 0.0, 0.0, 0.0 };

  // set isRogue to false and isActive to true for each particle
  // isActive = true as particle relased is active immediately
  par_ptr->state = ACTIVE;
  // par_ptr->isRogue = false;
  // par_ptr->isActive = true;

  long cellIdNew = interp->getCellId(par_ptr->pos);
  if ((WGD->icellflag[cellIdNew] == 0) || (WGD->icellflag[cellIdNew] == 2)) {
    // std::cerr << "WARNING invalid initial position" << std::endl;
    par_ptr->state = INACTIVE;
  }

  /*
  double det = txx * (tyy * tzz - tyz * tyz) - txy * (txy * tzz - tyz * txz) + txz * (txy * tyz - tyy * txz);
  if (std::abs(det) < 1e-10) {
    // std::cerr << "WARNING invalid position stress" << std::endl;
    par_ptr->isActive = false;
  }
  */
}

void PLUMEGeneralData::initializeParticleValues(const vec3 &pos,
                                                ParticleLSDM &particle_ldsm,
                                                TURBGeneralData *TGD)
{
  // get the sigma values from the QES grid for the particle value
  vec3 sig;
  // get the tau values from the QES grid for the particle value
  mat3sym tau;

  interp->interpTurbInitialValues(TGD, pos, tau, sig);


  // now set the initial velocity fluctuations for the particle
  // The  sqrt of the variance is to match Bailey's code
  // normally distributed random number
  vec3 velFluct0;
#ifdef _OPENMP
  velFluct0._1 = sig._1 * threadRNG[omp_get_thread_num()]->norRan();
  velFluct0._2 = sig._2 * threadRNG[omp_get_thread_num()]->norRan();
  velFluct0._3 = sig._3 * threadRNG[omp_get_thread_num()]->norRan();
#else
  velFluct0._1 = sig._1 * RNG->norRan();
  velFluct0._2 = sig._2 * RNG->norRan();
  velFluct0._3 = sig._3 * RNG->norRan();
#endif

  // now need to call makeRealizable on tau
  VectorMath::makeRealizable(invarianceTol, tau);

  particle_ldsm.reset(velFluct0, tau);
  /*
  // set the initial values for the old velFluct values
  par_ptr->velFluct_old = par_ptr->velFluct;

  // now need to call makeRealizable on tau
  VectorMath::makeRealizable(invarianceTol, tau);

  // set tau_old to the interpolated values for each position
  par_ptr->tau = tau;

  // set delta_velFluct values to zero for now
  par_ptr->delta_velFluct = { 0.0, 0.0, 0.0 };

  // set isRogue to false and isActive to true for each particle
  // isActive = true as particle relased is active immediately
  par_ptr->state = ACTIVE;
  // par_ptr->isRogue = false;
  // par_ptr->isActive = true;

  long cellIdNew = interp->getCellId(par_ptr->pos);
  if ((WGD->icellflag[cellIdNew] == 0) || (WGD->icellflag[cellIdNew] == 2)) {
    // std::cerr << "WARNING invalid initial position" << std::endl;
    par_ptr->state = INACTIVE;
  }*/

  /*
  double det = txx * (tyy * tzz - tyz * tyz) - txy * (txy * tzz - tyz * txz) + txz * (txy * tyz - tyy * txz);
  if (std::abs(det) < 1e-10) {
    // std::cerr << "WARNING invalid position stress" << std::endl;
    par_ptr->isActive = false;
  }
  */
}


float PLUMEGeneralData::getMaxVariance(const TURBGeneralData *TGD)
{
  // set the initial maximum value to a very small number. The idea is to go through each value of the data,
  // setting the current value to the max value each time the current value is bigger than the old maximum value
  float maximumVal = -10e-10;

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

void PLUMEGeneralData::setBCfunctions(const std::string &xBCtype,
                                      const std::string &yBCtype,
                                      const std::string &zBCtype)
{
  // the idea is to use the string input BCtype to determine which boundary
  // condition function to use later in the program

  // output some debug information
  if (debug) {
    std::cout << "xBCtype = " << xBCtype << ":" << domainXstart << " " << domainXend << std::endl;
    std::cout << "yBCtype = " << yBCtype << ":" << domainYstart << " " << domainYend << std::endl;
    std::cout << "zBCtype = " << zBCtype << ":" << domainZstart << " " << domainZend << std::endl;
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
void PLUMEGeneralData::printProgress(const double &time)
{
  updateCounts();
  if (verbose) {
    std::cout << "[QES-Plume]\t Time = " << simTimeCurr
              << " (t = " << time << " s, iter = " << simTimeIdx << "). "
              << "Particles: released = " << isReleasedCount << " "
              << "active = " << isActiveCount << " "
              << "rogue = " << isRogueCount << "." << std::endl;
  } else {
    std::cout << "[QES-Plume]\t Time = " << simTimeCurr
              << " (t = " << time << " s, iter = " << simTimeIdx << "). "
              << "Particles: released = " << isReleasedCount << " "
              << "active = " << isActiveCount << "." << std::endl;
  }
  // output advection loop runtime if in debug mode
  if (debug) {
    timers.printStoredTime("advection loop");
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
