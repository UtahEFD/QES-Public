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

/** @file ParticleModel.cpp
 * @brief This class represents information stored for each particle
 */

#include "ParticleModel.h"
// #include "TracerParticle_Concentration.h"

#include "PLUMEInputData.h"
#include "PLUMEGeneralData.h"
// #include "PI_Source.hpp"


ParticleModel::ParticleModel(QESDataTransport &data, const std::string &tag_in)
  : particleType(base), tag(tag_in), particles_control()
{
  std::cout << "[QES-Plume]\t Model: Base Particle - Tag: " << tag << std::endl;

  // deposition = new Deposition(data.get<WINDSGeneralData *>("WGD"));

  // particles = new ManagedContainer<TracerParticle>();
}

ParticleModel::~ParticleModel()
{
  // delete particles;
  delete stats;
}

/*void ParticleModel::initialize(const PlumeInputData *PID,
                                     WINDSGeneralData *WGD,
                                     TURBGeneralData *TGD,
                                     PLUMEGeneralData *PGD)
{
 QESDataTransport data;
 data.put("WGD", WGD);
 data.put("TGD", TGD);
 data.put("PGD", PGD);



 // stats = new TracerParticle_Statistics(PID, PGD, this);
 // concentration = new TracerParticle_Concentration(PID, pm);


 QESFileOutput_Interface *outfile;
 if (PGD->plumeParameters.plumeOutput) {
   outfile = new QESNetCDFOutput_v2(PGD->plumeParameters.outputFileBasename + "_" + tag + "_plumeOut.nc");
 } else {
   outfile = new QESNullOutput(PGD->plumeParameters.outputFileBasename + "_" + tag + "_plumeOut.nc");
 }
 stats = new StatisticsDirector(PID, PGD, outfile);
 if (PID->colParams) {
   stats->attach("concentration", new TracerParticle_Concentration(PID->colParams, this));
 }
 // other statistics can be added here
}*/

void ParticleModel::generateParticleList(QEStime &timeCurrent,
                                         const float &dt,
                                         WINDSGeneralData *WGD,
                                         TURBGeneralData *TGD,
                                         PLUMEGeneralData *PGD)
{
  int nbr_new_particle = 0;
  for (auto s : sources) {
    nbr_new_particle += s->generate(timeCurrent);
  }

  particles_control.check_resize(nbr_new_particle);
  particles_control.resize_companion(particles_core);
  particles_control.resize_companion(particles_lsdm);
  particles_control.resize_companion(particles_metadata);

  for (auto s : sources) {
    if (s->isActive(timeCurrent)) {
      std::vector<size_t> newIdx;
      particles_control.obtain_available(s->data().get_ref<std::vector<u_int32_t>>("ID").size(),
                                         newIdx);

      for (size_t k = 0; k < newIdx.size(); ++k) {
        particles_control[newIdx[k]].reset();
      }

      if (s->data().contains("mass")) {
        for (size_t k = 0; k < newIdx.size(); ++k) {
          particles_core[newIdx[k]].reset(s->data().get_ref<std::vector<u_int32_t>>("ID")[k],
                                          s->data().get_ref<std::vector<vec3>>("position")[k],
                                          s->data().get_ref<std::vector<float>>("diameter")[k],
                                          s->data().get_ref<std::vector<float>>("mass")[k],
                                          s->data().get_ref<std::vector<float>>("density")[k]);
        }
      } else {
        for (size_t k = 0; k < newIdx.size(); ++k) {
          particles_core[newIdx[k]].reset(s->data().get_ref<std::vector<u_int32_t>>("ID")[k],
                                          s->data().get_ref<std::vector<vec3>>("position")[k]);
        }
      }

      for (size_t k = 0; k < newIdx.size(); ++k) {
        particles_metadata[newIdx[k]].reset(particleType,
                                            particles_core[k].pos,
                                            timeCurrent,
                                            s->getID(),
                                            particles_core[k].m);
      }
    }
    // setParticle(time, s, particles);
  }

/*
double time = timeCurrent - PGD->getSimTimeStart();
for (auto source : sources) {
  nbr_new_particle += source->getNewParticleNumber(dt, time);
}
particles->sweep(nbr_new_particle);
for (auto source : sources) {
  source->emitParticles(dt, time, particles);
  // source->getInitializationData()
  //
}
*/
#pragma omp parallel for default(none) shared(WGD, TGD, PGD)
  for (auto k = 0u; k < particles_control.get_nbr_added(); ++k) {
    long cellId = PGD->interp->getCellId(particles_core[k].pos);
    if ((WGD->icellflag[cellId] == 0) || (WGD->icellflag[cellId] == 2)) {
      // std::cerr << "WARNING invalid initial position" << std::endl;
      particles_control[k].state = INACTIVE;
    } else {
      // set particle ID (use global particle counter)
      PGD->initializeParticleValues(particles_core[particles_control.get_added_index(k)].pos,
                                    particles_lsdm[particles_control.get_added_index(k)],
                                    TGD);
    }
  }
}


/*void ParticleModel::advect(const double &total_time_interval,
                           WINDSGeneralData *WGD,
                           TURBGeneralData *TGD,
                           PLUMEGeneralData *PGD)
{
#pragma omp parallel for default(none) shared(WGD, TGD, PGD, total_time_interval)
  for (auto k = 0u; k < particles_control.size(); ++k) {
    try {
      // ParticleCore &p_core = particles_core[k];

      float rhoAir = 1.225;// in kg m^-3
      float nuAir = 1.506E-5;// in m^2 s^-1

      // settling velocity
      float vs = 0;

      float timeRemainder = total_time_interval;

      long cellId = PGD->interp->getCellId(particles_core[k].pos);

      while (particles_control[k].state == ACTIVE && timeRemainder > 0.0) {

      //  now get the Lagrangian values for the current iteration from the wind/turbulence grid
      //  will need to use the interp3D function

        PGD->interp->interpWindsValues(WGD, particles_core[k].pos, particles_lsdm[k].velMean);

        // adjusting mean vertical velocity for settling velocity
        particles_lsdm[k].velMean._3 -= vs;

        // now calculate the particle timestep using the courant number, the velocity fluctuation from the last time,
        // and the grid sizes.
        // Uses timeRemainder as the timestep if it is smaller than the one calculated from the Courant number
        // int cellId = PGD->interp->getCellId(p->xPos, p->yPos, p->zPos);


        float dWall = WGD->mixingLengths[cellId];
        vec3 mTot = VectorMath::add(VectorMath::abs(particles_lsdm[k].velMean),
                                    VectorMath::abs(particles_lsdm[k].velFluct));
        float par_dt = PGD->calcCourantTimestep(dWall, mTot, timeRemainder);

        // std::cout << "par_dt = " << par_dt << std::endl;
        //  update the par_time, useful for debugging
        // par_time = par_time + par_dt;

        // PGD->GLE_solver->solve(p, par_dt, TGD, PGD);
        //PGD->GLE_solver->solve(par_dt, particles_control[k], particles_core[k], particles_lsdm[k], TGD, PGD);

        if (particles_control[k].state == ROGUE) {
          nbr_rogue++;
          break;
        }
        // Pete: Do you need this???
        // ONLY if this should never happen....
        //    assert( isRogue == false );

        // now update the particle position for this iteration
        vec3 dist = VectorMath::multiply(par_dt, VectorMath::add(particles_lsdm[k].velMean, particles_lsdm[k].velFluct));
        // x_n+1 = x_n + v*dt
        particles_core[k].pos = VectorMath::add(particles_core[k].pos, dist);

        vec3 velTot = VectorMath::add(particles_lsdm[k].velMean, particles_lsdm[k].velFluct);

        // Deposit mass (vegetation only right now)
        //if (p->depFlag && p->state == ACTIVE) {
        //  deposition->deposit(p, dist, velTot, vs, WGD, TGD, PGD->interp);
        //}

        // check and do wall (building and terrain) reflection (based in the method)
        if (particles_control[k].state == ACTIVE) {
          PGD->wallReflect->reflect(WGD,
                                    particles_core[k].pos,
                                    dist,
                                    particles_lsdm[k].velFluct,
                                    particles_control[k].state);
        }

        // now apply boundary conditions
        PGD->applyBC(particles_core[k].pos,
                     particles_lsdm[k].velFluct,
                     particles_control[k].state);

        // now update the old values to be ready for the next particle time iteration
        // the current values are already set for the next iteration by the above calculations
        // !!! this is extremely important for the next iteration to work accurately
        particles_lsdm[k].delta_velFluct = VectorMath::subtract(particles_lsdm[k].velFluct, particles_lsdm[k].velFluct_old);

        particles_lsdm[k].velFluct_old = particles_lsdm[k].velFluct;

        // now set the time remainder for the next loop
        // if the par_dt calculated from the Courant Number is greater than the timeRemainder,
        // the function for calculating par_dt will use the timeRemainder for the output par_dt
        // so this should result in a timeRemainder of exactly zero, no need for a tol.
        timeRemainder = timeRemainder - par_dt;

      }// while( isActive == true && timeRemainder > 0.0 )
    } catch (const std::out_of_range &oor) {
      // cell ID out of bound (assuming particle outside of domain)
      // std::cerr << "Particle: " << particles_core[k].ID << " state: " << particles_control[k].state << "\n"
      //          << particles_core[k].pos._1 << " " << particles_core[k].pos._2 << " " << particles_core[k].pos._3 << std::endl;
      particles_control[k].state = INACTIVE;
    }
  }//  END OF OPENMP WORK SHARE
}*/

void ParticleModel::process(QEStime &timeIn,
                            const float &dt,
                            WINDSGeneralData *WGD,
                            TURBGeneralData *TGD,
                            PLUMEGeneralData *PGD)
{
  stats->compute(timeIn, dt);
}

void ParticleModel::advect(const float &total_time_interval,
                           WINDSGeneralData *WGD,
                           TURBGeneralData *TGD,
                           PLUMEGeneralData *PGD)
{
#pragma omp parallel for default(none) shared(WGD, TGD, PGD, total_time_interval)
  for (auto k = 0u; k < particles_control.size(); ++k) {
    try {
      // ParticleCore &p_core = particles_core[k];

      float rhoAir = 1.225;// in kg m^-3
      float nuAir = 1.506E-5;// in m^2 s^-1

      // settling velocity
      float vs = 0;

      float timeRemainder = total_time_interval;

      bool isRogue = false;

      vec3 pos = particles_core[k].pos;
      // vec3 dist = { 0.0, 0.0, 0.0 };
      vec3 velMean = { 0.0, 0.0, 0.0 };

      float nuT;
      float CoEps = 1e-6;

      vec3 velFluct = particles_lsdm[k].velFluct;
      vec3 velFluct_old = particles_lsdm[k].velFluct_old;

      mat3sym tau_old = particles_lsdm[k].tau;
      mat3sym tau = particles_lsdm[k].tau;

      vec3 delta_velFluct = { 0.0, 0.0, 0.0 };

      while (particles_control[k].state == ACTIVE && timeRemainder > 0.0) {

        // now get the Lagrangian values for the current iteration from the wind/turbulence grid
        // will need to use the interp3D function

        PGD->interp->interpWindsValues(WGD, pos, velMean);

        // adjusting mean vertical velocity for settling velocity
        velMean._3 -= vs;

        long cellId = PGD->interp->getCellId(pos);
        float dWall = WGD->mixingLengths[cellId];
        vec3 mTot = VectorMath::add(VectorMath::abs(velMean),
                                    VectorMath::abs(velFluct));
        float dt = PGD->calcCourantTimestep(dWall, mTot, timeRemainder);

        // mat3sym tau = particles_lsdm[k].tau;
        vec3 flux_div = { 0.0, 0.0, 0.0 };

        PGD->interp->interpTurbValues(TGD, pos, tau, flux_div, nuT, CoEps);

        // now calculate the particle timestep using the courant number, the velocity fluctuation from the last time,
        // and the grid sizes.
        // Uses timeRemainder as the timestep if it is smaller than the one calculated from the Courant number
        // int cellId = PGD->interp->getCellId(p->xPos, p->yPos, p->zPos);


        // std::cout << "par_dt = " << par_dt << std::endl;
        //  update the par_time, useful for debugging
        // par_time = par_time + par_dt;

        // PGD->GLE_solver->solve(p, par_dt, TGD, PGD);
        // PGD->GLE_solver->solve(par_dt, particles_control[k], particles_core[k], particles_lsdm[k], TGD, PGD);

        // now need to call makeRealizable on tau
        VectorMath::makeRealizable(PGD->invarianceTol, tau);
        // now need to calculate the inverse values for tau
        // directly modifies the values of tau
        mat3 L = { tau._11, tau._12, tau._13, tau._12, tau._22, tau._23, tau._13, tau._23, tau._33 };

        isRogue = !VectorMath::invert(L);
        if (isRogue) {
          // std::cerr << "Particle # " << particles_core[k].ID << " is rogue, ";
          // int cellIdNew = interp->getCellId(xPos,yPos,zPos);
          // std::cerr << "ERROR in Matrix inversion of stress tensor" << std::endl;
          // std::cerr << "Particle # " << particles_core[k].ID << " is rogue " << std::endl;
          break;
        }

        // these are the random numbers for each direction

#ifdef _OPENMP
        vec3 vRandn = { PGD->threadRNG[omp_get_thread_num()]->norRan(),
                        PGD->threadRNG[omp_get_thread_num()]->norRan(),
                        PGD->threadRNG[omp_get_thread_num()]->norRan() };
#else
        vec3 vRandn = { PGD->RNG->norRan(), PGD->RNG->norRan(), PGD->RNG->norRan() };
#endif

        // now calculate a bunch of values for the current particle
        // calculate the time derivative of the stress tensor: (tau_current - tau_old)/dt
        mat3sym tau_ddt = { (tau._11 - tau_old._11) / dt,
                            (tau._12 - tau_old._12) / dt,
                            (tau._13 - tau_old._13) / dt,
                            (tau._22 - tau_old._22) / dt,
                            (tau._23 - tau_old._23) / dt,
                            (tau._33 - tau_old._33) / dt };

        // now calculate and set the A and b matrices for an Ax = b
        // A = -I + 0.5*(-CoEps*L + dTdt*L )*par_dt;
        mat3 A = { -1.0f + 0.5f * (-CoEps * L._11 + L._11 * tau_ddt._11 + L._12 * tau_ddt._12 + L._13 * tau_ddt._13) * dt,
                   -0.0f + 0.5f * (-CoEps * L._12 + L._12 * tau_ddt._11 + L._22 * tau_ddt._12 + L._23 * tau_ddt._13) * dt,
                   -0.0f + 0.5f * (-CoEps * L._13 + L._13 * tau_ddt._11 + L._23 * tau_ddt._12 + L._33 * tau_ddt._13) * dt,
                   -0.0f + 0.5f * (-CoEps * L._12 + L._11 * tau_ddt._12 + L._12 * tau_ddt._22 + L._13 * tau_ddt._23) * dt,
                   -1.0f + 0.5f * (-CoEps * L._22 + L._12 * tau_ddt._12 + L._22 * tau_ddt._22 + L._23 * tau_ddt._23) * dt,
                   -0.0f + 0.5f * (-CoEps * L._23 + L._13 * tau_ddt._12 + L._23 * tau_ddt._22 + L._33 * tau_ddt._23) * dt,
                   -0.0f + 0.5f * (-CoEps * L._13 + L._11 * tau_ddt._13 + L._12 * tau_ddt._23 + L._13 * tau_ddt._33) * dt,
                   -0.0f + 0.5f * (-CoEps * L._23 + L._12 * tau_ddt._13 + L._22 * tau_ddt._23 + L._23 * tau_ddt._33) * dt,
                   -1.0f + 0.5f * (-CoEps * L._33 + L._13 * tau_ddt._13 + L._23 * tau_ddt._23 + L._33 * tau_ddt._33) * dt };

        // b = -vectFluct - 0.5*vecFluxDiv*par_dt - sqrt(CoEps*par_dt)*vecRandn;
        vec3 b = { -velFluct_old._1 - 0.5f * flux_div._1 * dt - sqrtf(CoEps * dt) * vRandn._1,
                   -velFluct_old._2 - 0.5f * flux_div._2 * dt - sqrtf(CoEps * dt) * vRandn._2,
                   -velFluct_old._3 - 0.5f * flux_div._3 * dt - sqrtf(CoEps * dt) * vRandn._3 };

        // now prepare for the Ax=b calculation by calculating the inverted A matrix
        isRogue = !VectorMath::invert(A);
        if (isRogue) {
          // std::cerr << "ERROR in matrix inversion in Langevin equation" << std::endl;
          // std::cerr << "Particle # " << particles_core[k].ID << " is rogue " << std::endl;
          // isActive = false;
          break;
        }

        // now do the Ax=b calculation using the inverted matrix (vecFluct = A*b)
        VectorMath::multiply(A, b, velFluct);
        
        // now check to see if the value is rogue or not
        if (std::abs(velFluct._1) >= PGD->vel_threshold || isnan(velFluct._1)) {
          // std::cerr << "Particle # " << particles_core[k].ID << " is rogue" << std::endl;
          //  std::cerr << "uFluct = " << p_lsdm.velFluct._1 << ", CoEps = " << p_lsdm.CoEps << std::endl;
          //  velFluct._1 = 0.0;
          //  isActive = false;
          isRogue = true;
          // break;
        }
        if (std::abs(velFluct._2) >= PGD->vel_threshold || isnan(velFluct._2)) {
          // std::cerr << "Particle # " << particles_core[k].ID << " is rogue" << std::endl;
          // std::cerr << "vFluct = " << p_lsdm.velFluct._2 << ", CoEps = " << p_lsdm.CoEps << std::endl;
          // velFluct._2 = 0.0;
          // isActive = false;
          isRogue = true;
          // break;
        }
        if (std::abs(velFluct._3) >= PGD->vel_threshold || isnan(velFluct._3)) {
          // std::cerr << "Particle # " << particles_core[k].ID << " is rogue" << std::endl;
          // std::cerr << "wFluct = " << p_lsdm.velFluct._3 << ", CoEps = " << p_lsdm.CoEps << std::endl;
          // velFluct._3 = 0.0;
          // isActive = false;
          isRogue = true;
          // break;
        }

        if (isRogue) {
          /*std::cerr << "---------------------------------------------------------------------------- \n"
                    << "Particle: " << (isRogue ? "ROGUE" : "ACTIVE") << "\n"
                    << "ID:   " << particles_core[k].ID << " \n"
                    << "dt:   " << dt << "\n"
                    << "x:    " << pos._1 << " " << pos._2 << " " << pos._3 << "\n"
                    << "U:    " << velMean._1 << " " << velMean._2 << " " << velMean._3 << "\n"
                    << "u_o:  " << velFluct_old._1 << " " << velFluct_old._2 << " " << velFluct_old._3 << "\n"
                    << "u:    " << velFluct._1 << " " << velFluct._2 << " " << velFluct._3 << "\n"
                    << "b:    " << b._1 << " " << b._2 << " " << b._3 << "\n"
                    << "T:    " << tau._11 << " " << tau._12 << " " << tau._13 << " " << tau._22 << " " << tau._23 << " " << tau._33 << "\n"
                    << "dTdt: " << tau_ddt._11 << " " << tau_ddt._12 << " " << tau_ddt._13 << " " << tau_ddt._22 << " " << tau_ddt._23 << " " << tau_ddt._33 << "\n"
                    << "fdiv: " << flux_div._1 << " " << flux_div._2 << " " << flux_div._3 << "\n"
                    << "----------------------------------------------------------------------------" << std::endl;*/
          break;
        }


        // Pete: Do you need this???
        // ONLY if this should never happen....
        //    assert( isRogue == false );

        // now update the particle position for this iteration
        vec3 dist = VectorMath::multiply(dt, VectorMath::add(velMean, velFluct));
        // x_n+1 = x_n + v*dt
        pos = VectorMath::add(pos, dist);

        vec3 velTot = VectorMath::add(particles_lsdm[k].velMean, particles_lsdm[k].velFluct);

        // Deposit mass (vegetation only right now)
        // if (p->depFlag && p->state == ACTIVE) {
        //  deposition->deposit(p, dist, velTot, vs, WGD, TGD, PGD->interp);
        //}

        // check and do wall (building and terrain) reflection (based in the method)
        if (particles_control[k].state == ACTIVE) {
          PGD->wallReflect->reflect(WGD,
                                    pos,
                                    dist,
                                    velFluct,
                                    particles_control[k].state);
        }

        // now apply boundary conditions
        PGD->applyBC(pos,
                     velFluct,
                     particles_control[k].state);

        // now update the old values to be ready for the next particle time iteration
        // the current values are already set for the next iteration by the above calculations
        // !!! this is extremely important for the next iteration to work accurately
        delta_velFluct = VectorMath::subtract(velFluct, velFluct_old);

        velFluct_old = velFluct;

        tau_old = tau;

        // now set the time remainder for the next loop
        // if the par_dt calculated from the Courant Number is greater than the timeRemainder,
        // the function for calculating par_dt will use the timeRemainder for the output par_dt
        // so this should result in a timeRemainder of exactly zero, no need for a tol.
        timeRemainder = timeRemainder - dt;

      }// while( isActive == true && timeRemainder > 0.0 )

      particles_core[k].pos = pos;

      particles_lsdm[k].CoEps = CoEps;
      particles_lsdm[k].velMean = velMean;
      particles_lsdm[k].delta_velFluct = delta_velFluct;
      particles_lsdm[k].velFluct = velFluct;
      particles_lsdm[k].velFluct_old = velFluct_old;
      particles_lsdm[k].delta_velFluct = delta_velFluct;
      particles_lsdm[k].tau = tau_old;

      if (isRogue) {
        particles_control[k].state = ROGUE;
        nbr_rogue++;
      }

    } catch (const std::out_of_range &oor) {
      // cell ID out of bound (assuming particle outside of domain)
      /*std::cerr << "Particle: " << particles_core[k].ID << " state: " << particles_control[k].state << "\n"
                << particles_core[k].pos._1 << " " << particles_core[k].pos._2 << " " << particles_core[k].pos._3 << std::endl;*/
      particles_control[k].state = INACTIVE;
    }


  }//  END OF OPENMP WORK SHARE
}
