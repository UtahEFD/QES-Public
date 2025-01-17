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
    // -> call particle->reuse(data[k])
    // set particle ID (use global particle counter)
    PGD->initializeParticleValues(particles_core[particles_control.get_added_index(k)].pos,
                                  particles_lsdm[particles_control.get_added_index(k)],
                                  TGD);
  }
}


void ParticleModel::advect(const double &total_time_interval,
                           WINDSGeneralData *WGD,
                           TURBGeneralData *TGD,
                           PLUMEGeneralData *PGD)
{
#pragma omp parallel for default(none) shared(WGD, TGD, PGD, total_time_interval)
  for (auto k = 0u; k < particles_control.size(); ++k) {
    float rhoAir = 1.225;// in kg m^-3
    float nuAir = 1.506E-5;// in m^2 s^-1

    // settling velocity
    float vs = 0;

    float timeRemainder = total_time_interval;
    while (particles_control[k].state == ACTIVE && timeRemainder > 0.0) {

      /*
        now get the Lagrangian values for the current iteration from the wind/turbulence grid
        will need to use the interp3D function
      */

      PGD->interp->interpWindsValues(WGD, particles_core[k].pos, particles_lsdm[k].velMean);

      // adjusting mean vertical velocity for settling velocity

      particles_lsdm[k].velMean._3 -= vs;

      // now calculate the particle timestep using the courant number, the velocity fluctuation from the last time,
      // and the grid sizes.
      // Uses timeRemainder as the timestep if it is smaller than the one calculated from the Courant number
      // int cellId = PGD->interp->getCellId(p->xPos, p->yPos, p->zPos);
      long cellId = PGD->interp->getCellId(particles_core[k].pos);

      float dWall = WGD->mixingLengths[cellId];
      float par_dt = PGD->calcCourantTimestep(dWall,
                                              VectorMath::add(VectorMath::abs(particles_lsdm[k].velMean),
                                                              VectorMath::abs(particles_lsdm[k].velFluct)),
                                              timeRemainder);

      // std::cout << "par_dt = " << par_dt << std::endl;
      //  update the par_time, useful for debugging
      // par_time = par_time + par_dt;

      // PGD->GLE_solver->solve(p, par_dt, TGD, PGD);

      if (particles_control[k].state == ROGUE) {
        nbr_rogue++;
        break;
      }
      // Pete: Do you need this???
      // ONLY if this should never happen....
      //    assert( isRogue == false );

      // now update the particle position for this iteration
      vec3 dist = VectorMath::multiply(par_dt, VectorMath::add(particles_lsdm[k].velMean, particles_lsdm[k].velFluct));
      particles_core[k].pos = VectorMath::add(particles_core[k].pos, dist);

      vec3 velTot = VectorMath::add(particles_lsdm[k].velMean, particles_lsdm[k].velFluct);

      // Deposit mass (vegetation only right now)
      /*if (p->depFlag && p->state == ACTIVE) {
        deposition->deposit(p, dist, velTot, vs, WGD, TGD, PGD->interp);
      }*/

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

  }//  END OF OPENMP WORK SHARE
}

void ParticleModel::process(QEStime &timeIn,
                            const float &dt,
                            WINDSGeneralData *WGD,
                            TURBGeneralData *TGD,
                            PLUMEGeneralData *PGD)
{
  stats->compute(timeIn, dt);
}
