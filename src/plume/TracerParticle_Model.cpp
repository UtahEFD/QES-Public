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

/** @file Particle.h
 * @brief This class represents information stored for each particle
 */

#include "TracerParticle_Model.h"
#include "TracerParticle_Concentration.h"

#include "PLUMEInputData.h"
#include "PLUMEGeneralData.h"
#include "PI_Source.hpp"


TracerParticle_Model::TracerParticle_Model(QESDataTransport &data, const string &tag_in)
  : ParticleModel(ParticleType::tracer, tag_in),
    particles()
{
  std::cout << "[QES-Plume]\t Model: Tracer Particle - Tag: " << tag << std::endl;

  deposition = new Deposition(data.get<WINDSGeneralData *>("WGD"));

  // particles = new ManagedContainer<TracerParticle>();
}

TracerParticle_Model::~TracerParticle_Model()
{
  // delete particles;
  delete stats;
}

/*void TracerParticle_Model::initialize(const PlumeInputData *PID,
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

void TracerParticle_Model::generateParticleList(QEStime &timeCurrent,
                                                const float &dt,
                                                WINDSGeneralData *WGD,
                                                TURBGeneralData *TGD,
                                                PLUMEGeneralData *PGD)
{
  int nbr_new_particle = 0;
  for (auto s : sources)
    nbr_new_particle += s->generate(timeCurrent);

  particles.check_resize(nbr_new_particle);

  for (auto s : sources) {
    if (s->isActive(timeCurrent)) {
      for (size_t k = 0; k < s->data().get_ref<std::vector<u_int32_t>>("ID").size(); ++k) {
        // p.get(k)->pos_init = x[k];
        particles.insert();

        particles.last_added()->ID = s->data().get_ref<std::vector<u_int32_t>>("ID")[k];
        particles.last_added()->sourceIdx = s->getID();
        particles.last_added()->timeStrt = timeCurrent;
        particles.last_added()->pos_init = s->data().get_ref<std::vector<vec3>>("position")[k];

        if (s->data().contains("mass")) {
          particles.last_added()->m = s->data().get_ref<std::vector<float>>("mass")[k];
        }
      }
      std::vector<size_t> newIdx;
      particles.obtain_available(s->data().get_ref<std::vector<u_int32_t>>("ID").size(), newIdx);

      for (size_t k = 0; k < newIdx.size(); ++k) {
        // p.get(k)->pos_init = x[k];
        particles[newIdx[k]].ID = s->data().get_ref<std::vector<u_int32_t>>("ID")[k];
        particles[newIdx[k]].sourceIdx = s->getID();
        particles[newIdx[k]].timeStrt = timeCurrent;
        particles[newIdx[k]].pos_init = s->data().get_ref<std::vector<vec3>>("position")[k];

        if (s->data().contains("mass")) {
          particles[newIdx[k]].m = s->data().get_ref<std::vector<float>>("mass")[k];
        }
        // p.last_added()->d = s->data.get_ref<std::vector<float>>("diameter")[k];
        // p.last_added()->rho = s->data.get_ref<std::vector<float>>("density")[k];

        // [FM] How to reset the particle...
        // CORE[k].reset(s->data().get_ref<std::vector<u_int32_t>>("ID")[k],
        //               s->data().get_ref<std::vector<vec3>>("position")[k]);
        // METADATA[k].reset(type,CORE[k].pos,timeCurrent,s->getID(),CORE[k].m);
      }
    }
    // setParticle(time, s, particles);
  }
  /*
  float time = timeCurrent - PGD->getSimTimeStart();
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
  for (auto k = 0u; k < particles.get_nbr_added(); ++k) {
    // -> call particle->reuse(data[k])
    // set particle ID (use global particle counter)
    PGD->initializeParticleValues(particles.get_added(k), WGD, TGD);
  }
}


void TracerParticle_Model::advect(const float &total_time_interval,
                                  WINDSGeneralData *WGD,
                                  TURBGeneralData *TGD,
                                  PLUMEGeneralData *PGD)
{
#pragma omp parallel for default(none) shared(WGD, TGD, PGD, total_time_interval)
  for (auto k = 0u; k < particles.size(); ++k) {
    TracerParticle *p = particles.get_ptr(k);
    float rhoAir = 1.225;// in kg m^-3
    float nuAir = 1.506E-5;// in m^2 s^-1

    // settling velocity
    float vs = 0;

    float timeRemainder = total_time_interval;
    while (particles[k].state == ACTIVE && timeRemainder > 0.0) {

      /*
        now get the Lagrangian values for the current iteration from the wind/turbulence grid
        will need to use the interp3D function
      */

      PGD->interp->interpWindsValues(WGD, p->pos, p->velMean);

      // adjusting mean vertical velocity for settling velocity

      p->velMean._3 -= vs;

      // now calculate the particle timestep using the courant number, the velocity fluctuation from the last time,
      // and the grid sizes.
      // Uses timeRemainder as the timestep if it is smaller than the one calculated from the Courant number
      // int cellId = PGD->interp->getCellId(p->xPos, p->yPos, p->zPos);
      long cellId = PGD->interp->getCellId(p->pos);

      float dWall = WGD->mixingLengths[cellId];
      float par_dt = PGD->calcCourantTimestep(dWall,
                                              VectorMath::add(VectorMath::abs(p->velMean), VectorMath::abs(p->velFluct)),
                                              timeRemainder);

      // std::cout << "par_dt = " << par_dt << std::endl;
      //  update the par_time, useful for debugging
      // par_time = par_time + par_dt;

      // PGD->GLE_solver->solve(p, par_dt, TGD, PGD);

      if (p->state == ROGUE) {
        nbr_rogue++;
        break;
      }
      // Pete: Do you need this???
      // ONLY if this should never happen....
      //    assert( isRogue == false );

      // now update the particle position for this iteration
      vec3 dist = VectorMath::multiply(par_dt, VectorMath::add(p->velMean, p->velFluct));
      p->pos = VectorMath::add(p->pos, dist);

      vec3 velTot = VectorMath::add(p->velMean, p->velFluct);

      // Deposit mass (vegetation only right now)
      if (p->depFlag && p->state == ACTIVE) {
        deposition->deposit(p, dist, velTot, vs, WGD, TGD, PGD->interp);
      }

      // check and do wall (building and terrain) reflection (based in the method)
      if (p->state == ACTIVE) {
        PGD->wallReflect->reflect(WGD, p->pos, dist, p->velFluct, p->state);
      }

      // now apply boundary conditions
      PGD->applyBC(p);

      // now update the old values to be ready for the next particle time iteration
      // the current values are already set for the next iteration by the above calculations
      // !!! this is extremely important for the next iteration to work accurately
      p->delta_velFluct = VectorMath::subtract(p->velFluct, p->velFluct_old);

      p->velFluct_old = p->velFluct;

      // now set the time remainder for the next loop
      // if the par_dt calculated from the Courant Number is greater than the timeRemainder,
      // the function for calculating par_dt will use the timeRemainder for the output par_dt
      // so this should result in a timeRemainder of exactly zero, no need for a tol.
      timeRemainder = timeRemainder - par_dt;

    }// while( isActive == true && timeRemainder > 0.0 )

  }//  END OF OPENMP WORK SHARE
}

void TracerParticle_Model::process(QEStime &timeIn,
                                   const float &dt,
                                   WINDSGeneralData *WGD,
                                   TURBGeneralData *TGD,
                                   PLUMEGeneralData *PGD)
{
  stats->compute(timeIn, dt);
}
