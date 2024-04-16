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

/** @file Particle.h
 * @brief This class represents information stored for each particle
 */

#include "TracerParticle_Model.h"
#include "TracerParticle_Concentration.h"

#include "PlumeInputData.hpp"
#include "PLUMEGeneralData.h"
#include "PI_TracerParticle.hpp"

#include "util/DataSource.h"


TracerParticle_Model::TracerParticle_Model(const PI_TracerParticle *in)
  : ParticleModel(ParticleType::tracer, in->tag)
{
  std::cout << "[QES-Plume]\t Model: Tracer Particle - Tag: " << tag << std::endl;

  particles = new ManagedContainer<TracerParticle>();


  for (auto s : in->sources) {
    // now determine the number of particles to release for the source and update the overall count
    // totalParsToRelease += s->getNumParticles();

    // add source into the vector of sources
    sources.emplace_back(new TracerParticle_Source((int)sources.size(), s));
  }
}

TracerParticle_Model::~TracerParticle_Model()
{
  delete particles;
  delete stats;
}

void TracerParticle_Model::initialize(const PlumeInputData *PID,
                                      WINDSGeneralData *WGD,
                                      TURBGeneralData *TGD,
                                      PLUMEGeneralData *PGD)
{
  deposition = new Deposition(WGD);

  // stats = new TracerParticle_Statistics(PID, PGD, this);
  // concentration = new TracerParticle_Concentration(PID, pm);

  QESFileOutput_v2 *outfile = nullptr;
  if (PGD->plumeParameters.plumeOutput) {
    outfile = new QESNetCDFOutput_v2(PGD->plumeParameters.outputFileBasename + "_" + tag + "_plumeOut.nc");
  }
  stats = new StatisticsDirector(PID, PGD, outfile);
  if (PID->colParams) {
    stats->attach("concentration", new TracerParticle_Concentration(PID->colParams, this));
  }
  // other statistics can be added here
}

void TracerParticle_Model::generateParticleList(QEStime &timeCurrent,
                                                const float &dt,
                                                WINDSGeneralData *WGD,
                                                TURBGeneralData *TGD,
                                                PLUMEGeneralData *PGD)
{
  int nbr_new_particle = 0;
  double time = timeCurrent - PGD->getSimTimeStart();
  for (auto source : sources) {
    nbr_new_particle += source->getNewParticleNumber(dt, time);
  }
  particles->sweep(nbr_new_particle);
  for (auto source : sources) {
    source->emitParticles(dt, time, particles);
  }

#pragma omp parallel for default(none) shared(WGD, TGD, PGD)
  for (auto k = 0u; k < particles->get_nbr_added(); ++k) {
    // set particle ID (use global particle counter)
    PGD->initializeParticleValues(particles->get_added(k), WGD, TGD);
  }
}

void TracerParticle_Model::addSources(std::vector<TracerParticle_Source *> newSources)
{
  sources.insert(sources.end(), newSources.begin(), newSources.end());
}

void TracerParticle_Model::advect(const double &total_time_interval,
                                  WINDSGeneralData *WGD,
                                  TURBGeneralData *TGD,
                                  PLUMEGeneralData *PGD)
{
#pragma omp parallel for default(none) shared(WGD, TGD, PGD, total_time_interval)
  for (auto k = 0u; k < particles->size(); ++k) {
    Particle *p = particles->get(k);
    double rhoAir = 1.225;// in kg m^-3
    double nuAir = 1.506E-5;// in m^2 s^-1

    // settling velocity
    double vs = 0;

    double timeRemainder = total_time_interval;
    while (p->isActive && timeRemainder > 0.0) {

      /*
        now get the Lagrangian values for the current iteration from the wind/turbulence grid
        will need to use the interp3D function
      */

      PGD->interp->interpValues(WGD, p->xPos, p->yPos, p->zPos, p->uMean, p->vMean, p->wMean);

      // adjusting mean vertical velocity for settling velocity
      p->wMean -= vs;

      // now calculate the particle timestep using the courant number, the velocity fluctuation from the last time,
      // and the grid sizes. Uses timeRemainder as the timestep if it is smaller than the one calculated from the Courant number
      int cellId = PGD->interp->getCellId(p->xPos, p->yPos, p->zPos);

      double dWall = WGD->mixingLengths[cellId];
      double par_dt = PGD->calcCourantTimestep(dWall,
                                               std::abs(p->uMean) + std::abs(p->uFluct),
                                               std::abs(p->vMean) + std::abs(p->vFluct),
                                               std::abs(p->wMean) + std::abs(p->wFluct),
                                               timeRemainder);

      // std::cout << "par_dt = " << par_dt << std::endl;
      //  update the par_time, useful for debugging
      // par_time = par_time + par_dt;

      PGD->GLE_solver(p, par_dt, TGD);

      if (p->isRogue) {
        p->isActive = false;
        nbr_rogue++;
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
        deposition->deposit(p, disX, disY, disZ, uTot, vTot, wTot, vs, WGD, TGD, PGD->interp);
      }

      // check and do wall (building and terrain) reflection (based in the method)
      if (p->isActive) {
        p->isActive = PGD->wallReflect->reflect(WGD, p->xPos, p->yPos, p->zPos, disX, disY, disZ, p->uFluct, p->vFluct, p->wFluct);
      }

      // now apply boundary conditions
      PGD->applyBC(p);

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
