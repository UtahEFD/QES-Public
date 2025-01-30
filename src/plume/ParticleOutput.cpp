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

/** @file ParticleOutput.cpp
 */

#include "ParticleOutput.h"
#include "PLUMEGeneralData.h"

ParticleOutput::ParticleOutput(PI_ParticleOutputParameters *params, PLUMEGeneralData *PGD)
{
  // output frequency
  outputFrequency = params->outputFrequency;
  // time to start output
  if (params->outputStartTime < 0) {
    outputStartTime = PGD->getSimTimeStart();
    // set the initial next output time value
    nextOutputTime = outputStartTime + outputFrequency;
  } else {
    outputStartTime = PGD->getSimTimeStart() + params->outputStartTime;
    // set the initial next output time value
    nextOutputTime = outputStartTime;
  }
}

void ParticleOutput::save(QEStime &t, PLUMEGeneralData *PGD)
{
  if (t >= nextOutputTime) {
    for (const auto &pm : PGD->models) {
      pm.second->accept(new ExportParticleData(t, PGD));
    }
    nextOutputTime = nextOutputTime + outputFrequency;
  }
}

ExportParticleData::ExportParticleData(QEStime &t, PLUMEGeneralData *PGD)
  : time(t)
{
  fname_prefix = PGD->plumeParameters.outputFileBasename;
  fname_suffix = "particleData_";
  fname_suffix.append(time.getTimestamp());
  fname_suffix.append(".csv");

  file_prologue.append("# Simulation start time: " + PGD->getSimTimeStart().getTimestamp() + "\n");
  file_prologue.append("# Simulation current time: " + time.getTimestamp() + "\n");
}

void ExportParticleData::visit(ParticleModel *pm)
{
  std::string sout = fname_prefix + "_" + pm->get_tag() + "_" + fname_suffix;

  // opens an existing csv file or creates a new file.
  fstream fout;
  fout.open(sout, ios::out);
  fout << "# Particle Data File\n";
  fout << "# Model particle tag: " << pm->get_tag() << "\n";
  fout << file_prologue;
  fout << "#particleID,tStrt,sourceIdx,d,m,wdecay,"
       << "xPos_init,yPos_init,"
          "zPos_init,xPos,yPos,zPos,"
          "uFluct,vFluct,wFluct,"
          "delta_uFluct,delta_vFluct,delta_wFluct\n";

  for (auto k = 0u; k < pm->particles_control.size(); ++k) {// Insert the data to file
    if (pm->particles_control[k].state == ACTIVE) {
      fout << pm->particles_core[k].ID << ", "
           << pm->particles_metadata[k].time_start << ", "
           << pm->particles_metadata[k].source_id << ", "
           << pm->particles_core[k].d << ", "
           << pm->particles_core[k].m << ", "
           << pm->particles_core[k].wdecay << ", "
           << pm->particles_metadata[k].pos_init._1 << ", "
           << pm->particles_metadata[k].pos_init._2 << ", "
           << pm->particles_metadata[k].pos_init._3 << ", "
           << pm->particles_core[k].pos._1 << ", "
           << pm->particles_core[k].pos._2 << ", "
           << pm->particles_core[k].pos._3 << ", "
           << pm->particles_lsdm[k].velFluct._1 << ", "
           << pm->particles_lsdm[k].velFluct._2 << ", "
           << pm->particles_lsdm[k].velFluct._3 << ", "
           << pm->particles_lsdm[k].delta_velFluct._1 << ", "
           << pm->particles_lsdm[k].delta_velFluct._2 << ", "
           << pm->particles_lsdm[k].delta_velFluct._3 << "\n";
    }
  }

  fout.close();
}

/*void ExportParticleData::visit(TracerParticle_Model *pm)
{
  std::string sout = fname_prefix + "_" + pm->get_tag() + "_" + fname_suffix;

  // opens an existing csv file or creates a new file.
  fstream fout;
  fout.open(sout, ios::out);
  fout << "# Particle Data File\n";
  fout << "# Model particle tag: " << pm->get_tag() << "\n";
  fout << file_prologue;
  fout << "#particleID,tStrt,sourceIdx,"
       << "d,m,wdecay,"
       << "xPos_init,yPos_init,"
          "zPos_init,xPos,yPos,zPos,"
          "uFluct,vFluct,wFluct,"
          "delta_uFluct,delta_vFluct,delta_wFluct\n";

  for (auto k = 0u; k < pm->get_particles()->size(); ++k) {// Insert the data to file
    TracerParticle *p = pm->get_particles()->get_ptr(k);
    if (p->state == ACTIVE) {
      fout << p->ID << ", "
           << p->tStrt << ", "
           << p->sourceIdx << ", "
           << p->d << ", "
           << p->m << ", "
           << p->wdecay << ", "
           << p->pos_init._1 << ", "
           << p->pos_init._2 << ", "
           << p->pos_init._3 << ", "
           << p->pos._1 << ", "
           << p->pos._2 << ", "
           << p->pos._3 << ", "
           << p->velFluct._1 << ", "
           << p->velFluct._2 << ", "
           << p->velFluct._3 << ", "
           << p->delta_velFluct._1 << ", "
           << p->delta_velFluct._2 << ", "
           << p->delta_velFluct._3 << "\n";
    }
  }

  fout.close();
}*/
