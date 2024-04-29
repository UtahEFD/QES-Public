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

/** @file ParticleOutput.h
 */

#pragma once

#include "util/QEStime.h"

#include "PI_ParticleOutputParameters.hpp"
#include "ParticleModel_Visitor.h"

#include "TracerParticle_Model.h"
#include "TracerParticle_Source.h"

#include "HeavyParticle_Model.h"
#include "HeavyParticle_Source.h"

class PLUMEGeneralData;

class ParticleOutput
{
public:
  ParticleOutput(PI_ParticleOutputParameters *, PLUMEGeneralData *);
  ~ParticleOutput() = default;

  void save(QEStime &, PLUMEGeneralData *);

private:
  // time to start output
  QEStime outputStartTime;
  // output frequency
  float outputFrequency;
  // next output time value that is updated each time save is called and there is output
  QEStime nextOutputTime;
};

class ExportParticleData : public ParticleModel_Visitor
{
public:
  ExportParticleData(QEStime &t, PLUMEGeneralData *);

  ~ExportParticleData() = default;

  void visit(TracerParticle_Model *element) override;
  void visit(HeavyParticle_Model *element) override;

private:
  std::string fname_prefix, fname_suffix;
  QEStime time;
  std::string file_prologue;
};
