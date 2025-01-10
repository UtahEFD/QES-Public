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

/** @file Sources.hpp
 * @brief This class contains data and variables that set flags and
 * settngs read from the xml.
 *
 * @note Child of ParseInterface
 * @sa ParseInterface
 */

#pragma once

#include "util/ParseInterface.h"

#include "PI_SourceComponent.hpp"
#include "PI_SourceGeometry_Cube.hpp"
#include "PI_SourceGeometry_FullDomain.hpp"
#include "PI_SourceGeometry_Line.hpp"
#include "PI_SourceGeometry_Point.hpp"
#include "PI_SourceGeometry_SphereShell.hpp"

#include "PI_ReleaseType.hpp"
#include "PI_ReleaseType_instantaneous.hpp"
#include "PI_ReleaseType_continuous.hpp"
#include "PI_ReleaseType_duration.hpp"

#include "Source.h"

class PI_Source : public ParseInterface,
                  public SourceBuilderInterface
{
private:
protected:
public:
  PI_SourceComponent *m_sourceGeometry{};
  PI_ReleaseType *m_releaseType{};

public:
  // constructor
  PI_Source() = default;

  // destructor
  virtual ~PI_Source() = default;

  void setReleaseType();
  void setSourceGeometry();

  void parseValues() override
  {
    setReleaseType();
    setSourceGeometry();
  }

  void initialize(const float &timestep);

  Source *create(QESDataTransport &data) override;
};
