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

/** @file QESPlumeOutputDirector.h
 * @brief
 */

#pragma once

#include <string>

#include "util/QESOutputDirector.h"
#include "util/QESFileOutput.h"
#include "util/QESOutputInterface.h"
#include "util/QEStime.h"

#include "PlumeOutput.h"
#include "PlumeOutputParticleData.h"

enum FileType {
  type1,
  type2
};

class PLUMEOutputDirector : public QESOutputDirector
{
public:
  PLUMEOutputDirector() = default;
  ~PLUMEOutputDirector() = default;

  void save(const QEStime &) override;
  void enroll(QESOutputInterface *, const std::string &, FileType);

private:
  std::vector<QESOutputInterface *> tmp1;
  std::vector<QESFileOutput *> files;
};
