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

/** @file QESOutputDirector
 * @brief This is an interface for output management
 */

#pragma once

#include <string>
#include <utility>

#include "util/QESOutputDirector.h"
#include "util/QESFileOutput.h"
#include "util/QESOutputInterface.h"
#include "QEStime.h"

class QESOutputDirector
{
public:
  QESOutputDirector(std::string name) : basename(std::move(name))
  {
  }
  ~QESOutputDirector() = default;

  virtual void save(const QEStime &) {}
  virtual void attach(QESOutputInterface *, const std::string &) {}

protected:
  QESOutputDirector() = default;

  std::string basename;
  std::vector<QESOutputInterface *> tmp1;
  std::vector<QESFileOutput *> files;
};