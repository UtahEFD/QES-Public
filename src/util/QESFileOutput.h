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
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file QESFileOutput.h */

#pragma once

#include <string>
#include <iostream>

#include "QEStime.h"


/**
 * @class QESFileOutput
 * @brief Handles the saving of output files.
 *
 * Attributes are created based on the type of the data:
 *   - Attributes are stored in map_att_*
 *   - All possible attributes available for the derived class should be created by its CTOR.
 *   - Attributes are pushed back to output_* based on what is selected by output_fields
 *   - The methods allow to be type generic (as long as the data is either int, float, or double)
 */

class QESFileOutput
{
public:
  explicit QESFileOutput() = default;
  virtual ~QESFileOutput() = default;

  /**
   * :document this:
   *
   * @note Can be called outside.
   */
  virtual void save(QEStime) = 0;
  virtual void save(float) = 0;

  virtual void setStartTime(const QEStime &) = 0;
  virtual void setOutputTime(const QEStime &) = 0;

  virtual void createDimension(const std::string &, const std::string &, const std::string &, std::vector<int> *) = 0;
  virtual void createDimension(const std::string &, const std::string &, const std::string &, std::vector<float> *) = 0;
  virtual void createDimension(const std::string &, const std::string &, const std::string &, std::vector<double> *) = 0;

  virtual void createDimensionSet(const std::string &, const std::vector<std::string> &) = 0;

  // create attribute scalar based on type of data
  virtual void createField(const std::string &, const std::string &, const std::string &, const std::string &, int *) = 0;
  virtual void createField(const std::string &, const std::string &, const std::string &, const std::string &, float *) = 0;
  virtual void createField(const std::string &, const std::string &, const std::string &, const std::string &, double *) = 0;

  // create attribute vector based on type of data
  virtual void createField(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<int> *) = 0;
  virtual void createField(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<float> *) = 0;
  virtual void createField(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<double> *) = 0;
};