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

/**
 * @file DataSource.h
 */

#pragma once


#include <iostream>
#include <list>
#include <string>

#include "QEStime.h"
#include "QESFileOutput_v2.h"

class DataSourceInterface
{
  /*
   * This class is the subject/component interface
   */
public:
  DataSourceInterface() = default;
  virtual ~DataSourceInterface() = default;

  virtual void prepareDataAndPushToFile(QEStime) = 0;

  virtual void collect(QEStime &, const float &){};
  virtual void finalize(QEStime &){};
  virtual void reset(){};

protected:
  virtual void setOutputFields() = 0;
  virtual void attachToFile(QESFileOutput_Interface *) = 0;
  virtual void pushToFile(QEStime) = 0;
  virtual void notifyOfNewTimeEntry() = 0;

  virtual void defineDimension(const std::string &, const std::string &, const std::string &, std::vector<int> *) = 0;
  virtual void defineDimension(const std::string &, const std::string &, const std::string &, std::vector<float> *) = 0;
  virtual void defineDimension(const std::string &, const std::string &, const std::string &, std::vector<double> *) = 0;

  virtual void defineDimensionSet(const std::string &, const std::vector<std::string> &) = 0;

  // create attribute scalar based on type of data
  virtual void defineVariable(const std::string &, const std::string &, const std::string &, const std::string &, int *) = 0;
  virtual void defineVariable(const std::string &, const std::string &, const std::string &, const std::string &, float *) = 0;
  virtual void defineVariable(const std::string &, const std::string &, const std::string &, const std::string &, double *) = 0;

  // create attribute vector based on type of data
  virtual void defineVariable(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<int> *) = 0;
  virtual void defineVariable(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<float> *) = 0;
  virtual void defineVariable(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<double> *) = 0;
};

class DataSource : public DataSourceInterface
{
public:
  DataSource() = default;
  virtual ~DataSource() = default;

protected:
  void pushToFile(QEStime t) override;
  void attachToFile(QESFileOutput_Interface *out) override;
  void notifyOfNewTimeEntry() override;

  void defineDimension(const std::string &, const std::string &, const std::string &, std::vector<int> *) override;
  void defineDimension(const std::string &, const std::string &, const std::string &, std::vector<float> *) override;
  void defineDimension(const std::string &, const std::string &, const std::string &, std::vector<double> *) override;

  void defineDimensionSet(const std::string &, const std::vector<std::string> &) override;

  // create attribute scalar based on type of data
  void defineVariable(const std::string &, const std::string &, const std::string &, const std::string &, int *) override;
  void defineVariable(const std::string &, const std::string &, const std::string &, const std::string &, float *) override;
  void defineVariable(const std::string &, const std::string &, const std::string &, const std::string &, double *) override;

  // create attribute vector based on type of data
  void defineVariable(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<int> *) override;
  void defineVariable(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<float> *) override;
  void defineVariable(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<double> *) override;

private:
  QESFileOutput_Interface *m_output_file{};
  std::vector<std::string> m_output_fields{};

  bool m_pushed_to_file = false;

  friend QESFileOutput_v2;
  friend QESNullOutput;
};
