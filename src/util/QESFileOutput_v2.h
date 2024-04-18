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
 * @file QESFileOutput_v2.h
 */
#pragma once

#include <iostream>
#include <list>
#include <string>

#include "QEStime.h"

class DataSource;

class QESFileOutput_Interface
{
public:
  explicit QESFileOutput_Interface() = default;
  virtual ~QESFileOutput_Interface() = default;

  virtual void attachDataSource(DataSource *) = 0;
  /**
   * :document this:
   *
   * @note Can be called outside.
   */

  virtual void setStartTime(const QEStime &) = 0;
  virtual void newTimeEntry(const QEStime &) = 0;

  virtual void save(QEStime &) = 0;
  virtual void save(float) = 0;

protected:
  virtual void notifyDataSourcesOfNewTimeEntry() = 0;

  virtual void pushAllFieldsToFile(QEStime &) = 0;
  virtual void pushFieldsToFile(QEStime &, const std::vector<std::string> &) = 0;

  virtual void newDimension(const std::string &, const std::string &, const std::string &, std::vector<int> *) = 0;
  virtual void newDimension(const std::string &, const std::string &, const std::string &, std::vector<float> *) = 0;
  virtual void newDimension(const std::string &, const std::string &, const std::string &, std::vector<double> *) = 0;

  virtual void newDimensionSet(const std::string &, const std::vector<std::string> &) = 0;

  // new attribute scalar based on type of data
  virtual void newField(const std::string &, const std::string &, const std::string &, const std::string &, int *) = 0;
  virtual void newField(const std::string &, const std::string &, const std::string &, const std::string &, float *) = 0;
  virtual void newField(const std::string &, const std::string &, const std::string &, const std::string &, double *) = 0;

  // new attribute vector based on type of data
  virtual void newField(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<int> *) = 0;
  virtual void newField(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<float> *) = 0;
  virtual void newField(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<double> *) = 0;

  friend DataSource;
};

class QESFileOutput_v2 : public QESFileOutput_Interface
{
public:
  explicit QESFileOutput_v2() = default;
  virtual ~QESFileOutput_v2() = default;

  void attachDataSource(DataSource *) override;

  void save(QEStime &) override;
  void save(float t) override {}

  void notifyDataSourcesOfNewTimeEntry() override;

protected:
  std::list<DataSource *> m_list_data_source;
};

class QESNullOutput : public QESFileOutput_Interface
{
public:
  explicit QESNullOutput(const std::string &s)
  {
    std::cerr << "[!!!WARNING!!!]\toutput disabled for file: " << s << std::endl;
  }
  virtual ~QESNullOutput() = default;

  void attachDataSource(DataSource *s) override;

  void setStartTime(const QEStime &t) override {}
  void newTimeEntry(const QEStime &t) override {}

  void notifyDataSourcesOfNewTimeEntry() override {}
  void save(QEStime &t) override {}
  void save(float t) override {}

  virtual void pushAllFieldsToFile(QEStime &t) override {}
  virtual void pushFieldsToFile(QEStime &t, const std::vector<std::string> &s) override {}

  virtual void newDimension(const std::string &s1, const std::string &s2, const std::string &s3, std::vector<int> *d) override {}
  virtual void newDimension(const std::string &s1, const std::string &s2, const std::string &s3, std::vector<float> *d) override {}
  virtual void newDimension(const std::string &s1, const std::string &s2, const std::string &s3, std::vector<double> *d) override {}

  virtual void newDimensionSet(const std::string &s, const std::vector<std::string> &d) override {}

  // new attribute scalar based on type of data
  virtual void newField(const std::string &s1, const std::string &s2, const std::string &s3, const std::string &s4, int *d) override {}
  virtual void newField(const std::string &s1, const std::string &s2, const std::string &s3, const std::string &s4, float *d) override {}
  virtual void newField(const std::string &s1, const std::string &s2, const std::string &s3, const std::string &s4, double *d) override {}

  // new attribute vector based on type of data
  virtual void newField(const std::string &s1, const std::string &s2, const std::string &s3, const std::string &s4, std::vector<int> *d) override {}
  virtual void newField(const std::string &s1, const std::string &s2, const std::string &s3, const std::string &s4, std::vector<float> *d) override {}
  virtual void newField(const std::string &s1, const std::string &s2, const std::string &s3, const std::string &s4, std::vector<double> *d) override {}

protected:
  explicit QESNullOutput() = default;
};