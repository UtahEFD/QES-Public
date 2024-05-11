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

/** @file QESNetCDFOutput_v2.h */

#pragma once

#include <string>
#include <algorithm>
#include <vector>
#include <map>
#include <netcdf>

#include "QESFileOutput_v2.h"
#include "NetCDFOutput.h"
#include "QEStime.h"

using namespace netCDF;
using namespace netCDF::exceptions;

// Attribute for scalar/vector for each type

class Obj;

/**
 * @class QESNetCDFOutput_v2
 * @brief Handles the saving of output files.
 *
 * Attributes are created based on the type of the data:
 *   - Attributes are stored in map_att_*
 *   - All possible attributes available for the derived class should be created by its CTOR.
 *   - Attributes are pushed back to output_* based on what is selected by output_fields
 *   - The methods allow to be type generic (as long as the data is either int, float, or double)
 */

class QESNetCDFOutput_v2 : public NetCDFOutput
  , public QESFileOutput_v2
{
public:
  explicit QESNetCDFOutput_v2(const std::string &);
  virtual ~QESNetCDFOutput_v2()
  {}

  /**
   * :document this:
   *
   * @note Can be called outside.
   */

  void setStartTime(const QEStime &) override;
  void newTimeEntry(const QEStime &) override;

protected:
  QESNetCDFOutput_v2() = default;

  void pushAllFieldsToFile(QEStime &) override;
  void pushFieldsToFile(QEStime &, const std::vector<std::string> &) override;

  void newDimension(const std::string &, const std::string &, const std::string &, std::vector<int> *) override;
  void newDimension(const std::string &, const std::string &, const std::string &, std::vector<float> *) override;
  void newDimension(const std::string &, const std::string &, const std::string &, std::vector<double> *) override;

  void newDimensionSet(const std::string &, const std::vector<std::string> &) override;

  // create attribute scalar based on type of data
  void newField(const std::string &, const std::string &, const std::string &, const std::string &, int *) override;
  void newField(const std::string &, const std::string &, const std::string &, const std::string &, float *) override;
  void newField(const std::string &, const std::string &, const std::string &, const std::string &, double *) override;

  // create attribute vector based on type of data
  void newField(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<int> *) override;
  void newField(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<float> *) override;
  void newField(const std::string &, const std::string &, const std::string &, const std::string &, std::vector<double> *) override;

  // add fields based on output_fields
  void addOutputFields();
  void addOutputFields(const std::set<std::string> &);
  // removed field
  void rmOutputField(const std::string &);


private:
  /**< :document this: */
  NcDim NcDim_t;
  double time = 0;
  size_t output_counter = 0;

  /**< :document this: */
  NcDim NcDim_tstr;
  const int dateStrLen = 19;
  std::string timestamp;
  std::vector<char> timestamp_out;

  /**< :document this: */
  QEStime timeStart;
  bool flagStartTimeSet = false;
  QEStime timeCurrent;

  std::map<std::string, NcDim> output_dimensions;
  std::map<std::string, std::vector<NcDim>> output_dimension_sets;

  std::set<std::string> set_all_output_fields;

  std::vector<std::string> all_output_fields;
  std::vector<std::string> output_fields;
  /**< Vector containing fields to add to the NetCDF file
       @note This vector is used ONLY for creating fields
       (i.e. by the CTOR &add function) NOT to save them
       (i.e. by the function save) */

  std::map<std::string, Obj *> output_object;

  friend class ObjScalarInt;
  friend class ObjVectorInt;
  friend class ObjScalarFlt;
  friend class ObjVectorFlt;
  friend class ObjScalarDbl;
  friend class ObjVectorDbl;
};

class Obj
{
public:
  virtual void add(QESNetCDFOutput_v2 *) = 0;
  virtual void save(QESNetCDFOutput_v2 *, const size_t &) = 0;
};

class ObjScalarInt : public Obj
{
public:
  ObjScalarInt(int *d, std::string n, std::string l_n, std::string u, std::vector<NcDim> dims)
    : data(d), name(n), long_name(l_n), units(u), dimensions(dims)
  {}

  void add(QESNetCDFOutput_v2 *f) override
  {
    f->addField(name, units, long_name, dimensions, ncInt);
  }
  void save(QESNetCDFOutput_v2 *f, const size_t &output_counter) override
  {
    std::vector<size_t> scalar_index;
    scalar_index = { static_cast<unsigned long>(output_counter) };
    f->saveField1D(name, scalar_index, data);
  }

private:
  ObjScalarInt() = default;

  int *data{};
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};

class ObjVectorInt : public Obj
{
public:
  ObjVectorInt(std::vector<int> *d, std::string n, std::string l_n, std::string u, std::vector<NcDim> dims)
    : data(d), name(n), long_name(l_n), units(u), dimensions(dims)
  {}
  void add(QESNetCDFOutput_v2 *f) override
  {
    f->addField(name, units, long_name, dimensions, ncInt);
  }
  void save(QESNetCDFOutput_v2 *f, const size_t &output_counter) override
  {
    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;

    // if var is time dep -> special treatment for time
    if (dimensions[0].getName() == "t") {
      vector_index.push_back(static_cast<size_t>(output_counter));
      vector_size.push_back(1);
      for (unsigned int d = 1; d < dimensions.size(); d++) {
        size_t dim = dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    }
    // if var not time dep -> use direct dimensions
    else if (output_counter == 0) {
      for (unsigned int d = 0; d < dimensions.size(); d++) {
        size_t dim = dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    } else {
    }

    f->saveField2D(name, vector_index, vector_size, *data);
  }

private:
  ObjVectorInt() = default;

  std::vector<int> *data{};
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};

class ObjScalarFlt : public Obj
{
public:
  ObjScalarFlt(float *d, std::string n, std::string l_n, std::string u, std::vector<NcDim> dims)
    : data(d), name(n), long_name(l_n), units(u), dimensions(dims)
  {}
  void add(QESNetCDFOutput_v2 *f) override
  {
    f->addField(name, units, long_name, dimensions, ncFloat);
  }
  void save(QESNetCDFOutput_v2 *f, const size_t &output_counter) override
  {
    std::vector<size_t> scalar_index;
    scalar_index = { static_cast<unsigned long>(output_counter) };
    f->saveField1D(name, scalar_index, data);
  }

private:
  ObjScalarFlt() = default;
  float *data{};
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};

class ObjVectorFlt : public Obj
{
public:
  ObjVectorFlt(std::vector<float> *d, std::string n, std::string l_n, std::string u, std::vector<NcDim> dims)
    : data(d), name(n), long_name(l_n), units(u), dimensions(dims)
  {}
  void add(QESNetCDFOutput_v2 *f) override
  {
    f->addField(name, units, long_name, dimensions, ncFloat);
  }
  void save(QESNetCDFOutput_v2 *f, const size_t &output_counter) override
  {
    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;

    // if var is time dep -> special treatment for time
    if (dimensions[0].getName() == "t") {
      vector_index.push_back(static_cast<size_t>(output_counter));
      vector_size.push_back(1);
      for (unsigned int d = 1; d < dimensions.size(); d++) {
        size_t dim = dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    }
    // if var not time dep -> use direct dimensions
    else if (output_counter == 0) {
      for (unsigned int d = 0; d < dimensions.size(); d++) {
        size_t dim = dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    } else {
    }

    f->saveField2D(name, vector_index, vector_size, *data);
  }

private:
  ObjVectorFlt() = default;
  std::vector<float> *data{};
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};
class ObjScalarDbl : public Obj
{
public:
  ObjScalarDbl(double *d, std::string n, std::string l_n, std::string u, std::vector<NcDim> dims)
    : data(d), name(n), long_name(l_n), units(u), dimensions(dims)
  {}
  void add(QESNetCDFOutput_v2 *f) override
  {
    f->addField(name, units, long_name, dimensions, ncDouble);
  }
  void save(QESNetCDFOutput_v2 *f, const size_t &output_counter) override
  {
    std::vector<size_t> scalar_index;
    scalar_index = { static_cast<unsigned long>(output_counter) };
    f->saveField1D(name, scalar_index, data);
  }

private:
  ObjScalarDbl() = default;
  double *data{};
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};

class ObjVectorDbl : public Obj
{
public:
  ObjVectorDbl(std::vector<double> *d, std::string n, std::string l_n, std::string u, std::vector<NcDim> dims)
    : data(d), name(n), long_name(l_n), units(u), dimensions(dims)
  {}
  void add(QESNetCDFOutput_v2 *f) override
  {
    f->addField(name, units, long_name, dimensions, ncDouble);
  }
  void save(QESNetCDFOutput_v2 *f, const size_t &output_counter) override
  {
    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;

    // if var is time dep -> special treatment for time
    if (dimensions[0].getName() == "t") {
      vector_index.push_back(static_cast<size_t>(output_counter));
      vector_size.push_back(1);
      for (unsigned int d = 1; d < dimensions.size(); d++) {
        size_t dim = dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    }
    // if var not time dep -> use direct dimensions
    else if (output_counter == 0) {
      for (unsigned int d = 0; d < dimensions.size(); d++) {
        size_t dim = dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    } else {
    }

    f->saveField2D(name, vector_index, vector_size, *data);
  }

private:
  ObjVectorDbl() = default;
  std::vector<double> *data{};
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};
/*class ObjVectorChar : public Obj
{
  ObjVectorChar(std::vector<char> *d, std::string n, std::string l_n, std::string u, std::vector<NcDim> dims)
    : data(d), name(n), long_name(l_n), units(u), dimensions(dims)
  {}

  void save(QESNetCDFOutput_v2 *f, const int &output_counter) override
  {
    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;

    // if var is time dep -> special treatment for time
    if (dimensions[0].getName() == "t") {
      vector_index.push_back(static_cast<size_t>(output_counter));
      vector_size.push_back(1);
      for (unsigned int d = 1; d < dimensions.size(); d++) {
        int dim = dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    }
    // if var not time dep -> use direct dimensions
    else if (output_counter == 0) {
      for (unsigned int d = 0; d < dimensions.size(); d++) {
        int dim = dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    } else {
    }

    f->saveField2D(name, vector_index, vector_size, *data);
  }

private:
  ObjVectorChar() = default;
  std::vector<char> *data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};
*/
