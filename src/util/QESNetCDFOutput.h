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

/** @file QESNetCDFOutput.h */

#pragma once

#include <string>
#include <algorithm>
#include <vector>
#include <map>
#include <netcdf>

#include "NetCDFOutput.h"
#include "QEStime.h"

using namespace netCDF;
using namespace netCDF::exceptions;

// Attribute for scalar/vector for each type
struct AttScalarInt
{
  int *data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};
struct AttVectorInt
{
  std::vector<int> *data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};

struct AttScalarFlt
{
  float *data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};
struct AttVectorFlt
{
  std::vector<float> *data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};
struct AttScalarDbl
{
  double *data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};
struct AttVectorDbl
{
  std::vector<double> *data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};
struct AttVectorChar
{
  std::vector<char> *data;
  std::string name;
  std::string long_name;
  std::string units;
  std::vector<NcDim> dimensions;
};

/**
 * @class QESNetCDFOutput
 * @brief Handles the saving of output files.
 *
 * Attributes are created based on the type of the data:
 *   - Attributes are stored in map_att_*
 *   - All possible attributes available for the derived class should be created by its CTOR.
 *   - Attributes are pushed back to output_* based on what is selected by output_fields
 *   - The methods allow to be type generic (as long as the data is either int, float, or double)
 */

class QESNetCDFOutput : public NetCDFOutput
{
public:
  explicit QESNetCDFOutput(const std::string &);
  virtual ~QESNetCDFOutput()
  {}

  /**
   * :document this:
   *
   * @note Can be called outside.
   */
  virtual void save(QEStime) = 0;
  virtual void save(float) {}

protected:
  QESNetCDFOutput() {}

  // create attribute scalar based on type of data
  void createAttScalar(const std::string &, const std::string &, const std::string &, const std::vector<NcDim> &, int *);
  void createAttScalar(const std::string &, const std::string &, const std::string &, const std::vector<NcDim> &, float *);
  void createAttScalar(const std::string &, const std::string &, const std::string &, const std::vector<NcDim> &, double *);

  // create attribute vector based on type of data
  void createAttVector(const std::string &, const std::string &, const std::string &, const std::vector<NcDim> &, std::vector<int> *);
  void createAttVector(const std::string &, const std::string &, const std::string &, const std::vector<NcDim> &, std::vector<float> *);
  void createAttVector(const std::string &, const std::string &, const std::string &, const std::vector<NcDim> &, std::vector<double> *);
  void createAttVector(const std::string &, const std::string &, const std::string &, const std::vector<NcDim> &, std::vector<char> *);

  void setStartTime(QEStime);

  // add fields based on output_fields
  void addOutputFields();
  // removed field
  void rmOutputField(const std::string &);
  void rmTimeIndepFields();
  // save fields
  void saveOutputFields();

  virtual void setAllOutputFields()
  {}
  virtual bool validateFileOptions();

  NcDim NcDim_t;
  NcDim NcDim_tstr;
  const int dateStrLen = 19; /**< :document this: */
  QEStime timeStart;
  QEStime timeCurrent;
  bool flagStartTimeSet = false;
  double time = 0; /**< :document this: */

  std::vector<std::string> all_output_fields;
  std::vector<std::string> output_fields;
  /**< Vector containing fields to add to the NetCDF file
       @note This vector is used ONLY for creating fields
       (i.e. by the CTOR &add function) NOT to save them
       (i.e. by the function save) */

  ///@{
  /**
   * Output field in the NetCDF file for scalar/vector for each type.
   *
   * @note This is used ONLY to create and link fields.
   */
  std::map<std::string, AttScalarInt> map_att_scalar_int;
  std::map<std::string, AttScalarFlt> map_att_scalar_flt;
  std::map<std::string, AttScalarDbl> map_att_scalar_dbl;
  std::map<std::string, AttVectorInt> map_att_vector_int;
  std::map<std::string, AttVectorFlt> map_att_vector_flt;
  std::map<std::string, AttVectorDbl> map_att_vector_dbl;
  std::map<std::string, AttVectorChar> map_att_vector_char;
  ///@}

  ///@{
  /**
   * Vectors of output fields in the NetCDF file for scalar/vector for each type.
   *
   * @note This is used to save the fields, ONLY the fields in these 6 vectors will be saved.
   */
  std::vector<AttScalarInt> output_scalar_int;
  std::vector<AttScalarFlt> output_scalar_flt;
  std::vector<AttScalarDbl> output_scalar_dbl;
  std::vector<AttVectorInt> output_vector_int;
  std::vector<AttVectorFlt> output_vector_flt;
  std::vector<AttVectorDbl> output_vector_dbl;
  std::vector<AttVectorChar> output_vector_char;
  ///@}
private:
  std::string timestamp;
  std::vector<char> timestamp_out; /**< :document this: */
  int output_counter = 0; /**< :document this: */
};
