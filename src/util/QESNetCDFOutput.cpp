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

/**
 * @file QESNetCDFOutput.cpp
 * @brief Handles the saving of output files.
 * Attributes are created based on the type of the data:
 *   - Attributes are stored in map_att_*
 *   - All possible attributes available for the derived class should be created by its CTOR.
 *   - Attributes are pushed back to output_* based on what is selected by output_fields
 *   - The methods allow to be type generic (as long as the data is either int, float, or double)
 */

#include "QESNetCDFOutput.h"

QESNetCDFOutput::QESNetCDFOutput(const std::string &output_file)
  : NetCDFOutput(output_file)
{
  NcDim_t = addDimension("t");
  NcDim_tstr = addDimension("dateStrLen", dateStrLen);

  std::vector<NcDim> dim_vect_t;
  dim_vect_t.push_back(NcDim_t);
  addField("t", "s", "time since start of simulation", dim_vect_t, ncDouble);

  std::vector<NcDim> dim_vect_tstr;
  dim_vect_tstr.push_back(NcDim_t);
  dim_vect_tstr.push_back(NcDim_tstr);
  addField("timestamp", "--", "date time using format: YYYY-MM-DDThh:mm:ss", dim_vect_tstr, ncChar);

  timestamp_out.resize(dateStrLen, '0');
}

void QESNetCDFOutput::setStartTime(QEStime in)
{
  timeStart = in;
  addAtt("t", "simulation_start", timeStart.getTimestamp());
  flagStartTimeSet = true;
}

bool QESNetCDFOutput::validateFileOptions()
{

  if (all_output_fields.empty()) {
    std::cerr << "[QES-output] ERROR all output fields undefine -> cannot validate file options" << std::endl;
    exit(EXIT_FAILURE);
  }

  // check if all fileOptions->outputFields are possible
  bool doContains(true);
  std::size_t iter = 0, maxiter = output_fields.size();

  while (doContains && iter < maxiter) {
    doContains = find(all_output_fields.begin(), all_output_fields.end(), output_fields.at(iter)) != all_output_fields.end();
    iter++;
  }

  return doContains;
}

//----------------------------------------
// create attribute scalar
// -> int
void QESNetCDFOutput::createAttScalar(const std::string &name,
                                      const std::string &long_name,
                                      const std::string &units,
                                      const std::vector<NcDim> &dims,
                                      int *data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttScalarInt att = { data, name, long_name, units, dims };
  map_att_scalar_int.emplace(name, att);
}
// -> float
void QESNetCDFOutput::createAttScalar(const std::string &name,
                                      const std::string &long_name,
                                      const std::string &units,
                                      const std::vector<NcDim> &dims,
                                      float *data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttScalarFlt att = { data, name, long_name, units, dims };
  map_att_scalar_flt.emplace(name, att);
}
// -> double
void QESNetCDFOutput::createAttScalar(const std::string &name,
                                      const std::string &long_name,
                                      const std::string &units,
                                      const std::vector<NcDim> &dims,
                                      double *data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttScalarDbl att = { data, name, long_name, units, dims };
  map_att_scalar_dbl.emplace(name, att);
}

//----------------------------------------
// create attribute Vector
// -> int
void QESNetCDFOutput::createAttVector(const std::string &name,
                                      const std::string &long_name,
                                      const std::string &units,
                                      const std::vector<NcDim> &dims,
                                      std::vector<int> *data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttVectorInt att = { data, name, long_name, units, dims };
  map_att_vector_int.emplace(name, att);
}
// -> float
void QESNetCDFOutput::createAttVector(const std::string &name,
                                      const std::string &long_name,
                                      const std::string &units,
                                      const std::vector<NcDim> &dims,
                                      std::vector<float> *data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttVectorFlt att = { data, name, long_name, units, dims };
  map_att_vector_flt.emplace(name, att);
}
// -> double
void QESNetCDFOutput::createAttVector(const std::string &name,
                                      const std::string &long_name,
                                      const std::string &units,
                                      const std::vector<NcDim> &dims,
                                      std::vector<double> *data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttVectorDbl att = { data, name, long_name, units, dims };
  map_att_vector_dbl.emplace(name, att);
}

// -> char (for time)
void QESNetCDFOutput::createAttVector(const std::string &name,
                                      const std::string &long_name,
                                      const std::string &units,
                                      const std::vector<NcDim> &dims,
                                      std::vector<char> *data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  AttVectorChar att = { data, name, long_name, units, dims };
  map_att_vector_char.emplace(name, att);
}

//----------------------------------------
void QESNetCDFOutput::addOutputFields()
{
  /*
    This function add the  fields to the output vectors
    and link them to the NetCDF.

    Since the type is not know, one needs to loop through
    the 6 output vector to find it.

    FMargairaz
  */

  // create list of fields to save base on output_fields
  for (size_t i = 0; i < output_fields.size(); i++) {
    std::string key = output_fields[i];

    if (map_att_scalar_int.count(key)) {
      // scalar int
      output_scalar_int.push_back(map_att_scalar_int[key]);
    } else if (map_att_scalar_flt.count(key)) {
      // scalar flt
      output_scalar_flt.push_back(map_att_scalar_flt[key]);
    } else if (map_att_scalar_dbl.count(key)) {
      // scalar dbl
      output_scalar_dbl.push_back(map_att_scalar_dbl[key]);
    } else if (map_att_vector_int.count(key)) {
      // vector int
      output_vector_int.push_back(map_att_vector_int[key]);
    } else if (map_att_vector_flt.count(key)) {
      // vector flt
      output_vector_flt.push_back(map_att_vector_flt[key]);
    } else if (map_att_vector_dbl.count(key)) {
      // vector dbl
      output_vector_dbl.push_back(map_att_vector_dbl[key]);
    } else if (map_att_vector_char.count(key)) {
      // vector char
      output_vector_char.push_back(map_att_vector_char[key]);
    }
  }

  // add scalar fields
  // -> int
  for (const AttScalarInt &att : output_scalar_int) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncInt);
  }
  // -> float
  for (const AttScalarFlt &att : output_scalar_flt) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncFloat);
  }
  // -> double
  for (const AttScalarDbl &att : output_scalar_dbl) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
  }
  // add vector fields
  // -> int
  for (const AttVectorInt &att : output_vector_int) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncInt);
  }
  // -> int
  for (const AttVectorFlt &att : output_vector_flt) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncFloat);
  }
  // -> double
  for (const AttVectorDbl &att : output_vector_dbl) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
  }
  // -> char
  for (const AttVectorChar &att : output_vector_char) {
    addField(att.name, att.units, att.long_name, att.dimensions, ncChar);
  }
}


void QESNetCDFOutput::rmOutputField(const std::string &name)
{
  /*
    This function remove a field from the output vectors
    Since the type is not know, one needs to loop through
    the 6 output vector to find it.

    Note: the field CANNOT be added again.

    FMargairaz
  */

  // loop through scalar fields to remove
  // -> int
  for (unsigned int i = 0; i < output_scalar_int.size(); i++) {
    if (output_scalar_int[i].name == name) {
      output_scalar_int.erase(output_scalar_int.begin() + i);
      return;
    }
  }
  // -> float
  for (unsigned int i = 0; i < output_scalar_flt.size(); i++) {
    if (output_scalar_flt[i].name == name) {
      output_scalar_flt.erase(output_scalar_flt.begin() + i);
      return;
    }
  }

  // -> double
  for (unsigned int i = 0; i < output_scalar_dbl.size(); i++) {
    if (output_scalar_dbl[i].name == name) {
      output_scalar_dbl.erase(output_scalar_dbl.begin() + i);
      return;
    }
  }

  // loop through vector fields to remove
  // -> int
  for (unsigned int i = 0; i < output_vector_int.size(); i++) {
    if (output_vector_int[i].name == name) {
      output_vector_int.erase(output_vector_int.begin() + i);
      return;
    }
  }
  // -> float
  for (unsigned int i = 0; i < output_vector_flt.size(); i++) {
    if (output_vector_flt[i].name == name) {
      output_vector_flt.erase(output_vector_flt.begin() + i);
      return;
    }
  }
  // -> double
  for (unsigned int i = 0; i < output_vector_dbl.size(); i++) {
    if (output_vector_dbl[i].name == name) {
      output_vector_dbl.erase(output_vector_dbl.begin() + i);
      return;
    }
  }
  // -> char
  for (unsigned int i = 0; i < output_vector_char.size(); i++) {
    if (output_vector_char[i].name == name) {
      output_vector_char.erase(output_vector_char.begin() + i);
      return;
    }
  }
}

void QESNetCDFOutput::rmTimeIndepFields()
{
  /*
    This function remove time indep field from the output vectors
    Since the types are not know, one needs to loop through
    the 6 output vector to find it.

    Note: the fields CANNOT be added again.

    FMargairaz
  */

  // loop through scalar fields to remove
  // -> int
  for (unsigned int i = 0; i < output_scalar_int.size(); i++) {
    if (output_scalar_int[i].dimensions[0].getName() != "t") {
      output_scalar_int.erase(output_scalar_int.begin() + i);
    }
  }
  // -> float
  for (unsigned int i = 0; i < output_scalar_flt.size(); i++) {
    if (output_scalar_flt[i].dimensions[0].getName() != "t") {
      output_scalar_flt.erase(output_scalar_flt.begin() + i);
    }
  }

  // -> double
  for (unsigned int i = 0; i < output_scalar_dbl.size(); i++) {
    if (output_scalar_dbl[i].dimensions[0].getName() != "t") {
      output_scalar_dbl.erase(output_scalar_dbl.begin() + i);
    }
  }

  // loop through vector fields to remove
  // -> int
  for (unsigned int i = 0; i < output_vector_int.size(); i++) {
    if (output_vector_int[i].dimensions[0].getName() != "t") {
      output_vector_int.erase(output_vector_int.begin() + i);
    }
  }
  // -> float
  for (unsigned int i = 0; i < output_vector_flt.size(); i++) {
    if (output_vector_flt[i].dimensions[0].getName() != "t") {
      output_vector_flt.erase(output_vector_flt.begin() + i);
    }
  }
  // -> double
  for (unsigned int i = 0; i < output_vector_dbl.size(); i++) {
    if (output_vector_dbl[i].dimensions[0].getName() != "t") {
      output_vector_dbl.erase(output_vector_dbl.begin() + i);
    }
  }
}

void QESNetCDFOutput::saveOutputFields()
{
  /*
    This function save the fields from the output vectors
    Since the type is not know, one needs to loop through
    the 6 output vector to find it.

    FMargairaz
  */

  // std::cout << fields["t"].getDim(0).getSize() << std::endl;

  if (output_counter == 0 && !flagStartTimeSet) {
    setStartTime(timeCurrent);
    time = 0.0;
  } else {
    time = timeCurrent - timeStart;
  }

  std::vector<size_t> time_index;
  std::vector<size_t> time_size;
  time_index = { static_cast<unsigned long>(output_counter) };
  saveField1D("t", time_index, &time);

  timeCurrent.getTimestamp(timestamp);
  // std::copy(timestamp.begin(), timestamp.end(), timestamp_out.begin());
  for (int i = 0; i < dateStrLen; ++i) {
    timestamp_out[i] = timestamp[i];
  }
  time_index = { static_cast<unsigned long>(output_counter), 0 };
  time_size = { 1, static_cast<unsigned long>(dateStrLen) };
  saveField2D("timestamp", time_index, time_size, timestamp_out);

  // loop through scalar fields to save
  // -> int
  for (unsigned int i = 0; i < output_scalar_int.size(); i++) {
    std::vector<size_t> scalar_index;
    scalar_index = { static_cast<unsigned long>(output_counter) };
    saveField1D(output_scalar_int[i].name, scalar_index, output_scalar_int[i].data);
  }
  // -> float
  for (unsigned int i = 0; i < output_scalar_flt.size(); i++) {
    std::vector<size_t> scalar_index;
    scalar_index = { static_cast<unsigned long>(output_counter) };
    saveField1D(output_scalar_flt[i].name, scalar_index, output_scalar_flt[i].data);
  }
  // -> double
  for (unsigned int i = 0; i < output_scalar_dbl.size(); i++) {
    std::vector<size_t> scalar_index;
    scalar_index = { static_cast<unsigned long>(output_counter) };
    saveField1D(output_scalar_dbl[i].name, scalar_index, output_scalar_dbl[i].data);
  }

  // loop through vector fields to save
  // -> int
  for (unsigned int i = 0; i < output_vector_int.size(); i++) {

    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;

    // if var is time dep -> special treatment for time
    if (output_vector_int[i].dimensions[0].getName() == "t") {
      vector_index.push_back(static_cast<size_t>(output_counter));
      vector_size.push_back(1);
      for (unsigned int d = 1; d < output_vector_int[i].dimensions.size(); d++) {
        int dim = output_vector_int[i].dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    }
    // if var not time dep -> use direct dimensions
    else if (output_counter == 0) {
      for (unsigned int d = 0; d < output_vector_int[i].dimensions.size(); d++) {
        int dim = output_vector_int[i].dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    } else {
      continue;
    }

    saveField2D(output_vector_int[i].name, vector_index, vector_size, *output_vector_int[i].data);
  }
  // -> float
  for (unsigned int i = 0; i < output_vector_flt.size(); i++) {
    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;

    // if var is time dep -> special treatment for time
    if (output_vector_flt[i].dimensions[0].getName() == "t") {
      vector_index.push_back(static_cast<size_t>(output_counter));
      vector_size.push_back(1);
      for (unsigned int d = 1; d < output_vector_flt[i].dimensions.size(); d++) {
        int dim = output_vector_flt[i].dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    }
    // if var not time dep -> use direct dimensions
    else if (output_counter == 0) {
      for (unsigned int d = 0; d < output_vector_flt[i].dimensions.size(); d++) {
        int dim = output_vector_flt[i].dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    } else {
      continue;
    }

    saveField2D(output_vector_flt[i].name, vector_index, vector_size, *output_vector_flt[i].data);
  }
  // -> double
  for (unsigned int i = 0; i < output_vector_dbl.size(); i++) {
    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;

    // if var is time dep -> special treatment for time
    if (output_vector_dbl[i].dimensions[0].getName() == "t") {
      vector_index.push_back(static_cast<size_t>(output_counter));
      vector_size.push_back(1);
      for (unsigned int d = 1; d < output_vector_dbl[i].dimensions.size(); d++) {
        int dim = output_vector_dbl[i].dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    }
    // if var not time dep -> use direct dimensions
    else if (output_counter == 0) {
      for (unsigned int d = 0; d < output_vector_dbl[i].dimensions.size(); d++) {
        int dim = output_vector_dbl[i].dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    } else {
      continue;
    }

    saveField2D(output_vector_dbl[i].name, vector_index, vector_size, *output_vector_dbl[i].data);
  }
  // -> Char for time
  for (unsigned int i = 0; i < output_vector_char.size(); i++) {
    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;

    // if var is time dep -> special treatment for time
    if (output_vector_char[i].dimensions[0].getName() == "t") {
      vector_index.push_back(static_cast<size_t>(output_counter));
      vector_size.push_back(1);
      for (unsigned int d = 1; d < output_vector_char[i].dimensions.size(); d++) {
        int dim = output_vector_char[i].dimensions[d].getSize();
        vector_index.push_back(0);
        vector_size.push_back(static_cast<unsigned long>(dim));
      }
    }

    saveField2D(output_vector_char[i].name, vector_index, vector_size, *output_vector_char[i].data);
  }

  output_counter++;
}
