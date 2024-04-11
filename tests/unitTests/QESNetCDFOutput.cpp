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
  std::vector<NcDim> dim_vect_t;
  dim_vect_t.push_back(NcDim_t);
  addField("t", "s", "time since start of simulation", dim_vect_t, ncDouble);
  output_dimensions.insert({ "t", NcDim_t });
  output_dimension_sets.insert({ "time", { NcDim_t } });

  NcDim_tstr = addDimension("dateStrLen", dateStrLen);
  std::vector<NcDim> dim_vect_tstr;
  dim_vect_tstr.push_back(NcDim_t);
  dim_vect_tstr.push_back(NcDim_tstr);
  addField("timestamp", "--", "date time using format: YYYY-MM-DDThh:mm:ss", dim_vect_tstr, ncChar);
  timestamp_out.resize(dateStrLen, '0');
}

void QESNetCDFOutput::setStartTime(const QEStime &in)
{
  timeStart = in;
  addAtt("t", "simulation_start", timeStart.getTimestamp());
  flagStartTimeSet = true;
}

void QESNetCDFOutput::setOutputTime(const QEStime &in)
{
  timeCurrent = in;
}

bool QESNetCDFOutput::validateFileOptions()
{

  if (all_output_fields.empty()) {
    std::cerr << "[QES-output] ERROR all output fields undefined -> cannot validate file options" << std::endl;
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
// create dimension
// -> int
void QESNetCDFOutput::newDimension(const std::string &name,
                                   const std::string &long_name,
                                   const std::string &units,
                                   std::vector<int> *data)
{
  if (output_dimensions.find(name) == output_dimensions.end()) {
    NcDim ncDim = addDimension(name, data->size());
    std::vector<NcDim> dimensions;
    dimensions.push_back(ncDim);
    addField(name, units, long_name, dimensions, ncInt);
    saveField2D(name, *data);
    output_dimensions.insert({ name, ncDim });
  } else {
    std::cerr << "[ERROR] Dimension already exits" << std::endl;
    exit(1);
  }
}
// -> float
void QESNetCDFOutput::newDimension(const std::string &name,
                                   const std::string &long_name,
                                   const std::string &units,
                                   std::vector<float> *data)
{
  if (output_dimensions.find(name) == output_dimensions.end()) {
    NcDim ncDim = addDimension(name, data->size());
    std::vector<NcDim> dimensions;
    dimensions.push_back(ncDim);
    addField(name, units, long_name, dimensions, ncFloat);
    saveField2D(name, *data);
    output_dimensions.insert({ name, ncDim });
  } else {
    std::cerr << "[ERROR] Dimension already exits" << std::endl;
    exit(1);
  }
}
// -> double
void QESNetCDFOutput::newDimension(const std::string &name,
                                   const std::string &long_name,
                                   const std::string &units,
                                   std::vector<double> *data)
{
  if (output_dimensions.find(name) == output_dimensions.end()) {
    NcDim ncDim = addDimension(name, data->size());
    std::vector<NcDim> dimensions;
    dimensions.push_back(ncDim);
    addField(name, units, long_name, dimensions, ncDouble);
    saveField2D(name, *data);
    output_dimensions.insert({ name, ncDim });
    // set_all_output_fields.insert(name);
  } else {
    std::cerr << "[ERROR] Dimension already exits" << std::endl;
    exit(1);
  }
}
// sets of dimension
void QESNetCDFOutput::newDimensionSet(const std::string &name,
                                      const std::vector<std::string> &dims)
{
  if (output_dimension_sets.find(name) == output_dimension_sets.end()) {
    std::vector<NcDim> dim_vect;
    for (auto s : dims) {
      if (output_dimensions.find(s) != output_dimensions.end()) {
        dim_vect.push_back(output_dimensions[s]);
      } else {
        std::cerr << "[ERROR] in set " << name << " dimension " << s << " does not exits" << std::endl;
        exit(1);
      }
    }
    output_dimension_sets.insert({ name, dim_vect });
  } else {
    if (output_dimension_sets[name].size() == dims.size()) {
      for (auto k = 0; k < output_dimension_sets[name].size(); ++k) {
        if (output_dimension_sets[name][k].getName() != dims[k]) {
          std::cerr << "[!!!ERROR!!!]\tSet of dimensions already exits and contain different dimensions" << std::endl;
          exit(1);
        } else {
          // dimension in set is compatible.
          // note: file cannot have 2 dimension with the same name
        }
      }
    } else {
      std::cerr << "[!!!ERROR!!!]\tSet of dimensions already exits and is not compatible" << std::endl;
      exit(1);
    }
  }
}

//----------------------------------------
// create attribute scalar
// -> int
void QESNetCDFOutput::newField(const std::string &name,
                               const std::string &long_name,
                               const std::string &units,
                               const std::string &dims,
                               int *data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  if (output_dimension_sets.find(dims) == output_dimension_sets.end()) {
    std::cerr << "[ERROR] Set of dimensions does not exit adding " << name << std::endl;
    exit(1);
  } else {
    output_object.emplace(name, new ObjScalarInt(data, name, long_name, units, { output_dimension_sets[dims] }));
    output_object[name]->add(this);
    // AttScalarInt att = { data, name, long_name, units, { output_dimensions[dims] } };
    // map_att_scalar_int.emplace(name, att);
    set_all_output_fields.insert(name);
    // addOutputFields({ name });
  }
}
// -> float
void QESNetCDFOutput::newField(const std::string &name,
                               const std::string &long_name,
                               const std::string &units,
                               const std::string &dims,
                               float *data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  if (output_dimension_sets.find(dims) == output_dimension_sets.end()) {
    std::cerr << "[ERROR] Set of dimensions does not exit adding " << name << std::endl;
    exit(1);
  } else {
    output_object.emplace(name, new ObjScalarFlt(data, name, long_name, units, { output_dimension_sets[dims] }));
    output_object[name]->add(this);
    // AttScalarFlt att = { data, name, long_name, units, { output_dimensions[dims] } };
    // map_att_scalar_flt.emplace(name, att);
    set_all_output_fields.insert(name);
    // addOutputFields({ name });
  }
}
// -> double
void QESNetCDFOutput::newField(const std::string &name,
                               const std::string &long_name,
                               const std::string &units,
                               const std::string &dims,
                               double *data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  if (output_dimension_sets.find(dims) == output_dimension_sets.end()) {
    std::cerr << "[ERROR] Set of dimensions does not exit adding " << name << std::endl;
    exit(1);
  } else {
    output_object.emplace(name, new ObjScalarDbl(data, name, long_name, units, { output_dimension_sets[dims] }));
    output_object[name]->add(this);
    // AttScalarDbl att = { data, name, long_name, units, { output_dimensions[dims] } };
    // map_att_scalar_dbl.emplace(name, att);
    set_all_output_fields.insert(name);
    // addOutputFields({ name });
  }
}

//----------------------------------------
// create attribute Vector
// -> int
void QESNetCDFOutput::newField(const std::string &name,
                               const std::string &long_name,
                               const std::string &units,
                               const std::string &dims,
                               std::vector<int> *data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  if (output_dimension_sets.find(dims) == output_dimension_sets.end()) {
    std::cerr << "[ERROR] Set of dimensions does not exit adding " << name << std::endl;
    exit(1);
  } else {
    output_object.emplace(name, new ObjVectorInt(data, name, long_name, units, output_dimension_sets[dims]));
    output_object[name]->add(this);
    // AttVectorInt att = { data, name, long_name, units, output_dimension_sets[dims] };
    // map_att_vector_int.emplace(name, att);
    set_all_output_fields.insert(name);
    // addOutputFields({ name });
  }
}
// -> float
void QESNetCDFOutput::newField(const std::string &name,
                               const std::string &long_name,
                               const std::string &units,
                               const std::string &dims,
                               std::vector<float> *data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  if (output_dimension_sets.find(dims) == output_dimension_sets.end()) {
    std::cerr << "[ERROR] Set of dimensions does not exit adding " << name << std::endl;
    exit(1);
  } else {
    output_object.emplace(name, new ObjVectorFlt(data, name, long_name, units, output_dimension_sets[dims]));
    output_object[name]->add(this);
    // AttVectorFlt att = { data, name, long_name, units, output_dimension_sets[dims] };
    // map_att_vector_flt.emplace(name, att);
    set_all_output_fields.insert(name);
    // addOutputFields({ name });
  }
}
// -> double
void QESNetCDFOutput::newField(const std::string &name,
                               const std::string &long_name,
                               const std::string &units,
                               const std::string &dims,
                               std::vector<double> *data)
{
  // FM -> here I do not know what is the best way to add the ref to data.
  if (output_dimension_sets.find(dims) == output_dimension_sets.end()) {
    std::cerr << "[ERROR] Set of dimensions does not exit adding " << name << std::endl;
    exit(1);
  } else {
    output_object.emplace(name, new ObjVectorDbl(data, name, long_name, units, output_dimension_sets[dims]));
    output_object[name]->add(this);
    // AttVectorDbl att = { data, name, long_name, units, output_dimension_sets[dims] };
    // map_att_vector_dbl.emplace(name, att);
    set_all_output_fields.insert(name);
    // addOutputFields({ name });
  }
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
  for (auto s : output_fields) {
    output_object[s]->add(this);
  }
}
//----------------------------------------
void QESNetCDFOutput::addOutputFields(const std::set<std::string> &new_fields)
{
  /*
   * This function add the  fields to the output vectors
   * and link them to the NetCDF.
   *
   * Since the type is not know, one needs to loop through
   * the 6 output vector to find it.
   *  - FMargairaz
   */


  // checking that all the fields are defined.
  std::set<std::string> result;
  std::set_difference(new_fields.begin(),
                      new_fields.end(),
                      set_all_output_fields.begin(),
                      set_all_output_fields.end(),
                      std::inserter(result, result.end()));

  if (!result.empty()) {
    std::cerr << "[ERROR] Fields ";
    for (const std::string &r : result) { std::cerr << r << " "; }
    std::cerr << "not defined" << std::endl;
    exit(1);
  }

  // create list of fields to save base on output_fields
  for (auto s : new_fields) {
    output_object[s]->add(this);
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
  output_object.erase(name);
}


void QESNetCDFOutput::newTimeEntry(QEStime &timeIn)
{
  timeCurrent = timeIn;
  output_counter = fields["t"].getDim(0).getSize();

  std::cout << "[TEST OUTPUT] " << fields["t"].getDim(0).getSize() << std::endl;
  std::cout << "[TEST OUTPUT] " << timeCurrent << std::endl;

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
}

void QESNetCDFOutput::saveOutputFields(QEStime &timeIn)
{
  /*
    This function save the fields from the output vectors
    Since the type is not know, one needs to loop through
    the 6 output vector to find it.

  FMargairaz
      */

  if (timeIn != timeCurrent) {
    std::cerr << "[ERROR] attempting to save with wrong timestamp" << std::endl;
    exit(1);
  }
  for (const auto &s : set_all_output_fields) {
    output_object[s]->save(this, output_counter);
  }
}

void QESNetCDFOutput::saveOutputFields(QEStime &timeIn, const std::vector<std::string> &fields)
{
  /*
    This function save the fields from the output vectors
    Since the type is not know, one needs to loop through
    the 6 output vector to find it.

  FMargairaz
      */
  if (timeIn != timeCurrent) {
    std::cerr << "[ERROR] attempting to save with wrong timestamp" << std::endl;
    exit(1);
  }
  for (const auto &s : fields) {
    output_object[s]->save(this, output_counter);
  }
}