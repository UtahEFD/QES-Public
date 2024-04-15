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
 * @file QESNetCDFOutput_v2.cpp
 * @brief Handles the saving of output files.
 * Attributes are created based on the type of the data:
 *   - Attributes are stored in map_att_*
 *   - All possible attributes available for the derived class should be created by its CTOR.
 *   - Attributes are pushed back to output_* based on what is selected by output_fields
 *   - The methods allow to be type generic (as long as the data is either int, float, or double)
 */

#include "QESNetCDFOutput_v2.h"

QESNetCDFOutput_v2::QESNetCDFOutput_v2(const std::string &output_file)
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

void QESNetCDFOutput_v2::setStartTime(const QEStime &in)
{
  timeStart = in;
  addAtt("t", "simulation_start", timeStart.getTimestamp());
  flagStartTimeSet = true;
}

void QESNetCDFOutput_v2::newTimeEntry(const QEStime &timeIn)
{
  if (timeCurrent == timeIn) {
    // std::cerr << "[!!!WARNING!!!]\ttime for new entry already exists" << std::endl;
    return;
  }

  timeCurrent = timeIn;
  output_counter = fields["t"].getDim(0).getSize();

  // check if start time is define (needed for "time")
  if (output_counter == 0 && !flagStartTimeSet) {
    std::cerr << "[!!!WARNING!!!]\tstart time not defined in output -> use first time entry as start time" << std::endl;
    setStartTime(timeCurrent);
    time = 0.0;
  } else {
    time = timeCurrent - timeStart;
  }

  // push time to file
  std::vector<size_t> time_index;
  std::vector<size_t> time_size;
  time_index = { static_cast<unsigned long>(output_counter) };
  saveField1D("t", time_index, &time);

  // push timestamp to file (note: timestamp is char[])
  timeCurrent.getTimestamp(timestamp);
  // std::copy(timestamp.begin(), timestamp.end(), timestamp_out.begin());
  for (int i = 0; i < dateStrLen; ++i) {
    timestamp_out[i] = timestamp[i];
  }
  time_index = { static_cast<unsigned long>(output_counter), 0 };
  time_size = { 1, static_cast<unsigned long>(dateStrLen) };
  saveField2D("timestamp", time_index, time_size, timestamp_out);

  // notify all data sources of new time entry (reset push2file flags)
  notifyDataSourcesOfNewTimeEntry();
}

//----------------------------------------
// create dimension
// -> int
void QESNetCDFOutput_v2::newDimension(const std::string &name,
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
void QESNetCDFOutput_v2::newDimension(const std::string &name,
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
void QESNetCDFOutput_v2::newDimension(const std::string &name,
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
void QESNetCDFOutput_v2::newDimensionSet(const std::string &name,
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
void QESNetCDFOutput_v2::newField(const std::string &name,
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
    set_all_output_fields.insert(name);
  }
}
// -> float
void QESNetCDFOutput_v2::newField(const std::string &name,
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
    set_all_output_fields.insert(name);
  }
}
// -> double
void QESNetCDFOutput_v2::newField(const std::string &name,
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
    set_all_output_fields.insert(name);
  }
}

//----------------------------------------
// create attribute Vector
// -> int
void QESNetCDFOutput_v2::newField(const std::string &name,
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
    set_all_output_fields.insert(name);
  }
}
// -> float
void QESNetCDFOutput_v2::newField(const std::string &name,
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
    set_all_output_fields.insert(name);
  }
}
// -> double
void QESNetCDFOutput_v2::newField(const std::string &name,
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
    set_all_output_fields.insert(name);
  }
}


//----------------------------------------
void QESNetCDFOutput_v2::addOutputFields()
{
  // create list of fields to save base on output_fields
  for (auto s : output_fields) {
    output_object[s]->add(this);
  }
}
//----------------------------------------
void QESNetCDFOutput_v2::addOutputFields(const std::set<std::string> &new_fields)
{
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


void QESNetCDFOutput_v2::rmOutputField(const std::string &name)
{
  // remove object from the map
  output_object.erase(name);
}

void QESNetCDFOutput_v2::pushAllFieldsToFile(QEStime &timeIn)
{

  if (timeIn != timeCurrent) {
    std::cerr << "[ERROR] attempting to save with wrong timestamp" << std::endl;
    exit(1);
  }
  for (const auto &s : set_all_output_fields) {
    output_object[s]->save(this, output_counter);
  }
}

void QESNetCDFOutput_v2::pushFieldsToFile(QEStime &timeIn, const std::vector<std::string> &fields)
{

  if (timeIn != timeCurrent) {
    std::cerr << "[ERROR] attempting to save with wrong timestamp" << std::endl;
    exit(1);
  }
  for (const auto &s : fields) {
    output_object[s]->save(this, output_counter);
  }
}