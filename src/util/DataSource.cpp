/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Behnam Bozorgmehr
 * Copyright (c) 2025 Jeremy A. Gibbs
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Eric R. Pardyjak
 * Copyright (c) 2025 Zachary Patterson
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Lucas Ulmer
 * Copyright (c) 2025 Pete Willemsen
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
 * @file DataSource.cpp
 */

#include "DataSource.h"

void DataSource::attachToFile(QESFileOutput_Interface *out)
{
  m_output_file = out;
}

void DataSource::pushToFile(QEStime t)
{
  if (!m_pushed_to_file) {
    m_output_file->pushFieldsToFile(t, m_output_fields);
    m_pushed_to_file = true;
  }
}

void DataSource::notifyOfNewTimeEntry()
{
  // std::cout << "[DATA SOURCE] notified of new time entry" << std::endl;
  m_pushed_to_file = false;
}

void DataSource::defineDimension(const std::string &name,
                                 const std::string &long_name,
                                 const std::string &units,
                                 std::vector<int> *data)
{
  m_output_file->newDimension(name, long_name, units, data);
}

void DataSource::defineDimension(const std::string &name,
                                 const std::string &long_name,
                                 const std::string &units,
                                 std::vector<float> *data)
{
  m_output_file->newDimension(name, long_name, units, data);
}

void DataSource::defineDimension(const std::string &name,
                                 const std::string &long_name,
                                 const std::string &units,
                                 std::vector<double> *data)
{
  m_output_file->newDimension(name, long_name, units, data);
}

void DataSource::defineDimensionSet(const std::string &name, const std::vector<std::string> &dims)
{
  m_output_file->newDimensionSet(name, dims);
}

void DataSource::defineVariable(const std::string &name,
                                const std::string &long_name,
                                const std::string &units,
                                const std::string &dims,
                                int *data)
{
  m_output_file->newField(name, long_name, units, dims, data);
  m_output_fields.push_back(name);
}
void DataSource::defineVariable(const std::string &name,
                                const std::string &long_name,
                                const std::string &units,
                                const std::string &dims,
                                float *data)
{
  m_output_file->newField(name, long_name, units, dims, data);
  m_output_fields.push_back(name);
}

void DataSource::defineVariable(const std::string &name,
                                const std::string &long_name,
                                const std::string &units,
                                const std::string &dims,
                                double *data)
{
  m_output_file->newField(name, long_name, units, dims, data);
  m_output_fields.push_back(name);
}

void DataSource::defineVariable(const std::string &name,
                                const std::string &long_name,
                                const std::string &units,
                                const std::string &dims,
                                std::vector<int> *data)
{
  m_output_file->newField(name, long_name, units, dims, data);
  m_output_fields.push_back(name);
}

void DataSource::defineVariable(const std::string &name,
                                const std::string &long_name,
                                const std::string &units,
                                const std::string &dims,
                                std::vector<float> *data)
{
  m_output_file->newField(name, long_name, units, dims, data);
  m_output_fields.push_back(name);
}

void DataSource::defineVariable(const std::string &name,
                                const std::string &long_name,
                                const std::string &units,
                                const std::string &dims,
                                std::vector<double> *data)
{
  m_output_file->newField(name, long_name, units, dims, data);
  m_output_fields.push_back(name);
}
