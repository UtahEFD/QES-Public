//
// Created by Fabien Margairaz on 4/3/24.
//

#include "DataSource.h"

void DataSource::attach(QESFileOutput *out)
{
  std::cout << "[DATA SOURCE] call attach" << std::endl;
  m_output_file = out;
}

void DataSource::save(QEStime t)
{
  m_output_file->saveOutputFields(t, m_output_fields);
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
