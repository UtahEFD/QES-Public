//
// Created by Fabien Margairaz on 4/3/24.
//
#pragma once

#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <list>
#include <string>

#include "util/QEStime.h"
#include "QESFileOutput.h"

class DataSourceInterface
{
  /*
   * This class is the subject/component interface
   */
public:
  DataSourceInterface() = default;
  ~DataSourceInterface() = default;

  virtual void save(QEStime) = 0;

protected:
  virtual void setOutputFields() = 0;
  virtual void attach(QESFileOutput *) = 0;
};

class FileInterface
{
  /*
   * This class is the subject/component interface
   */
public:
  FileInterface() = default;
  ~FileInterface() = default;

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
  , public FileInterface
{
public:
  DataSource() = default;
  ~DataSource() = default;

  void save(QEStime t) override;

protected:
  void attach(QESFileOutput *out) override;

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
  QESFileOutput *m_output_file;
  std::vector<std::string> m_output_fields;

  friend QESFileOutput;
};
