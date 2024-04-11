//
// Created by Fabien Margairaz on 4/1/24.
//
#pragma once

#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <list>
#include <string>

#include "util/QEStime.h"

class DataSource;

class QESFileOutputInterface
{
public:
  explicit QESFileOutputInterface() = default;
  virtual ~QESFileOutputInterface() = default;

  virtual void attach(DataSource *) = 0;
  /**
   * :document this:
   *
   * @note Can be called outside.
   */

  virtual void newTimeEntry(QEStime &) = 0;

  virtual void save(QEStime &) = 0;
  virtual void save(float) = 0;
  
protected:
  virtual void setStartTime(const QEStime &) = 0;
  virtual void setOutputTime(const QEStime &) = 0;

  virtual void saveOutputFields(QEStime &) = 0;
  virtual void saveOutputFields(QEStime &, const std::vector<std::string> &) = 0;

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
};

class QESFileOutput : public QESFileOutputInterface
{
public:
  explicit QESFileOutput() = default;
  virtual ~QESFileOutput() = default;

  void attach(DataSource *) override;
  void save(QEStime &) override;

protected:
  std::list<DataSource *> m_list_output;

  friend DataSource;
};