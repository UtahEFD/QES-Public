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

class QESOutputInterface
{
  /*
   * This class is the subject/component interface
   */
public:
  QESOutputInterface() = default;
  ~QESOutputInterface() = default;

  virtual void attach(QESFileOutput *) = 0;
  virtual void setOutputFields() = 0;
  virtual void save(QEStime) = 0;
};

class QESOutput : public QESOutputInterface
{
public:
  void attach(QESFileOutput *out) override;
  void save(QEStime t) override;

protected:
  QESFileOutput *m_output_file;
  std::vector<std::string> output_fields;
};
