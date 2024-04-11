//
// Created by Fabien Margairaz on 4/1/24.
//
//
#include "QESFileOutput.h"
#include "DataSource.h"

void QESFileOutput::attach(DataSource *out)
{
  std::cout << "[FILE] call attach" << std::endl;
  out->attach(this);
  out->setOutputFields();
  m_list_output.push_back(out);
};

void QESFileOutput::save(QEStime &timeIn)
{
  std::cout << "[FILE] call all saves" << std::endl;
  for (auto o : m_list_output) {
    o->save(timeIn);
  }
};