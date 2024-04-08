//
// Created by Fabien Margairaz on 4/1/24.
//
//
#include "QESFileOutput.h"
#include "QESOutput.h"

void QESFileOutput::attach(QESOutputInterface *out)
{
  std::cout << "[FILE] call attach" << std::endl;
  m_list_output.push_back(out);
};

void QESFileOutput::save(QEStime &timeIn)
{
  std::cout << "[FILE] call all saves" << std::endl;
  for (auto o : m_list_output) {
    o->save(timeIn);
  }
};