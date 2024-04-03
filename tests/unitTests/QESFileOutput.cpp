//
// Created by Fabien Margairaz on 4/1/24.
//
//
#include "QESFileOutput.h"
#include "QESOutput.h"

void QESFileOutput::attach(QESOutputInterface *out)
{
  std::cout << "call attach" << std::endl;
  out->attach(this);
  m_list_output.push_back(out);
};