//
// Created by Fabien Margairaz on 4/3/24.
//

#include "QESOutput.h"

void QESOutput::attach(QESFileOutputInterface *out)
{
  std::cout << "[OUTPUT SET] call attach" << std::endl;
  out->attach(this);
  m_output_file = out;
}

void QESOutput::save(QEStime t)
{
  m_output_file->saveOutputFields(t, output_fields);
}
