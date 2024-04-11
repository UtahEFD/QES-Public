//
// Created by Fabien Margairaz on 4/3/24.
//

#include "QESOutput.h"

void QESOutput::attach(QESFileOutput *out)
{
  std::cout << "[OUTPUT SET] call attach" << std::endl;
  m_output_file = out;
}

void QESOutput::save(QEStime t)
{
  m_output_file->saveOutputFields(t, output_fields);
}
