//
// Created by Fabien Margairaz on 4/3/24.
//

#include "QESOutput.h"

void QESOutput::attach(QESFileOutputInterface *out)
{
  m_output_file = out;
}

void QESOutput::save(QEStime t)
{
  m_output_file->save(t);
}
