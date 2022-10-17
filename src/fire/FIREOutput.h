#pragma once

#include <string>
#include <vector>
#include <algorithm>

#include "winds/WINDSGeneralData.h"
#include "winds/WINDSInputData.h"
#include "util/QESNetCDFOutput.h"
#include "util/QEStime.h"
#include "Fire.h"

/* Specialized output classes derived from QESNetCDFOutput for 
   cell center data (used primarly for vizualization)
*/
class FIREOutput : public QESNetCDFOutput
{
public:
  FIREOutput(WINDSGeneralData *, Fire *, std::string);
  ~FIREOutput()
  {}

  void save(QEStime);


private:
  // data container for output (on cell-center without ghost cell)
  std::vector<float> x_out, y_out, z_out;
  std::vector<int> icellflag_out;
  std::vector<float> u_out, v_out, w_out;

  // copy of pointer for data access
  WINDSGeneralData *wgd_;
  Fire *fire_;
};
