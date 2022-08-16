#include <iostream>
#include <netcdf>
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include "winds/WINDSGeneralData.h"

#include "handleCompareArgs.h"

struct stats
{
  float maxDif;
  float avgDif;
  float totalDif;
  void getStats(std::string v)
  {
    std::cout << "stat on " << v << "\n"
              << "- total diff = " << totalDif << "\n"
              << "- max diff   = " << maxDif << "\n"
              << "- avg diff   = " << avgDif << "\n"
              << std::endl;
  }
};

stats compareVar(vector<float> &var1, vector<float> &var2);
stats compareVar(vector<int> &var1, vector<int> &var2);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  Args arguments;
  arguments.processArguments(argc, argv);

  WINDSGeneralData *WGD1 = new WINDSGeneralData(arguments.inputWINDSFile1);
  WINDSGeneralData *WGD2 = new WINDSGeneralData(arguments.inputWINDSFile2);

  stats comp;
  comp = compareVar(WGD1->terrain, WGD2->terrain);
  comp.getStats("terrain");

  for (int index = 0; index < WGD1->totalTimeIncrements; index++) {
    WGD1->loadNetCDFData(index);
    WGD2->loadNetCDFData(index);

    comp = compareVar(WGD1->u0, WGD2->u0);
    comp.getStats("u");

    comp = compareVar(WGD1->v0, WGD2->v0);
    comp.getStats("v");

    comp = compareVar(WGD1->w0, WGD2->w0);
    comp.getStats("w");

    comp = compareVar(WGD1->icellflag, WGD2->icellflag);
    comp.getStats("icellflag");
  }
  exit(EXIT_SUCCESS);
  return 0;
}


stats compareVar(vector<float> &var1, vector<float> &var2)
{
  stats out;
  out.maxDif = 0;
  out.avgDif = 0;
  out.totalDif = 0;

  float tmp = 0;

  for (size_t i = 0; i < var1.size(); ++i) {
    tmp = std::abs(var1[i] - var2[i]);
    if (tmp > out.maxDif) out.maxDif = tmp;
    out.totalDif += tmp;
  }

  out.avgDif = out.totalDif / var1.size();
  return out;
}

stats compareVar(vector<int> &var1, vector<int> &var2)
{
  stats out;
  out.maxDif = 0;
  out.avgDif = 0;
  out.totalDif = 0;

  float tmp = 0;
  for (size_t i = 0; i < var1.size(); ++i) {
    tmp = std::abs(var1[i] - var2[i]);
    if (tmp > out.maxDif) out.maxDif = tmp;
    out.totalDif += tmp;
  }

  out.avgDif = out.totalDif / var1.size();
  return out;
}
