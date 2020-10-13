#pragma once


#include "util/ParseInterface.h"

class URBInputData;
class URBGeneralData;


class TimeSeries : public ParseInterface
{
private:

public:

  int site_blayer_flag = 1;
  float site_z0;
  std::vector<float> site_wind_dir, site_z_ref, site_U_ref;
  float site_one_overL;
  float site_canopy_H, site_atten_coeff;



  virtual void parseValues()
  {
    parsePrimitive<int>(false, site_blayer_flag, "boundaryLayerFlag");
    parsePrimitive<float>(true, site_z0, "siteZ0");
    parsePrimitive<float>(true, site_one_overL, "reciprocal");
    parseMultiPrimitives<float>(true, site_z_ref, "height");
    parseMultiPrimitives<float>(true, site_U_ref, "speed");
    parseMultiPrimitives<float>(true, site_wind_dir, "direction");
    parsePrimitive<float>(false, site_canopy_H, "canopyHeight");
    parsePrimitive<float>(false, site_atten_coeff, "attenuationCoefficient");
  }

};
