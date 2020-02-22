#pragma once

#include <string>

#include "URBGeneralData.h"
#include "NetCDFOutputGeneric.h"

/* Specialized output classes derived from NetCDFOutputGeneric for 
   face center data (used for turbulence,...)
*/
class URBOutputWorkspace : public NetCDFOutputGeneric
{
public:
  URBOutputWorkspace()
    : NetCDFOutputGeneric()
  {}
  
  URBOutputWorkspace(URBGeneralData*,std::string);
  ~URBOutputWorkspace()	       
  {}
  
  //save function be call outside 
  void save(float);
 
private:

  std::vector<float> x_cc,y_cc,z_cc;

  URBGeneralData* ugd_;

  // Building data
  void setBuildingFields(NcDim*,NcDim*);
  void getBuildingFields();
  
  bool buildingFieldsSet = false;

  std::vector<float> building_rotation,canopy_rotation;

  std::vector<float> L,W,H;
  std::vector<float> length_eff,width_eff,height_eff,base_height; 
  std::vector<float> building_cent_x, building_cent_y;
  
  std::vector<int> i_start, i_end, j_start, j_end, k_end,k_start;
  std::vector<int> i_cut_start, i_cut_end, j_cut_start, j_cut_end, k_cut_end;
  std::vector<int> i_building_cent, j_building_cent;
  
  std::vector<float> upwind_dir,Lr;

};
