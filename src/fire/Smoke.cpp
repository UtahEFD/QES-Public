// Smoke.cpp
// This class specifies the smoke sources for QES-Fire
// Matthew Moody 10/04/2023
//

#include "Smoke.h"
#include <fstream>
#include <iostream>
#include <vector>
#include "plume/Plume.hpp"

using namespace std;

Smoke ::Smoke(){
};

void Smoke ::genSmoke(WINDSGeneralData *WGD, Fire *fire, Plume *plume)
{
  // get domain information
  nx = WGD->nx;
  ny = WGD->ny;
  nz = WGD->nz;
  dx = WGD->dx;
  dy = WGD->dy;
  dz = WGD->dz;
  
  
  for (int j = 1; j < ny - 2; j++){
    for (int i = 1; i <nx - 2; i++){
      int idx = i+j*(nx-1);
      
      if (fire->smoke_flag[idx] == 1){
	//add source here
	// get location of source
	x_pos = i*dx;
	y_pos = j*dy;
	z_pos = WGD->terrain[idx]+1;
	ppt = 20;
	std::cout<<"x = "<<x_pos<<", y = "<<y_pos<<", z = "<<z_pos<<std::endl;
	SourceFire source = SourceFire(x_pos, y_pos, z_pos, ppt);
	source.setSource();
	std::vector<Source *> sourceList;
	sourceList.push_back(dynamic_cast<Source*>(&source));
	plume->addSources(sourceList);
	// turn off smoke flag so new source not added next time step
	fire->smoke_flag[idx] = 0;
	// clear add source vector
        
      }
      
    }
  }
  
}
