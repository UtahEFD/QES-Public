/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Matthew Moody
 * Copyright (c) 2025 Jeremy Gibbs
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Brian Bailey
 * Copyright (c) 2025 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Fire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/**
 * @file Smoke.cpp
 * @brief This class specifies smoke sources for QES-Fire and QES-Plume integration
 */

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
