/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/**
 * @file TURBWallTerrain.cpp
 * @brief :document this:
 */

#include "TURBWallTerrain.h"

void TURBWallTerrain::defineWalls(WINDSGeneralData *WGD,TURBGeneralData *TGD) {
    // fill array with cellid  of cutcell cells
    get_cutcell_wall_id(WGD,icellflag_cutcell);
    // fill itrublfag with cutcell flag
    set_cutcell_wall_flag(TGD,iturbflag_cutcell);

    // [FM] temporary fix -> use stairstep within the cut-cell
    get_stairstep_wall_id(WGD,icellflag_terrain);
    set_stairstep_wall_flag(TGD,iturbflag_stairstep);

    /*
      [FM] temporary fix -> when cut-cell treatement is implmented
      if(cutcell_wall_id.size()==0) {
      get_stairstep_wall_id(WGD,icellflag_terrain);
      set_stairstep_wall_flag(TGD,iturbflag_stairstep);
      } else {
      use_cutcell = true;
      }
    */

    return;
}


void TURBWallTerrain::setWallsBC(WINDSGeneralData *WGD,TURBGeneralData *TGD){

    /*
      This function apply the loglow at the wall for terrain
      Note:
      - only stair-step is implemented
      - need to do: cut-cell for terrain
    */
    int nx = WGD->nx;
    int ny = WGD->ny;

    if(!use_cutcell) {
        float z0=0.01;
        for (size_t id=0; id < stairstep_wall_id.size(); ++id){

            int id_cc=stairstep_wall_id[id];
            int k = (int)(id_cc / ((nx-1)*(ny-1)));
            int j = (int)((id_cc - k*(nx-1)*(ny-1))/(nx-1));
            int i = id_cc -  j*(nx-1) - k*(nx-1)*(ny-1);
            int id_2d = i + j*nx;

            // set_loglaw_stairstep_at_id_cc need z0 at the cell center
            // -> z0 is averaged over the 4 face of the cell
            z0=0.25*(WGD->z0_domain_u.at(id_2d)+WGD->z0_domain_u.at(id_2d+1)+
                     WGD->z0_domain_v.at(id_2d)+WGD->z0_domain_v.at(id_2d+nx));

            set_loglaw_stairstep_at_id_cc(WGD,TGD,id_cc,icellflag_terrain,z0);
        }
    } else {
        // [FM] temporary fix because the cut-cell are messing with the wall
        // at the terrain
        for(size_t i=0; i < cutcell_wall_id.size(); i++) {
            int id_cc=cutcell_wall_id[i];
            TGD->Sxx[id_cc]=0.0;
            TGD->Sxy[id_cc]=0.0;
            TGD->Sxz[id_cc]=0.0;
            TGD->Syy[id_cc]=0.0;
            TGD->Syz[id_cc]=0.0;
            TGD->Szz[id_cc]=0.0;
            TGD->Lm[id_cc]=0.0;
        }
    }
}
