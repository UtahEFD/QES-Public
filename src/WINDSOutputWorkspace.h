/*
 * QES-Winds
 *
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
 *
 */


#pragma once

#include <string>

#include "WINDSGeneralData.h"
#include "QESNetCDFOutput.h"

/* Specialized output classes derived from QESNetCDFOutput for
   face center data (used for turbulence,...)
*/
class WINDSOutputWorkspace : public QESNetCDFOutput
{
public:
    WINDSOutputWorkspace()
        : QESNetCDFOutput()
    {}

    WINDSOutputWorkspace(WINDSGeneralData*,std::string);
    ~WINDSOutputWorkspace()
    {}

    //save function be call outside
    void save(float);

private:

    std::vector<float> x_cc,y_cc,z_cc,z_face,dz_array;

    WINDSGeneralData* WGD_;

    // [FM] Feb.28.2020 OBSOLETE
    // Building data functions:
    void setBuildingFields(NcDim*,NcDim*);
    void getBuildingFields();

    // [FM] Feb.28.2020 OBSOLETE
    // Buidling data variables
    bool buildingFieldsSet = false;

    // [FM] Feb.28.2020 OBSOLETE
    // These variables are used to convert data structure in array so it can be stored in
    // NetCDF file. (Canopy can be building, need to specify)
    // size of these vector = number of buidlings
    std::vector<float> building_rotation,canopy_rotation;

    std::vector<float> L,W,H;
    std::vector<float> length_eff,width_eff,height_eff,base_height;
    std::vector<float> building_cent_x, building_cent_y;

    std::vector<int> i_start, i_end, j_start, j_end, k_end,k_start;
    std::vector<int> i_cut_start, i_cut_end, j_cut_start, j_cut_end, k_cut_end;
    std::vector<int> i_building_cent, j_building_cent;

    std::vector<float> upwind_dir,Lr;

};
