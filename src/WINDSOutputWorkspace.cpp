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
 * @file WINDSOutputWorkspace.cpp
 * @brief Specialized output classes derived from QESNetCDFOutput for
 * face center data (used for turbulence,...)
 */

#include "WINDSOutputWorkspace.h"

WINDSOutputWorkspace::WINDSOutputWorkspace(WINDSGeneralData *WGD,std::string output_file)
    : QESNetCDFOutput(output_file)
{
    std::cout<<"[Output] \t Setting fields of workspace file"<<std::endl;

    // set list of fields to save, no option available for this file
    output_fields = {"t","times","x_cc","y_cc","z_cc","z_face","dz_array",
                     "u","v","w","icellflag",
                     "terrain","z0_u","z0_v",
                     "e","f","g","h","m","n","building_volume_frac","terrain_volume_frac",
                     "mixlength"};

    // copy of WGD pointer
    WGD_=WGD;

    // domain size information:
    int nx = WGD_->nx;
    int ny = WGD_->ny;
    int nz = WGD_->nz;

    // Location of face centers in z-dir
    z_cc.resize( nz-1 );
    dz_array.resize( nz-1, 0.0 );
    z_face.resize( nz-1 );
    for (auto k=0; k<nz-1; k++) {
        z_cc[k] = WGD_->z[k];
        dz_array[k] = WGD_->dz_array[k];
        z_face[k] = WGD_->z_face[k];
    }

    // Location of face centers in x-dir
    x_cc.resize( nx-1 );
    for (auto i=0; i<nx-1; i++) {
        x_cc[i] = (i+0.5)*WGD_->dx;
    }
    // Location of face centers in y-dir
    y_cc.resize( ny-1 );
    for (auto j=0; j<ny-1; j++) {
        y_cc[j] = (j+0.5)*WGD_->dy;
    }

    timestamp.resize( dateStrLen, '0' );

    // time dimension
    NcDim NcDim_t=addDimension("t");
    NcDim NcDim_tstr=addDimension("dateStrLen",dateStrLen);
    
    // create attributes for time dimension
    std::vector<NcDim> dim_vect_t;
    dim_vect_t.push_back(NcDim_t);
    createAttScalar("t","time","s",dim_vect_t,&time);
    
    // create attributes for time dimension
    std::vector<NcDim> dim_vect_tstr;
    dim_vect_tstr.push_back(NcDim_t);
    dim_vect_tstr.push_back(NcDim_tstr);
    createAttVector("times","date time","-",dim_vect_tstr,&timestamp);
    
    // set face-centered data dimensions
    // space dimensions
    NcDim NcDim_x_fc=addDimension("x",WGD_->nx);
    NcDim NcDim_y_fc=addDimension("y",WGD_->ny);
    NcDim NcDim_z_fc=addDimension("z",WGD_->nz);

    // 3D vector dimension (time dep)
    std::vector<NcDim> dim_vect_fc;
    dim_vect_fc.push_back(NcDim_t);
    dim_vect_fc.push_back(NcDim_z_fc);
    dim_vect_fc.push_back(NcDim_y_fc);
    dim_vect_fc.push_back(NcDim_x_fc);
    // create attributes
    createAttVector("u","x-component velocity","m s-1",dim_vect_fc,&(WGD_->u));
    createAttVector("v","y-component velocity","m s-1",dim_vect_fc,&(WGD_->v));
    createAttVector("w","z-component velocity","m s-1",dim_vect_fc,&(WGD_->w));

    // set cell-centered data dimensions
    // space dimensions
    NcDim NcDim_x_cc=addDimension("x_cc",WGD_->nx-1);
    NcDim NcDim_y_cc=addDimension("y_cc",WGD_->ny-1);
    NcDim NcDim_z_cc=addDimension("z_cc",WGD_->nz-1);

    // create attributes space dimensions
    std::vector<NcDim> dim_vect_x_cc;
    dim_vect_x_cc.push_back(NcDim_x_cc);
    createAttVector("x_cc","x-distance","m",dim_vect_x_cc,&x_cc);
    std::vector<NcDim> dim_vect_y_cc;
    dim_vect_y_cc.push_back(NcDim_y_cc);
    createAttVector("y_cc","y-distance","m",dim_vect_y_cc,&y_cc);
    std::vector<NcDim> dim_vect_z_cc;
    dim_vect_z_cc.push_back(NcDim_z_cc);
    createAttVector("z_cc","z-distance","m",dim_vect_z_cc,&z_cc);
    createAttVector("z_face","z location of the faces","m",dim_vect_z_cc,&z_face);
    createAttVector("dz_array","dz size of the cells","m",dim_vect_z_cc,&dz_array);

    // create 2D vector (surface, indep of time)
    std::vector<NcDim> dim_vect_2d;
    dim_vect_2d.push_back(NcDim_y_cc);
    dim_vect_2d.push_back(NcDim_x_cc);
    // create attributes
    createAttVector("terrain","terrain height","m",dim_vect_2d,&(WGD_->terrain));
    createAttVector("z0_u","terrain areo roughness, u","m",dim_vect_2d,&(WGD_->z0_domain_u));
    createAttVector("z0_v","terrain areo roughness, v","m",dim_vect_2d,&(WGD_->z0_domain_v));

    createAttVector("mixlength","distance to nearest object","m",{NcDim_z_cc,NcDim_y_cc,NcDim_x_cc},&(WGD_->mixingLengths));

    // 3D vector dimension (time dep)
    std::vector<NcDim> dim_vect_cc;
    dim_vect_cc.push_back(NcDim_t);
    dim_vect_cc.push_back(NcDim_z_cc);
    dim_vect_cc.push_back(NcDim_y_cc);
    dim_vect_cc.push_back(NcDim_x_cc);

    // create attributes
    createAttVector("icellflag","icell flag value","--",dim_vect_cc,&(WGD_->icellflag));

    // attributes for coefficients for SOR solver
    createAttVector("e","e cut-cell coefficient","--",dim_vect_cc,&(WGD_->e));
    createAttVector("f","f cut-cell coefficient","--",dim_vect_cc,&(WGD_->f));
    createAttVector("g","g cut-cell coefficient","--",dim_vect_cc,&(WGD_->g));
    createAttVector("h","h cut-cell coefficient","--",dim_vect_cc,&(WGD_->h));
    createAttVector("m","m cut-cell coefficient","--",dim_vect_cc,&(WGD_->m));
    createAttVector("n","n cut-cell coefficient","--",dim_vect_cc,&(WGD_->n));

    // attribute for the volume fraction (cut-cell)
    createAttVector("building_volume_frac","building volume fraction","--",dim_vect_cc,&(WGD_->building_volume_frac));
    createAttVector("terrain_volume_frac","terrain volume fraction","--",dim_vect_cc,&(WGD_->terrain_volume_frac));

    // create output fields
    addOutputFields();
}


// Save output at cell-centered values
void WINDSOutputWorkspace::save(ptime timeOut)
{

    // set time
    time = (double)output_counter;
    
    std::string s=to_iso_extended_string(timeOut);
    std::copy(s.begin(), s.end(), timestamp.begin());
   
    // save fields
    saveOutputFields();

    // remmove time indep from output array after first save
    if (output_counter==0) {
        rmTimeIndepFields();
    }

    // increment for next time insertion
    output_counter +=1;

};



// [FM] Feb.28.2020 OBSOLETE
void WINDSOutputWorkspace::setBuildingFields(NcDim* NcDim_t,NcDim* NcDim_building)
{
    int nBuildings=WGD_->allBuildingsV.size();

    building_rotation.resize(nBuildings,0.0);
    canopy_rotation.resize(nBuildings,0.0);

    L.resize(nBuildings,0.0);
    W.resize(nBuildings,0.0);
    H.resize(nBuildings,0.0);

    length_eff.resize(nBuildings,0.0);
    width_eff.resize(nBuildings,0.0);
    height_eff.resize(nBuildings,0.0);
    base_height.resize(nBuildings,0.0);

    building_cent_x.resize(nBuildings,0.0);
    building_cent_y.resize(nBuildings,0.0);

    i_start.resize(nBuildings,0);
    i_end.resize(nBuildings,0);
    j_start.resize(nBuildings,0);
    j_end.resize(nBuildings,0);
    k_end.resize(nBuildings,0);

    i_cut_start.resize(nBuildings,0);
    i_cut_end.resize(nBuildings,0);
    j_cut_start.resize(nBuildings,0);
    j_cut_end.resize(nBuildings,0);
    k_cut_end.resize(nBuildings,0);

    i_building_cent.resize(nBuildings,0);
    j_building_cent.resize(nBuildings,0);

    upwind_dir.resize(nBuildings,0.0);
    Lr.resize(nBuildings,0.0);

    // vector of dimension for building information
    std::vector<NcDim> dim_vect_building;
    dim_vect_building.push_back(*NcDim_building);

    // create attributes
    createAttVector("building_rotation","rotation of building","rad",dim_vect_building,&building_rotation);
    createAttVector("canopy_rotation","rotation of canopy","rad",dim_vect_building,&building_rotation);

    createAttVector("L","length of building","m",dim_vect_building,&L);
    createAttVector("W","width of building","m",dim_vect_building,&L);
    createAttVector("H","height of building","m",dim_vect_building,&H);

    createAttVector("height_eff","effective height","m",dim_vect_building,&height_eff);
    createAttVector("base_height","base height","m",dim_vect_building,&base_height);

    createAttVector("building_cent_x","x-coordinate of centroid","m",dim_vect_building,&building_cent_x);
    createAttVector("building_cent_y","y-coordinate of centroid","m",dim_vect_building,&building_cent_y);

    createAttVector("i_start","x-index start","--",dim_vect_building,&i_start);
    createAttVector("i_end","x-index end","--",dim_vect_building,&i_end);
    createAttVector("j_start","y-index start","--",dim_vect_building,&j_start);
    createAttVector("j_end","y-index end","--",dim_vect_building,&j_end);
    createAttVector("k_start","z-index end","--",dim_vect_building,&k_end);

    createAttVector("i_cut_start","x-index start cut-cell","--",dim_vect_building,&i_cut_start);
    createAttVector("i_cut_end","x-index end cut-cell","--",dim_vect_building,&i_cut_end);
    createAttVector("j_cut_start","y-index start cut-cell","--",dim_vect_building,&j_cut_start);
    createAttVector("j_cut_end","y-index end cut-cell","--",dim_vect_building,&j_cut_end);
    createAttVector("k_cut_start","z-index end cut-cell","--",dim_vect_building,&k_cut_end);

    createAttVector("i_building_cent","x-index of centroid","--",dim_vect_building,&i_building_cent);
    createAttVector("i_building_cent","y-index of centroid","--",dim_vect_building,&i_building_cent);

    // temporary vector to add the fields into output_fields for output.
    std::vector<string> tmp_fields;
    tmp_fields.clear();  // clear the vector
    tmp_fields={"building_rotation","canopy_rotation","L","W","H","height_eff","base_height",
                "building_cent_x","building_cent_y",
                "i_start","i_end","j_start","j_end","k_start",
                "i_cut_start","i_cut_end","j_cut_start","j_cut_end","k_cut_start",
                "i_building_cent","j_building_cent"};
    output_fields.insert(output_fields.end(),tmp_fields.begin(),tmp_fields.end());

    // vector of dimension for time dep building information
    std::vector<NcDim> dim_vect_building_t;
    dim_vect_building_t.push_back(*NcDim_t);
    dim_vect_building_t.push_back(*NcDim_building);

    // create attributes
    createAttVector("length_eff","effective length","m",dim_vect_building_t,&length_eff);
    createAttVector("width_eff","effective width","m",dim_vect_building_t,&width_eff);

    createAttVector("upwind_dir","upwind wind direction","rad",dim_vect_building_t,&upwind_dir);
    createAttVector("Lr","Length of far wake zone","m",dim_vect_building_t,&Lr);

    // temporary vector to add the fields into output_fields for output.
    tmp_fields.clear();  // clear the vector
    tmp_fields={"length_eff","width_eff","upwind_dir","Lr"};
    output_fields.insert(output_fields.end(),tmp_fields.begin(),tmp_fields.end());

    return;
}

// [FM] Feb.28.2020 OBSOLETE
void WINDSOutputWorkspace::getBuildingFields()
{
    int nBuildings=WGD_->allBuildingsV.size();

    // information only needed once (at output_counter==0)
    if (output_counter==0) {
        // copy time independent fields
        for(int id=0;id<nBuildings;++id) {
            building_rotation[id]=WGD_->allBuildingsV[id]->building_rotation;
            canopy_rotation[id]=WGD_->allBuildingsV[id]->canopy_rotation;

            L[id]=WGD_->allBuildingsV[id]->L;
            W[id]=WGD_->allBuildingsV[id]->W;
            H[id]=WGD_->allBuildingsV[id]->H;

            height_eff[id]=WGD_->allBuildingsV[id]->height_eff;
            base_height[id]=WGD_->allBuildingsV[id]->base_height;

            building_cent_x[id]=WGD_->allBuildingsV[id]->building_cent_x;
            building_cent_y[id]=WGD_->allBuildingsV[id]->building_cent_y;

            i_start[id]=WGD_->allBuildingsV[id]->i_start;
            i_end[id]=WGD_->allBuildingsV[id]->i_end;
            j_start[id]=WGD_->allBuildingsV[id]->j_start;
            j_end[id]=WGD_->allBuildingsV[id]->j_end;
            k_end[id]=WGD_->allBuildingsV[id]->k_end;

            i_cut_start[id]=WGD_->allBuildingsV[id]->i_cut_start;
            i_cut_end[id]=WGD_->allBuildingsV[id]->i_cut_end;
            j_cut_start[id]=WGD_->allBuildingsV[id]->j_cut_start;
            j_cut_end[id]=WGD_->allBuildingsV[id]->j_cut_end;
            k_cut_end[id]=WGD_->allBuildingsV[id]->k_cut_end;

            i_building_cent[id]=WGD_->allBuildingsV[id]->i_building_cent;
            j_building_cent[id]=WGD_->allBuildingsV[id]->j_building_cent;
        }
    }

    // copy time dependent fields
    for(int id=0;id<nBuildings;++id) {
        length_eff[id]=WGD_->allBuildingsV[id]->length_eff;
        width_eff[id]=WGD_->allBuildingsV[id]->width_eff;

        upwind_dir[id]=WGD_->allBuildingsV[id]->upwind_dir;
        Lr[id]=WGD_->allBuildingsV[id]->Lr;
    }

    return;
}
