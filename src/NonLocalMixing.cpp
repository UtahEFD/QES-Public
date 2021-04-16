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
 * @file NonLocalMixing.cpp
 * @brief :document this:
 * @sa LocalMixing
 */

#include "PolyBuilding.h"

// These take care of the circular reference
#include "WINDSInputData.h"
#include "WINDSGeneralData.h"
#include "TURBGeneralData.h"

void PolyBuilding::NonLocalMixing (WINDSGeneralData* WGD, TURBGeneralData* TGD,int building_id)
{

    int nx = WGD->nx;
    int ny = WGD->ny;
    int nz = WGD->nz;

    const float sigUOrg   = 2.0;
    const float sigVOrg   = 2.0;
    const float sigWOrg   = 1.3;
    const float sigUConst = sigUOrg;
    const float sigVConst = sigVOrg;
    const float sigWConst = sigWOrg;

    const float kvonk = 0.4;

    float U_ref_V; //reference veocity veritcal (top of building)
    float U_ref_r,U_ref_l,dU_ref_H; //reference velocities horizontal (left and right)
    float U_a; //axis velocity
    float u_h,v_h; //interpolated velocites

    // x,y positons of the polybuilding verteces in the rotated coord syst rotated with the wind dir and building center
    //float xp,yp; //working variables

    float ustar=0,ustarH=0,ustarV=0; //fiction velocities

    //float z_build;                  // z value of each building point from its base height
    //int k_bottom, k_top;
    //int kk;

    int icell_face,icell_cent;
    int icell_face_cl,icell_cent_cl;

    std::vector<float> Lr_face, Lr_node;
    std::vector<int> perpendicular_flag;
    Lr_face.resize (polygonVertices.size(), -1.0);       // Length of wake for each face
    Lr_node.resize (polygonVertices.size(), 0.0);       // Length of wake for each node
    perpendicular_flag.resize (polygonVertices.size(), 0);
    upwind_rel_dir.resize (polygonVertices.size(), 0.0);      // Upwind reletive direction for each face
    float z_build;                  // z value of each building point from its base height
    float yc, xc;

    float Lr_local, Lr_local_u, Lr_local_v, Lr_local_w;   // Local length of the wake for each velocity component
    float x_wall = 0.0, x_wall_u = 0.0, x_wall_v = 0.0, x_wall_w = 0.0;

    float y_norm, canyon_factor;
    int x_id_min;

    float Lr_ave;                         // Average length of Lr
    float total_seg_length;               // Length of each edge
    int index_previous, index_next;       // Indices of previous and next nodes
    int stop_id = 0;
    int kk = k_start;
    float tol;
    float farwake_factor = 3;
    float epsilon = 10e-10;
    int u_wake_flag, v_wake_flag, w_wake_flag;

    int i_cl, j_cl;
    float xp, yp;

    int i_u, j_u, i_v, j_v, i_w, j_w;          // i and j indices for x, y and z directions
    float xu, yu, xv, yv, xw, yw;
    float dn_u, dn_v, dn_w;             // Length of cavity zone

    int k_bottom = 1, k_top = 1;
    int count_cl_outofbound = 0;

    if ( xi.size() == 0 ) {
        // exit if building has no area (computed in polygonWake.cpp)
        std::cout<< "[WARNING] building ill-defined (no area) -> use local mixing (building id="<< building_id << ")" <<std::endl;
        return;
    }
    if (k_start < 1 || k_start > nz-2 || k_end < 1 || k_end > nz-2 ) {
        // exit if building start/end point is ill-defined (computed in polygonWake.cpp)
        std::cout<< "[WARNING] building ill-defined (k out of bound) -> use local mixing (building id="<< building_id << ")" <<std::endl;
        return;
    }
    /*
    if ( height_eff >= WGD->z[nz-1] ) {
        std::cout << "domain = "<< WGD->z[0] << " " << WGD->z[nz-2] << " " << WGD->z[nz-1] << std::endl;
        std::cout << "buidling = "<< base_height << " " << height_eff << std::endl;
        // exit if building above top domain is ill-defined (computed in polygonWake.cpp)
        std::cout<< "[WARNING] building ill-defined -> use local mixing (building id="<< building_id << ")" <<std::endl;
        return;
        }
    */


    tol = 0.01*M_PI/180.0;
    // Calculating length of the downwind wake based on Fackrell (1984) formulation
    Lr = 1.8*height_eff*W_over_H/(pow(L_over_H,0.3)*(1+0.24*W_over_H));

    for (size_t id=0; id<polygonVertices.size()-1; id++) {
        // Finding faces that are eligible for applying the far-wake parameterizations
        // angle between two points should be in -180 to 0 degree
        if ( abs(upwind_rel_dir[id]) < 0.5*M_PI) {
            // Calculate length of the far wake zone for each face
            Lr_face[id] = Lr*cos(upwind_rel_dir[id]);
        }
    }

    Lr_ave = total_seg_length = 0.0;
    // This loop interpolates the value of Lr for eligible faces to nodes of those faces
    for (size_t id=0; id<polygonVertices.size()-1; id++) {
        // If the face is eligible for parameterization
        if (Lr_face[id] > 0.0) {
            index_previous = (id+polygonVertices.size()-2)%(polygonVertices.size()-1);     // Index of previous face
            index_next = (id+1)%(polygonVertices.size()-1);           // Index of next face
            if (Lr_face[index_previous] < 0.0 && Lr_face[index_next] < 0.0) {
                Lr_node[id] = Lr_face[id];
                Lr_node[id+1] = Lr_face[id];
            } else if (Lr_face[index_previous] < 0.0) {
                Lr_node[id] = Lr_face[id];
                Lr_node[id+1] = ((yi[index_next]-yi[index_next+1])*Lr_face[index_next]+(yi[id]-yi[index_next])*Lr_face[id])/(yi[id]-yi[index_next+1]);
            } else if (Lr_face[index_next] < 0.0) {
                Lr_node[id] = ((yi[id]-yi[index_next])*Lr_face[id]+(yi[index_previous]-yi[id])*Lr_face[index_previous])/(yi[index_previous]-yi[index_next]);
                Lr_node[id+1] = Lr_face[id];
            } else {
                Lr_node[id] = ((yi[id]-yi[index_next])*Lr_face[id]+(yi[index_previous]-yi[id])*Lr_face[index_previous])/(yi[index_previous]-yi[index_next]);
                Lr_node[id+1] = ((yi[index_next]-yi[index_next+1])*Lr_face[index_next]+(yi[id]-yi[index_next])*Lr_face[id])/(yi[id]-yi[index_next+1]);
            }
            Lr_ave += Lr_face[id]*(yi[id]-yi[index_next]);
            total_seg_length += (yi[id]-yi[index_next]);
        }

        if ((polygonVertices[id+1].x_poly > polygonVertices[0].x_poly-0.1) && (polygonVertices[id+1].x_poly < polygonVertices[0].x_poly+0.1)
            && (polygonVertices[id+1].y_poly > polygonVertices[0].y_poly-0.1) && (polygonVertices[id+1].y_poly < polygonVertices[0].y_poly+0.1))
        {
            stop_id = id;
            break;
        }
    }

    Lr = Lr_ave/total_seg_length;
    for (auto k = 1; k <= k_start; k++) {
        k_bottom = k;
        if (base_height <= WGD->z.at(k)) {
            break;
        }
    }

    for (auto k = k_start; k < WGD->nz-2; k++) {
        k_top = k;
        if (height_eff < WGD->z.at(k+1)) {
            break;
        }
    }
    for (auto k = k_start; k < k_end; k++) {
        kk = k;
        if (0.75*H+base_height <= WGD->z.at(k)) {
            break;
        }
    }

    // interpolation of velocity at the top of the building
    icell_face=i_building_cent + j_building_cent*nx + (k_top+1)*nx*ny;
    u_h=0.5*(WGD->u[icell_face]+WGD->u[icell_face+1]);
    v_h=0.5*(WGD->v[icell_face]+WGD->v[icell_face+nx]);
    //w_h=0.5*(WGD->w[icell_face]+WGD->w[icell_face+nx*ny]);

    // verical velocity reference and verical fiction velocity
    //U_ref_V=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);
    U_ref_V=u_h*cos(upwind_dir)+v_h*sin(upwind_dir);
    ustarV=kvonk*U_ref_V;

    // scale factor = scale the dxy as a function of the angle of the flow
    float scale_factor=1;
    /*
      if( 1.0/cos(upwind_dir) <= sqrt(2) ) {
      scale_factor=1.0/cos(upwind_dir);
      } else {
      scale_factor=1.0/sin(upwind_dir);
      }
    */
    float dxy=scale_factor*WGD->dxy;

    // x,y positons of the polybuilding verteces in the rotated coord syst rotated with the wind dir and building center
    std::vector<float> xp_i, yp_i;
    xp_i.resize (polygonVertices.size(), 0.0);
    yp_i.resize (polygonVertices.size(), 0.0);

    for (size_t id=0; id<polygonVertices.size()-1; id++)
    {
        // Finding projection of polygon vertex on rotated coord syst rotated with the wind dir and building center
        // x,y
        xp_i[id] = cos(upwind_dir)*(polygonVertices[id].x_poly-building_cent_x) + sin(upwind_dir)*(polygonVertices[id].y_poly - building_cent_y);
        yp_i[id] =-sin(upwind_dir)*(polygonVertices[id].x_poly-building_cent_x) + cos(upwind_dir)*(polygonVertices[id].y_poly - building_cent_y);
    }
    // find min and max in the yp direction
    int minIndex = std::min_element(yp_i.begin(),yp_i.end()) - yp_i.begin();
    int maxIndex = std::max_element(yp_i.begin(),yp_i.end()) - yp_i.begin();

    // min (right) and max (left) x-postion of the outer point of the polybuilding
    // if xp < 0 -> add small amount to ensure point is inside
    // if xp > 0 -> substract small amount ot ensure point is inside
    float xp_l, xp_r;
    xp_r=xp_i[minIndex];
    if(xp_r < 0.0) {
        xp_r += 0.5*dxy;
    } else {
        xp_r -= 0.5*dxy;
    }
    xp_l=xp_i[maxIndex];
    if(xp_l < 0.0) {
        xp_l += 0.5*dxy;
    } else {
        xp_l -= 0.5*dxy;
    }

    // min (right) and max (left) y-postion of the outer point of the polybuilding
    float yp_l, yp_r;
    yp_r=yp_i[minIndex]+0.5*dxy;
    yp_l=yp_i[maxIndex]-0.5*dxy;

    /* FM (debug notes) this
     * This section is not used in the final version following Willams et 2004
     // x-coord of maximun downstream vertex
     float xp_d;
     maxIndex = std::max_element(xp_i.begin(),xp_i.end()) - xp_i.begin();
     xp_d=xp_i[maxIndex]-0.5*dxy;

     // set itrubflag for debug
     float x_r, y_r;
     x_r = cos(upwind_dir)*xp_r - sin(upwind_dir)*yp_r + building_cent_x;
     y_r = sin(upwind_dir)*xp_r + cos(upwind_dir)*yp_r + building_cent_y;

     int i_r = floor(x_r/WGD->dx);
     int j_r = floor(y_r/WGD->dy);

     for (auto k=k_top+1; k>=k_bottom; k--) {
     icell_cent = i_r + j_r*(nx-1) + k*(ny-1)*(nx-1);
     TGD->iturbflag[icell_cent]=12;
     }

     float x_l, y_l;
     x_l = cos(upwind_dir)*xp_l - sin(upwind_dir)*yp_l + building_cent_x;
     y_l = sin(upwind_dir)*xp_l + cos(upwind_dir)*yp_l + building_cent_y;

     int i_l = floor(x_l/WGD->dx);
     int j_l = floor(y_l/WGD->dy);

     for (auto k=k_top+1; k>=k_bottom; k--) {
     icell_cent = i_l + j_l*(nx-1) + k*(ny-1)*(nx-1);
     TGD->iturbflag[icell_cent]=12;
     }
    */

    ///////
    float xp_ref_r, yp_ref_r;
    float x_ref_r, y_ref_r;
    xp_ref_r=xp_r;
    yp_ref_r=yp_r-3.0*dxy;

    x_ref_r = cos(upwind_dir)*xp_ref_r - sin(upwind_dir)*yp_ref_r + building_cent_x;
    y_ref_r = sin(upwind_dir)*xp_ref_r + cos(upwind_dir)*yp_ref_r + building_cent_y;

    int i_ref_r = floor(x_ref_r/WGD->dx);
    int j_ref_r = floor(y_ref_r/WGD->dy);

    if ( i_ref_r >= WGD->nx-2 && i_ref_r <= 0 && j_ref_r >= WGD->ny-2 && j_ref_r <= 0) {
        std::cout<< "[WARNING] right ref point outside domain -> use local mixing (building id="<< building_id << ")" <<std::endl;
        return;
    }
    for (auto k=k_top+1; k>=k_bottom; k--) {
        icell_cent = i_ref_r + j_ref_r*(nx-1) + k*(ny-1)*(nx-1);
        //TGD->iturbflag[icell_cent]=12;
    }

    ///////
    float xp_ref_l, yp_ref_l;
    float x_ref_l, y_ref_l;
    xp_ref_l=xp_l;
    yp_ref_l=yp_l+3.0*dxy;

    x_ref_l = cos(upwind_dir)*xp_ref_l - sin(upwind_dir)*yp_ref_l + building_cent_x;
    y_ref_l = sin(upwind_dir)*xp_ref_l + cos(upwind_dir)*yp_ref_l + building_cent_y;

    int i_ref_l = floor(x_ref_l/WGD->dx);
    int j_ref_l = floor(y_ref_l/WGD->dy);

    if ( i_ref_l >= WGD->nx-2 && i_ref_l <= 0 && j_ref_l >= WGD->ny-2 && j_ref_l <= 0) {
        std::cout<< "[WARNING] left ref point outside domain -> use local mixing (building id="<< building_id << ")" <<std::endl;
        return;
    }
    for (auto k=k_top+1; k>=k_bottom; k--) {
        icell_cent = i_ref_l + j_ref_l*(nx-1) + k*(ny-1)*(nx-1);
        //TGD->iturbflag[icell_cent]=12;
    }


    //std::cout << "buidling id" << building_id << " k_start=" << k_start << " k_end=" << k_end << " kk=" << kk << std::endl;

    for (auto k=k_top; k>=k_bottom; k--)
    {
        z_build = WGD->z[k] - base_height;

        //reference velocity left
        icell_face = i_ref_l + j_ref_l*nx + k*ny*nx;
        u_h=0.5*(WGD->u[icell_face]+WGD->u[icell_face+1]);
        v_h=0.5*(WGD->v[icell_face]+WGD->v[icell_face+nx]);
        //w_h=0.5*(WGD->w[icell_face]+WGD->w[icell_face+nx*ny]);
        U_ref_l=u_h*cos(upwind_dir)+v_h*sin(upwind_dir);
        //U_ref_l=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);

        //reference velocity right
        icell_face = i_ref_r + j_ref_r*nx + k*ny*nx;
        u_h=0.5*(WGD->u[icell_face]+WGD->u[icell_face+1]);
        v_h=0.5*(WGD->v[icell_face]+WGD->v[icell_face+nx]);
        //w_h=0.5*(WGD->w[icell_face]+WGD->w[icell_face+nx*ny]);
        U_ref_r=u_h*cos(upwind_dir)+v_h*sin(upwind_dir);
        //U_ref_r=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);

        for (auto id=0; id<=stop_id; id++)
        {
            if (abs(upwind_rel_dir[id]) < 0.5*M_PI)
            {
                if (abs(upwind_rel_dir[id]) < tol)
                {
                    perpendicular_flag[id]= 1;
                    x_wall = xi[id];
                }
                for (auto y_id=0; y_id <= 2*ceil(abs(yi[id]-yi[id+1])/WGD->dxy); y_id++)
                {
                    yc = yi[id]-0.5*y_id*WGD->dxy;
                    Lr_local = Lr_node[id]+(yc-yi[id])*(Lr_node[id+1]-Lr_node[id])/(yi[id+1]-yi[id]);
                    // Checking to see whether the face is perpendicular to the wind direction
                    if(perpendicular_flag[id] == 0)
                    {
                        x_wall = ((xi[id+1]-xi[id])/(yi[id+1]-yi[id]))*(yc-yi[id])+xi[id];
                    }
                    if (yc >= 0.0)
                    {
                        y_norm = y2;
                    }
                    else
                    {
                        y_norm = y1;
                    }
                    canyon_factor = 1.0;
                    x_id_min = -1;
                    for (auto x_id=1; x_id <= ceil(Lr_local/WGD->dxy); x_id++)
                    {
                        xc = x_id*WGD->dxy;
                        int i = ((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/WGD->dx;
                        int j = ((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/WGD->dy;
                        if ( i >= WGD->nx-2 && i <= 0 && j >= WGD->ny-2 && j <= 0)
                        {
                            break;
                        }
                        int icell_cent = i+j*(WGD->nx-1)+kk*(WGD->nx-1)*(WGD->ny-1);
                        if ( WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2 && x_id_min < 0)
                        {
                            x_id_min = x_id;
                        }
                        if ( (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2) && x_id_min > 0)
                        {
                            canyon_factor = xc/Lr;
                            break;
                        }
                    }
                    x_id_min = -1;
                    for (auto x_id=1; x_id <= 2*ceil(farwake_factor*Lr_local/WGD->dxy); x_id++)
                    {
                        u_wake_flag = 1;
                        v_wake_flag = 1;
                        w_wake_flag = 1;

                        xc = 0.5*x_id*WGD->dxy;
                        int i = ((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/WGD->dx;
                        int j = ((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/WGD->dy;
                        if (i >= WGD->nx-2 || i <= 0 || j >= WGD->ny-2 || j <= 0)
                        {
                            break;
                        }
                        icell_cent = i+j*(WGD->nx-1)+k*(WGD->nx-1)*(WGD->ny-1);
                        if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2 && x_id_min < 0)
                        {
                            x_id_min = x_id;
                        }
                        if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2)
                        {
                            if (x_id_min >= 0)
                            {
                                if (WGD->ibuilding_flag[icell_cent] == building_id)
                                {
                                    x_id_min = -1;
                                }
                                else if (canyon_factor < 1.0)
                                {
                                    break;
                                }
                                else if (WGD->icellflag[i+j*(WGD->nx-1)+kk*(WGD->nx-1)*(WGD->ny-1)] == 0 ||
                                         WGD->icellflag[i+j*(WGD->nx-1)+kk*(WGD->nx-1)*(WGD->ny-1)] == 2)
                                {
                                    break;
                                }

                            }
                        }

                        if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2)
                        {

                            i_u = std::round(((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/WGD->dx);
                            j_u = ((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/WGD->dy;
                            if (i_u < WGD->nx-1 && i_u > 0 && j_u < WGD->ny-1 && j_u > 0)
                            {
                                xp = i_u*WGD->dx-building_cent_x;
                                yp = (j_u+0.5)*WGD->dy-building_cent_y;
                                xu = xp*cos(upwind_dir)+yp*sin(upwind_dir);
                                yu = -xp*sin(upwind_dir)+yp*cos(upwind_dir);
                                Lr_local_u = Lr_node[id]+(yu-yi[id])*(Lr_node[id+1]-Lr_node[id])/(yi[id+1]-yi[id]);
                                if (perpendicular_flag[id] > 0)
                                {
                                    x_wall_u = xi[id];

                                }
                                else
                                {
                                    x_wall_u = ((xi[id+1]-xi[id])/(yi[id+1]-yi[id]))*(yu-yi[id])+ xi[id];
                                }

                                xu -= x_wall_u;
                                if (abs(yu) < abs(y_norm) && abs(y_norm) > epsilon && z_build < height_eff && height_eff > epsilon)
                                {
                                    dn_u = sqrt((1.0-pow((yu/y_norm), 2.0))*(1.0-pow((z_build/height_eff),2.0))*pow((canyon_factor*Lr_local_u),2.0));
                                }
                                else
                                {
                                    dn_u = 0.0;
                                }
                                if (xu > farwake_factor*dn_u)
                                {
                                    u_wake_flag = 0;
                                }
                            }

                            i_v = ((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/WGD->dx;
                            j_v = std::round(((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/WGD->dy);
                            if (i_v<WGD->nx-1 && i_v>0 && j_v<WGD->ny-1 && j_v>0)
                            {
                                xp = (i_v+0.5)*WGD->dx-building_cent_x;
                                yp = j_v*WGD->dy-building_cent_y;
                                xv = xp*cos(upwind_dir)+yp*sin(upwind_dir);
                                yv = -xp*sin(upwind_dir)+yp*cos(upwind_dir);
                                Lr_local_v = Lr_node[id]+(yv-yi[id])*(Lr_node[id+1]-Lr_node[id])/(yi[id+1]-yi[id]);
                                if (perpendicular_flag[id] > 0)
                                    {
                                        x_wall_v = xi[id];
                                    }
                                else
                                {
                                    x_wall_v = ((xi[id+1]-xi[id])/(yi[id+1]-yi[id]))*(yv-yi[id]) + xi[id];
                                }
                                xv -= x_wall_v;

                                if (abs(yv) < abs(y_norm) && abs(y_norm) > epsilon && z_build < height_eff && height_eff > epsilon)
                                {
                                    dn_v = sqrt((1.0-pow((yv/y_norm), 2.0))*(1.0-pow((z_build/height_eff),2.0))*pow((canyon_factor*Lr_local_v),2.0));
                                }
                                else
                                {
                                    dn_v = 0.0;
                                }
                                if (xv > farwake_factor*dn_v)
                                {
                                    v_wake_flag = 0;
                                }
                            }

                            i_w = ceil(((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/WGD->dx)-1;
                            j_w = ceil(((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/WGD->dy)-1;
                            //check if position in domain
                            if (i_w<WGD->nx-2 && i_w>0 && j_w<WGD->ny-2 && j_w>0) {

                                xp = (i_w+0.5)*WGD->dx-building_cent_x;
                                yp = (j_w+0.5)*WGD->dy-building_cent_y;

                                xw = xp*cos(upwind_dir)+yp*sin(upwind_dir);
                                yw = -xp*sin(upwind_dir)+yp*cos(upwind_dir);

                                Lr_local_w = Lr_node[id]+(yw-yi[id])*(Lr_node[id+1]-Lr_node[id])/(yi[id+1]-yi[id]);
                                if (perpendicular_flag[id] > 0) {
                                    x_wall_w = xi[id];
                                } else {
                                    x_wall_w = ((xi[id+1]-xi[id])/(yi[id+1]-yi[id]))*(yw-yi[id]) + xi[id];
                                }

                                xw -= x_wall_w;

                                if (abs(yw) < abs(y_norm) && abs(y_norm) > epsilon && z_build < height_eff && height_eff > epsilon) {
                                    dn_w = sqrt((1.0-pow(yw/y_norm, 2.0))*(1.0-pow(z_build/height_eff,2.0))*pow(canyon_factor*Lr_local_w,2.0));
                                } else {
                                    dn_w = 0.0;
                                }

                                if (xw > farwake_factor*dn_w) {
                                    w_wake_flag = 0;
                                }

                                icell_cent = i_w + j_w*(WGD->nx-1) + k*(WGD->nx-1)*(WGD->ny-1);
                                icell_face = i_w + j_w*WGD->nx + k*WGD->nx*WGD->ny;
                                if (dn_w > 0.0 && w_wake_flag == 1 && yw <= yi[id] && yw >= yi[id+1] &&
                                    WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2)
                                {
                                    // location of the center line
                                    i_cl = ceil((cos(upwind_dir)*xw + building_cent_x)/WGD->dx)-1;
                                    j_cl = ceil((sin(upwind_dir)*xw + building_cent_y)/WGD->dy)-1;

                                    //check if position in domain
                                    if (i_cl<WGD->nx-2 && i_cl>0 && j_cl<WGD->ny-2 && j_cl>0) {
                                        // index for centerline
                                        icell_cent_cl = i_cl + j_cl*(nx-1) + k*(ny-1)*(nx-1);
                                        icell_face_cl = i_cl + j_cl*nx + k*ny*nx;

                                        // velocity interpolated at the center line
                                        u_h=0.5*(WGD->u[icell_face_cl]+WGD->u[icell_face_cl+1]);
                                        v_h=0.5*(WGD->v[icell_face_cl]+WGD->v[icell_face_cl+nx]);
                                        //w_h=0.5*(WGD->w[icell_face_cl]+WGD->w[icell_face_cl+nx*ny]);
                                        U_a=u_h*cos(upwind_dir)+v_h*sin(upwind_dir);
                                        //U_a=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);

                                        //TGD->iturbflag[icell_cent_cl]=12;

                                    } else { //if centerline outside, assume U_centerline 90% of U_ref_l
                                        count_cl_outofbound++;
                                        U_a=0.9*U_ref_l;
                                    }
                                    // horizontal velocity different
                                    dU_ref_H = max(abs(U_ref_l-U_a),abs(U_ref_r-U_a));
                                    ustarH = kvonk*dU_ref_H;

                                    // Velocity interoplated at current location
                                    u_h=0.5*(WGD->u[icell_face]+WGD->u[icell_face+1]);
                                    v_h=0.5*(WGD->v[icell_face]+WGD->v[icell_face+nx]);
                                    //w_h=0.5*(WGD->w[icell_face]+WGD->w[icell_face+nx*ny]);
                                    U_a=u_h*cos(upwind_dir)+v_h*sin(upwind_dir);
                                    //U_a=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);

                                    // Far wake zone
                                    if (xw > dn_w) {
                                        if (canyon_factor == 1) {
                                            // horizontal mixing
                                            ustar=ustarH;
                                            float ustar2 = ustar*ustar;

                                            TGD->txx[icell_cent] = sigUConst*sigUConst*ustar2;
                                            TGD->tyy[icell_cent] = sigVConst*sigVConst*ustar2;
                                            TGD->tzz[icell_cent] = sigWConst*sigWConst*ustar2;

                                            TGD->txy[icell_cent] = abs(yw)/(0.5*width_eff)*ustar2;
                                            TGD->tyz[icell_cent] = 0.0;
                                            TGD->txz[icell_cent] = 0.0;

                                            TGD->Lm[icell_cent] = width_eff;

                                            TGD->iturbflag[icell_cent]=11;

                                            TGD->tke[icell_cent]=0.5*(TGD->txx[icell_cent]+TGD->tyy[icell_cent]+TGD->tzz[icell_cent]);
                                            TGD->CoEps[icell_cent]=5.7* pow(ustar,3.0)/(TGD->Lm[icell_cent]);

                                        }
                                    } else { // Downwind Cavity zone
                                        if ( dU_ref_H/(0.5*width_eff) >= U_a/(0.75*H)) {
                                            // horizontal mixing
                                            ustar=ustarH;
                                            float ustar2 = ustar*ustar;

                                            TGD->txx[icell_cent] = sigUConst*sigUConst*ustar2;
                                            TGD->tyy[icell_cent] = sigVConst*sigVConst*ustar2;
                                            TGD->tzz[icell_cent] = sigWConst*sigWConst*ustar2;

                                            TGD->txy[icell_cent] = abs(yw)/(0.5*width_eff)*ustar2;
                                            TGD->tyz[icell_cent] = 0.0;
                                            TGD->txz[icell_cent] = 0.0;

                                            TGD->Lm[icell_cent] = width_eff;

                                            TGD->iturbflag[icell_cent]=11;
                                        } else {
                                            // vertical mixing
                                            ustar=ustarV;
                                            float ustar2 = ustar*ustar;

                                            TGD->txx[icell_cent] = sigUConst*sigUConst*ustar2;
                                            TGD->tyy[icell_cent] = sigVConst*sigVConst*ustar2;
                                            TGD->tzz[icell_cent] = sigWConst*sigWConst*ustar2;

                                            TGD->txy[icell_cent] = 0;
                                            TGD->tyz[icell_cent] = -ustar2*sin(upwind_dir);//projection with wind dir
                                            TGD->txz[icell_cent] = -ustar2*cos(upwind_dir);//projection with wind dir

                                            TGD->Lm[icell_cent] = 0.75*H;

                                            TGD->iturbflag[icell_cent]=12;

                                        }

                                        TGD->tke[icell_cent]=0.5*(TGD->txx[icell_cent]+TGD->tyy[icell_cent]+TGD->tzz[icell_cent]);
                                        TGD->CoEps[icell_cent]=5.7* pow(ustar,3.0)/(TGD->Lm[icell_cent]);

                                    }
                                }
                                // end of wake
                                if (u_wake_flag == 0 && v_wake_flag == 0 && w_wake_flag == 0) {
                                    break;
                                }

                            }
                        }
                    }
                }
            }
        }
    }
    if(count_cl_outofbound>0) {
        std::cout << "[WARNING] " << count_cl_outofbound << " points of the centerline outside of domain " <<
            "(building id=" << building_id << ")" << std::endl;
    }

    return;
}
