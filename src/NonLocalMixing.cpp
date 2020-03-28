#include "PolyBuilding.h"

// These take care of the circular reference
#include "URBInputData.h"
#include "URBGeneralData.h"
#include "TURBGeneralData.h"

void PolyBuilding::NonLocalMixing (URBGeneralData* UGD, TURBGeneralData* TGD) 
{
    
    int nx = UGD->nx;
    int ny = UGD->ny;
    int nz = UGD->nz;

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
    float u_h,v_h,w_h; //interpolated velocites

    // x,y positons of the polybuilding verteces in the rotated coord syst rotated with the wind dir and building center
    float xp,yp; //working variables
    

    float ustar=0,ustarH=0,ustarV=0; //fiction velocities
    int id=1;
    
    float z_build;                  // z value of each building point from its base height
    int k_bottom, k_top;
    int kk;

    int id_face,id_cell;
    int building_id=1;
    
    // interpolation of velocity at the top of the building
    id_face=i_building_cent + j_building_cent*nx + (k_end+1)*nx*ny;
    u_h=0.5*(UGD->u[id_face]+UGD->u[id_face+1]); 
    v_h=0.5*(UGD->v[id_face]+UGD->v[id_face+nx]); 
    w_h=0.5*(UGD->w[id_face]+UGD->w[id_face+nx*ny]);
    
    // verical velocity reference and verical fiction velocity
    U_ref_V=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);
    ustarV=kvonk*U_ref_V;
    
    // scale factor = scale the dxy as a function of the angle of the flow
    float scale_factor=1;
    if( 1.0/cos(upwind_dir) <= sqrt(2) ) {
        scale_factor=1.0/cos(upwind_dir);
    } else {
        scale_factor=1.0/sin(upwind_dir);
    }
    float dxy=scale_factor*UGD->dxy;
     
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

    // x-coord of maximun downstream vertex
    float xp_d;
    maxIndex = std::max_element(xp_i.begin(),xp_i.end()) - xp_i.begin();  
    xp_d=xp_i[maxIndex]-0.5*dxy;    

    // set itrubflag for debug
    if (false) {
        float x_r, y_r;
        x_r = cos(upwind_dir)*xp_r - sin(upwind_dir)*yp_r + building_cent_x;
        y_r = sin(upwind_dir)*xp_r + cos(upwind_dir)*yp_r + building_cent_y;
        
        int i_r = floor(x_r/UGD->dx);
        int j_r = floor(y_r/UGD->dy);
        
        for(int k=k_start;k<k_end;++k) {
            id_cell = i_r + j_r*(nx-1) + k*(ny-1)*(nx-1);
            TGD->iturbflag[id_cell]=12;
        }
        
        float x_l, y_l;
        x_l = cos(upwind_dir)*xp_l - sin(upwind_dir)*yp_l + building_cent_x;
        y_l = sin(upwind_dir)*xp_l + cos(upwind_dir)*yp_l + building_cent_y;
        
        int i_l = floor(x_l/UGD->dx);
        int j_l = floor(y_l/UGD->dy);
        
        for(int k=k_start;k<k_end;++k) {
            id_cell = i_l + j_l*(nx-1) + k*(ny-1)*(nx-1);
            TGD->iturbflag[id_cell]=12;
        }
    }

    ///////
    float xp_ref_r, yp_ref_r;
    float x_ref_r, y_ref_r;
    xp_ref_r=xp_r;
    yp_ref_r=yp_r-3.0*dxy;
    
    x_ref_r = cos(upwind_dir)*xp_ref_r - sin(upwind_dir)*yp_ref_r + building_cent_x;
    y_ref_r = sin(upwind_dir)*xp_ref_r + cos(upwind_dir)*yp_ref_r + building_cent_y;

    int i_ref_r = floor(x_ref_r/UGD->dx);
    int j_ref_r = floor(y_ref_r/UGD->dy);
    
    for(int k=k_start;k<k_end;++k) {
        id_cell = i_ref_r + j_ref_r*(nx-1) + k*(ny-1)*(nx-1);
        TGD->iturbflag[id_cell]=12;
    }
    
    ///////
    float xp_ref_l, yp_ref_l;
    float x_ref_l, y_ref_l;
    xp_ref_l=xp_l;
    yp_ref_l=yp_l+3.0*dxy;
    
    x_ref_l = cos(upwind_dir)*xp_ref_l - sin(upwind_dir)*yp_ref_l + building_cent_x;
    y_ref_l = sin(upwind_dir)*xp_ref_l + cos(upwind_dir)*yp_ref_l + building_cent_y;

    int i_ref_l = floor(x_ref_l/UGD->dx);
    int j_ref_l = floor(y_ref_l/UGD->dy);
    
    for(int k=k_start;k<k_end;++k) {
        id_cell = i_ref_l + j_ref_l*(nx-1) + k*(ny-1)*(nx-1);
        TGD->iturbflag[id_cell]=12;
    }
    
    for (auto k = 1; k <= k_start; k++) {
        k_bottom = k;
        if (base_height <= UGD->z[k])
            break;
    }
    
    for (auto k = k_start; k < UGD->nz-1; k++) {
        k_top = k;
        if (height_eff < UGD->z[k+1])
            break;
    }
    
    for (auto k = k_start; k < k_end; k++) {
        kk = k;
        if (0.75*H+base_height <= UGD->z[k])
            break;
    }
    
    for (auto k=k_top; k>=k_bottom; k--) {
        z_build = UGD->z[k] - base_height;
        
        //reference velocity left
        id_face = i_ref_l + j_ref_l*nx + k*ny*nx;
        u_h=0.5*(UGD->u[id_face]+UGD->u[id_face+1]); 
        v_h=0.5*(UGD->v[id_face]+UGD->v[id_face+nx]); 
        w_h=0.5*(UGD->w[id_face]+UGD->w[id_face+nx*ny]);
        U_ref_l=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);
        
        //reference velocity right
        id_face = i_ref_r + j_ref_r*nx + k*ny*nx;
        u_h=0.5*(UGD->u[id_face]+UGD->u[id_face+1]); 
        v_h=0.5*(UGD->v[id_face]+UGD->v[id_face+nx]); 
        w_h=0.5*(UGD->w[id_face]+UGD->w[id_face+nx*ny]);
        U_ref_r=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);
        
        for (auto x_ds=0; x_ds < 2.0*ceil(3.0*Lr/dxy); x_ds++) {    
            xp=xp_d+dxy+x_ds*0.5*dxy;
            yp=0.0;
            
            int i = (cos(upwind_dir)*xp - sin(upwind_dir)*yp + building_cent_x)/UGD->dx;
            int j = (sin(upwind_dir)*xp + cos(upwind_dir)*yp + building_cent_y)/UGD->dy;
            id_cell = i + j*(nx-1) + k*(ny-1)*(nx-1);    
            id_face = i + j*nx + k*ny*nx;
            
            if ( i > UGD->nx-2 && i < 0 && j > UGD->ny-2 && j < 0) {
                break;
            }
            if (TGD->iturbflag[id_cell] != 0 && TGD->iturbflag[id_cell] != 10) {
                
                u_h=0.5*(UGD->u[id_face]+UGD->u[id_face+1]); 
                v_h=0.5*(UGD->v[id_face]+UGD->v[id_face+nx]); 
                w_h=0.5*(UGD->w[id_face]+UGD->w[id_face+nx*ny]);
                U_a=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);
                dU_ref_H = max(abs(U_ref_l-U_a),abs(U_ref_r-U_a));
                ustarH = kvonk*dU_ref_H;
                
                if ( dU_ref_H/(0.5*width_eff) >= U_a/(0.75*H)) { 
                    ustar=ustarH;
                    float ustar2 = ustar*ustar;
                    
                    TGD->tau11[id_cell] = sigUConst*sigUConst*ustar2;
                    TGD->tau22[id_cell] = sigVConst*sigVConst*ustar2;
                    TGD->tau33[id_cell] = sigWConst*sigWConst*ustar2;
                    
                    TGD->tau12[id_cell] = 0.0*ustar2;
                    TGD->tau23[id_cell] = 0.0; 
                    TGD->tau13[id_cell] = 0.0;
                    
                    TGD->Lm[id_cell] = width_eff;
        
                    TGD->iturbflag[id_cell]=11; 
                } else {
                    ustar=ustarV;
                    float ustar2 = ustar*ustar;
                    
                    TGD->tau11[id_cell] = sigUConst*sigUConst*ustar2;
                    TGD->tau22[id_cell] = sigVConst*sigVConst*ustar2;
                    TGD->tau33[id_cell] = sigWConst*sigWConst*ustar2;
                    
                    TGD->tau12[id_cell] = 0;
                    TGD->tau23[id_cell] = -ustar2*sin(upwind_dir);//projection with wind dir
                    TGD->tau13[id_cell] = -ustar2*cos(upwind_dir);//projection with wind dir
                    
                    TGD->Lm[id_cell] = 0.75*H; 
                    
                    TGD->iturbflag[id_cell]=12;
                } 
                
                TGD->tke[id_cell]=0.5*(TGD->tau11[id_cell]+TGD->tau22[id_cell]+TGD->tau33[id_cell]);
                TGD->CoEps[id_cell]=5.7* pow(ustar,3.0)/(TGD->Lm[id_cell]); 
            
                //left main wake
                for(auto y_sl=0; y_sl < 2*ceil(abs(yp_l)/dxy); ++y_sl) {
                    yp= y_sl*0.5*dxy;
                    
                    int i = (cos(upwind_dir)*xp - sin(upwind_dir)*yp + building_cent_x)/UGD->dx;
                    int j = (sin(upwind_dir)*xp + cos(upwind_dir)*yp + building_cent_y)/UGD->dy;
                    id_cell = i + j*(nx-1) + k*(ny-1)*(nx-1);    
                    id_face = i + j*nx + k*ny*nx;
                
                    if ( i > UGD->nx-2 && i < 0 && j > UGD->ny-2 && j < 0) { 
                        break;
                    }
                    if (TGD->iturbflag[id_cell] != 0 && TGD->iturbflag[id_cell] != 10) {
                        u_h=0.5*(UGD->u[id_face]+UGD->u[id_face+1]); 
                        v_h=0.5*(UGD->v[id_face]+UGD->v[id_face+nx]); 
                        w_h=0.5*(UGD->w[id_face]+UGD->w[id_face+nx*ny]);
                        U_a=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);
                        
                        if ( dU_ref_H/(0.5*width_eff) >= U_a/(0.75*H)) { 
                            ustar=ustarH;
                            float ustar2 = ustar*ustar;
                            
                            TGD->tau11[id_cell] = sigUConst*sigUConst*ustar2;
                            TGD->tau22[id_cell] = sigVConst*sigVConst*ustar2;
                            TGD->tau33[id_cell] = sigWConst*sigWConst*ustar2;
                            
                            TGD->tau12[id_cell] = abs(yp)/(0.5*width_eff)*ustar2;
                            TGD->tau23[id_cell] = 0.0; 
                            TGD->tau13[id_cell] = 0.0;
                        
                            TGD->Lm[id_cell] = width_eff;
                            
                            TGD->iturbflag[id_cell]=11; 
                        } else {
                            ustar=ustarV;
                            float ustar2 = ustar*ustar;
                            
                            TGD->tau11[id_cell] = sigUConst*sigUConst*ustar2;
                            TGD->tau22[id_cell] = sigVConst*sigVConst*ustar2;
                            TGD->tau33[id_cell] = sigWConst*sigWConst*ustar2;
                            
                            TGD->tau12[id_cell] = 0;
                            TGD->tau23[id_cell] = -ustar2*sin(upwind_dir);//projection with wind dir
                            TGD->tau13[id_cell] = -ustar2*cos(upwind_dir);//projection with wind dir
                            
                            TGD->Lm[id_cell] = 0.75*H; 
                            
                            TGD->iturbflag[id_cell]=12;
                        } 
                        
                        TGD->tke[id_cell]=0.5*(TGD->tau11[id_cell]+TGD->tau22[id_cell]+TGD->tau33[id_cell]);
                        TGD->CoEps[id_cell]=5.7* pow(ustar,3.0)/(TGD->Lm[id_cell]); 
                        
                    }
                }
                
                //right main wake
                for(auto y_sr=0; y_sr < 2*ceil(abs(yp_r)/dxy); ++y_sr) {
                    yp= -y_sr*0.5*dxy;
                    
                    int i = (cos(upwind_dir)*xp - sin(upwind_dir)*yp + building_cent_x)/UGD->dx;
                    int j = (sin(upwind_dir)*xp + cos(upwind_dir)*yp + building_cent_y)/UGD->dy;
                    id_cell = i + j*(nx-1) + k*(ny-1)*(nx-1);    
                    id_face = i + j*nx + k*ny*nx;

                    if ( i > UGD->nx-2 && i < 0 && j > UGD->ny-2 && j < 0) {
                        break;
                    }
                    if (TGD->iturbflag[id_cell] != 0 && TGD->iturbflag[id_cell] != 10) {
                        u_h=0.5*(UGD->u[id_face]+UGD->u[id_face+1]); 
                        v_h=0.5*(UGD->v[id_face]+UGD->v[id_face+nx]); 
                        w_h=0.5*(UGD->w[id_face]+UGD->w[id_face+nx*ny]);
                        U_a=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);
                        
                        if ( dU_ref_H/(0.5*width_eff) >= U_a/(0.75*H)) { 
                            ustar=ustarH;
                            float ustar2 = ustar*ustar;
                            
                            TGD->tau11[id_cell] = sigUConst*sigUConst*ustar2;
                            TGD->tau22[id_cell] = sigVConst*sigVConst*ustar2;
                            TGD->tau33[id_cell] = sigWConst*sigWConst*ustar2;
                            
                            TGD->tau12[id_cell] = abs(yp)/(0.5*width_eff)*ustar2;
                            TGD->tau23[id_cell] = 0.0; 
                            TGD->tau13[id_cell] = 0.0;
                    
                            TGD->Lm[id_cell] = width_eff;
                            
                            TGD->iturbflag[id_cell]=11; 
                        } else {
                            ustar=ustarV;
                            float ustar2 = ustar*ustar;
                            
                            TGD->tau11[id_cell] = sigUConst*sigUConst*ustar2;
                            TGD->tau22[id_cell] = sigVConst*sigVConst*ustar2;
                            TGD->tau33[id_cell] = sigWConst*sigWConst*ustar2;
                            
                            TGD->tau12[id_cell] = 0;
                            TGD->tau23[id_cell] = -ustar2*sin(upwind_dir);//projection with wind dir
                            TGD->tau13[id_cell] = -ustar2*cos(upwind_dir);//projection with wind dir
                            
                            TGD->Lm[id_cell] = 0.75*H; 
                            
                            TGD->iturbflag[id_cell]=12;
                        } 
                        
                        TGD->tke[id_cell]=0.5*(TGD->tau11[id_cell]+TGD->tau22[id_cell]+TGD->tau33[id_cell]);
                        TGD->CoEps[id_cell]=5.7* pow(ustar,3.0)/(TGD->Lm[id_cell]); 
                        
                    }
                }
            }
        }
        
        // down stream reference point (xp=most downstream vertex + dxy,yp=0)
        xp=xp_d+dxy;
        yp=0.0;
        
        int i = (cos(upwind_dir)*xp - sin(upwind_dir)*yp + building_cent_x)/UGD->dx;
        int j = (sin(upwind_dir)*xp + cos(upwind_dir)*yp + building_cent_y)/UGD->dy;
        id_face = i + j*nx + k*ny*nx;
        
        u_h=0.5*(UGD->u[id_face]+UGD->u[id_face+1]); 
        v_h=0.5*(UGD->v[id_face]+UGD->v[id_face+nx]); 
        w_h=0.5*(UGD->w[id_face]+UGD->w[id_face+nx*ny]);
        U_a=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);
        dU_ref_H = max(abs(U_ref_l-U_a),abs(U_ref_r-U_a));
        ustarH = kvonk*dU_ref_H;
        
        //left side wake
        for (auto x_ds=0; x_ds <= 2*ceil(abs(xp_d-xp_l)/dxy); x_ds++) {    
            xp=xp_l+x_ds*0.5*dxy;
            yp = 0.0;
            
        
            for(auto y_sl=0; y_sl < 2*ceil(abs(yp_l)/dxy); ++y_sl) {
                yp= y_sl*0.5*dxy;
                
                int i = (cos(upwind_dir)*xp - sin(upwind_dir)*yp + building_cent_x)/UGD->dx;
                int j = (sin(upwind_dir)*xp + cos(upwind_dir)*yp + building_cent_y)/UGD->dy;
                int cell = i + j*(nx-1) + k*(ny-1)*(nx-1);    
                
                if ( i > UGD->nx-2 && i < 0 && j > UGD->ny-2 && j < 0) {
                    break;
                }
                if (TGD->iturbflag[cell] != 0 && TGD->iturbflag[cell] != 10) {
                    u_h=0.5*(UGD->u[id_face]+UGD->u[id_face+1]); 
                    v_h=0.5*(UGD->v[id_face]+UGD->v[id_face+nx]); 
                    w_h=0.5*(UGD->w[id_face]+UGD->w[id_face+nx*ny]);
                    U_a=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);
                    
                    if ( dU_ref_H/(0.5*width_eff) >= U_a/(0.75*H)) { 
                        ustar=ustarH;
                        float ustar2 = ustar*ustar;
                        
                        TGD->tau11[id_cell] = sigUConst*sigUConst*ustar2;
                        TGD->tau22[id_cell] = sigVConst*sigVConst*ustar2;
                        TGD->tau33[id_cell] = sigWConst*sigWConst*ustar2;
                        
                        TGD->tau12[id_cell] = abs(yp)/(0.5*width_eff)*ustar2;
                        TGD->tau23[id_cell] = 0.0; 
                        TGD->tau13[id_cell] = 0.0;
                        
                        TGD->Lm[id_cell] = width_eff;
                        
                        TGD->iturbflag[id_cell]=11; 
                    } else {
                        ustar=ustarV;
                        float ustar2 = ustar*ustar;
                        
                        TGD->tau11[id_cell] = sigUConst*sigUConst*ustar2;
                        TGD->tau22[id_cell] = sigVConst*sigVConst*ustar2;
                        TGD->tau33[id_cell] = sigWConst*sigWConst*ustar2;
                        
                        TGD->tau12[id_cell] = 0;
                        TGD->tau23[id_cell] = -ustar2*sin(upwind_dir);//projection with wind dir
                        TGD->tau13[id_cell] = -ustar2*cos(upwind_dir);//projection with wind dir
                        
                        TGD->Lm[id_cell] = 0.75*H; 
                        
                        TGD->iturbflag[id_cell]=12;
                    } 
                    
                    TGD->tke[id_cell]=0.5*(TGD->tau11[id_cell]+TGD->tau22[id_cell]+TGD->tau33[id_cell]);
                    TGD->CoEps[id_cell]=5.7* pow(ustar,3.0)/(TGD->Lm[id_cell]);    
                    
                }
            } 
        }
        
        //right side wake
        for (auto x_ds=0; x_ds <= 2*ceil(abs(xp_d-xp_r)/dxy); x_ds++) {    
            xp=xp_r+x_ds*0.5*dxy;
            yp=0.0;
            
        
            for(auto y_sr=0; y_sr < 2*ceil(abs(yp_r)/dxy); ++y_sr) {
                yp=-y_sr*0.5*dxy;
                
                int i = (cos(upwind_dir)*xp - sin(upwind_dir)*yp + building_cent_x)/UGD->dx;
                int j = (sin(upwind_dir)*xp + cos(upwind_dir)*yp + building_cent_y)/UGD->dy;
                id_cell = i + j*(nx-1) + k*(ny-1)*(nx-1);    
                
                if ( i > UGD->nx-2 && i < 0 && j > UGD->ny-2 && j < 0) {
                    break;
                }
                if (TGD->iturbflag[id_cell] != 0 && TGD->iturbflag[id_cell] != 10) {
                    u_h=0.5*(UGD->u[id_face]+UGD->u[id_face+1]); 
                    v_h=0.5*(UGD->v[id_face]+UGD->v[id_face+nx]); 
                    w_h=0.5*(UGD->w[id_face]+UGD->w[id_face+nx*ny]);
                    U_a=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);
                    
                    if ( dU_ref_H/(0.5*width_eff) >= U_a/(0.75*H)) { 
                        ustar=ustarH;
                        float ustar2 = ustar*ustar;
                        
                        TGD->tau11[id_cell] = sigUConst*sigUConst*ustar2;
                        TGD->tau22[id_cell] = sigVConst*sigVConst*ustar2;
                        TGD->tau33[id_cell] = sigWConst*sigWConst*ustar2;
                        
                        TGD->tau12[id_cell] = abs(yp)/(0.5*width_eff)*ustar2;
                        TGD->tau23[id_cell] = 0.0; 
                        TGD->tau13[id_cell] = 0.0;
                        
                        TGD->Lm[id_cell] = width_eff;
                        
                        TGD->iturbflag[id_cell]=11; 
                    } else {
                        ustar=ustarV;
                        float ustar2 = ustar*ustar;
                        
                        TGD->tau11[id_cell] = sigUConst*sigUConst*ustar2;
                        TGD->tau22[id_cell] = sigVConst*sigVConst*ustar2;
                        TGD->tau33[id_cell] = sigWConst*sigWConst*ustar2;
                        
                        TGD->tau12[id_cell] = 0;
                        TGD->tau23[id_cell] = -ustar2*sin(upwind_dir);//projection with wind dir
                        TGD->tau13[id_cell] = -ustar2*cos(upwind_dir);//projection with wind dir
                        
                        TGD->Lm[id_cell] = 0.75*H; 
                        
                        TGD->iturbflag[id_cell]=12;
                    } 
                    
                    TGD->tke[id_cell]=0.5*(TGD->tau11[id_cell]+TGD->tau22[id_cell]+TGD->tau33[id_cell]);
                    TGD->CoEps[id_cell]=5.7* pow(ustar,3.0)/(TGD->Lm[id_cell]); 
                    
                }
            }   
        }
    }   
    
    return;

}




/*
if(ustarH>ustarV){
        ustar=ustarH;
        Lm=wth.at(ibld);
    }
    else{
        ustar=ustarV;
        Lm=0.75*hgt.at(ibld);
    }
    
    float ustar2 = ustar*ustar;
    TGD->tau11[id_cell] = sigUConst*sigUConst*ustar2;
    TGD->tau22[id_cell] = sigVConst*sigVConst*ustar2;
    TGD->tau33[id_cell] = sigWConst*sigWConst*ustar2;
    if(ustarH>ustarV) {
        ustar=ustarH;
        float ustar2 = ustar*ustar;
        
        TGD->tau11[id_cell] = sigUConst*sigUConst*ustar2;
        TGD->tau22[id_cell] = sigVConst*sigVConst*ustar2;
        TGD->tau33[id_cell] = sigWConst*sigWConst*ustar2;
        
        TGD->tau12[id_cell] = ustar2; //to finish (y-ya)/yW/2+
        TGD->tau23[id_cell] = 0.0; 
        TGD->tau13[id_cell] = 0.0;
        
        TGD->Lm[id_cell] = 0; //effective width
        
    } else {
        ustar=ustarV;
        float ustar2 = ustar*ustar;
        
        TGD->tau11[id_cell] = sigUConst*sigUConst*ustar2;
        TGD->tau22[id_cell] = sigVConst*sigVConst*ustar2;
        TGD->tau33[id_cell] = sigWConst*sigWConst*ustar2;
        
        TGD->tau12[id_cell] = 0;
        TGD->tau23[id_cell] = -ustar2;//to finish (projection with wind dir)
        TGD->tau13[id_cell] = -ustar2;//to finish (projection with wind dir)
        
        TGD->Lm[id_cell] = 0; //height
    }

    TGD->tke[id_cell]=0.5*(TGD->tau11[id_cell]+TGD->tau22[id_cell]+TGD->tau33[id_cell]);
    TGD->CoEps[id_cell]=5.7* pow(ustar,3.0)/(TGD->Lm[id_cell]); 
    
*/
