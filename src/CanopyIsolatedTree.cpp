#include "CanopyIsolatedTree.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

// set et attenuation coefficient 
void CanopyIsolatedTree::canopyInitial(WINDSGeneralData *WGD,int building_id)
{
    // When THIS canopy calls this function, we need to do the
    // following:
    //readCanopy(nx, ny, nz, landuse_flag, num_canopies, lu_canopy_flag,
    //canopy_atten, canopy_top);
    
    // this function need to be called to defined the boundary of the canopy and the icellflags
    canopyDefineBoundary(WGD,building_id,cellFlagTree);
    
    if(ceil(1.5*k_end) > WGD->nz-1) {
        std::cerr << "ERROR domain too short for tree method" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (auto j=0; j<ny_canopy; j++) {
        for (auto i=0; i<nx_canopy; i++) {
            int icell_2d = i + j*nx_canopy;
            for (auto k=canopy_bot_index[icell_2d]; k<=canopy_top_index[icell_2d]; k++) {
                int icell_3d = i + j*nx_canopy + k*nx_canopy*ny_canopy;
                // initiate all attenuation coefficients to the canopy coefficient
                canopy_atten[icell_3d] = attenuationCoeff;     
            }
        }
    }

    return;
}


void CanopyIsolatedTree::canopyVegetation(WINDSGeneralData* WGD, int building_id)
{
  
    // apply canopy parameterization
    canopyCioncoParam(WGD);		
    
    // apply wake parameterization
    canopyWake(WGD,building_id);

    return;
}





void CanopyIsolatedTree::canopyWake(WINDSGeneralData* WGD, int building_id)
{

    int u_vegwake_flag(0),v_vegwake_flag(0),w_vegwake_flag(0);
    const int wake_stream_coef=11;
    const int wake_span_coef=4;
    const float lambda_sq=0.08;
    const float epsilon=10e-10;
    
    float z0;
    float z_b;
    float x_c,y_c,z_c,yw1,yw3,y_norm;
    float x_p,y_p,x_u,y_u,x_v,y_v,x_w,y_w;
    float x_wall,x_wall_u,x_wall_v,x_wall_w,dn_u,dn_v,dn_w;
    float u_defect,u_c,r_center,theta,delta,B_h;
    float ustar_wake(0),ustar_us(0),mag_us(0);

    int i(0),j(0),k(0);
    int k_bottom(1),k_top(WGD->nz-2);
    int icell_cent,icell_face,icell_2d;

    float Lt=0.5*W;
    Lr=H;

    icell_face = i_building_cent + j_building_cent*WGD->nx + k_end*WGD->nx*WGD->ny;
    u0_h = WGD->u0[icell_face];         // u velocity at the height of building at the centroid
    v0_h = WGD->v0[icell_face];         // v velocity at the height of building at the centroid

    upwind_dir=atan2(v0_h,u0_h);
    //float upwind_mag=sqrt(u0_h*u0_h + v0_h*v0_h);
        
    yw1 = 0.5*wake_span_coef*H;
    yw3 =-0.5*wake_span_coef*H;

    y_norm=yw1;

    for (auto k = 1; k <= k_start; k++) {
        k_bottom = k;
        if (base_height <= WGD->z[k])
            break;
    }
    
    for (auto k = k_start; k < WGD->nz-2; k++) {
        k_top = k;
        if (1.5*H < WGD->z[k+1])
            break;
    }
    k_top++;

    // mathod 1 -> location of upsteam data point (5% of 1/2 building length)
    // method 2 -> displaced log profile
    if(ustar_method==1) {
        i=ceil(((-1.05*Lt)*cos(upwind_dir)-0.0*sin(upwind_dir)+building_cent_x)/WGD->dx);
        j=ceil(((-1.05*Lt)*sin(upwind_dir)+0.0*cos(upwind_dir)+building_cent_y)/WGD->dy);
        k=k_end-1;

        // linearized indexes
        icell_2d = i + j*(WGD->nx);
        icell_face = i + j*(WGD->nx) + k*(WGD->nx)*(WGD->ny);
        
        z0=WGD->z0_domain_u[icell_2d];
            
        // upstream velocity
        float utmp=WGD->u0[icell_face];
        float vtmp=WGD->v0[icell_face];
        mag_us=sqrt(utmp*utmp+vtmp*vtmp);
        // height above ground 
        z_b=WGD->z[k]-base_height;
        // friction velocity 
        ustar_us=mag_us*WGD->vk/(log((z_b+z0)/z0));
        ustar_wake=ustar_us/mag_us; 
    } else if(ustar_method==2) {
        i=i_building_cent;
        j=j_building_cent;
        k=ceil(1.5*k_end);

        // linearized indexes
        icell_2d = (i+1-i_start) + (j+1-j_start)*nx_canopy;
        icell_face = i + j*(WGD->nx) + k*(WGD->nx)*(WGD->ny);
        
        // velocity above the canopy
        float utmp=WGD->u0[icell_face];
        float vtmp=WGD->v0[icell_face];
        mag_us=sqrt(utmp*utmp+vtmp*vtmp);
        z_b=WGD->z[k]-base_height;
        
        ustar_us=mag_us*WGD->vk/(log((z_b+canopy_d[icell_2d])/canopy_z0[icell_2d]));
        ustar_wake=ustar_us/mag_us; 
        
        //std::cout << z_b << " " << mag_us << " " << ustar_us << std::endl;
    } else {
        ustar_wake=0.323/6.15;
    }

    for(auto k=k_top; k>=k_bottom; k--) {
        
        // absolute z-coord within building above ground
        z_b=WGD->z[k]-base_height;
        // z-coord relative to center of tree (zMaxLAI) 
        z_c=z_b-zMaxLAI;
        
        for(auto y_idx=1;y_idx<2*ceil((yw1-yw3)/WGD->dxy);++y_idx) {
            
            // y-coord relative to center of tree (zMaxLAI) 
            y_c=0.5*float(y_idx)*WGD->dxy+yw3;
            
            if(std::abs(y_c) > std::abs(y_norm)) {
                continue;
            }else if(std::abs(y_c)>Lt && std::abs(y_c)<=yw1) {
                // y_cp=y_c-Lt(ibuild)
                // xwall=sqrt((Lt(ibuild)**2.)-(y_cp**2.))
                x_wall=0;
            } else {
                x_wall=0;
                //x_wall=sqrt(pow(Lt,2)-pow(y_c,2));
            }
            
            int x_idx_min = -1;
            for(auto x_idx=0;x_idx<=2.0*ceil(wake_stream_coef*Lr/WGD->dxy);++x_idx) {
                u_vegwake_flag=1;
                v_vegwake_flag=1;
                w_vegwake_flag=1;
                
                // x-coord relative to center of tree (zMaxLAI) 
                x_c=0.5*float(x_idx)*WGD->dxy;

                i=ceil(((x_c+x_wall)*cos(upwind_dir)-y_c*sin(upwind_dir)+building_cent_x)/WGD->dx);
                j=ceil(((x_c+x_wall)*sin(upwind_dir)+y_c*cos(upwind_dir)+building_cent_y)/WGD->dy);
                //check if in the domain
                if (i >= WGD->nx-2 && i <= 0 && j >= WGD->ny-2 && j <= 0)
                    break;
                
                // linearized indexes
                icell_cent = i+j*(WGD->nx-1)+k*(WGD->nx-1)*(WGD->ny-1);
                
                //check if not in canopy/building set start (was x_idx_min < 0) to x_idx_min > 0
                if (WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2 && x_idx_min < 0)
                    x_idx_min = x_idx;
                
                if (WGD->icellflag[icell_cent] == 0 || WGD->icellflag[icell_cent] == 2 || 
                    WGD->icellflag[icell_cent] == cellFlagTree)
                {
                    // check for canopy/building/terrain that will disrupt the wake
                    if (x_idx_min >= 0) {
                        if (WGD->ibuilding_flag[icell_cent] == building_id) {
                            x_idx_min = -1;
                        } else if (WGD->icellflag[i+j*(WGD->nx-1)+k*(WGD->nx-1)*(WGD->ny-1)] == 0 || 
                                   WGD->icellflag[i+j*(WGD->nx-1)+k*(WGD->nx-1)*(WGD->ny-1)] == 2) {
                            break;
                        }   
                    }
                }
                
                if(WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2 && 
                   WGD->icellflag[icell_cent] != cellFlagTree) {
                    // START OF WAKE VELOCITY PARAMETRIZATION

                    // wake u-values
                    // ij coord of u-face
                    int i_u = std::round(((x_c+x_wall)*cos(upwind_dir)-y_c*sin(upwind_dir)+building_cent_x)/WGD->dx);
                    int j_u = ((x_c+x_wall)*sin(upwind_dir)+y_c*cos(upwind_dir)+building_cent_y)/WGD->dy;
                    if(i_u < WGD->nx-1 && i_u > 0 && j_u < WGD->ny-1 && j_u > 0) {
                        // not rotated relative coordinate of u-face
                        x_p = i_u*WGD->dx-building_cent_x;
                        y_p = (j_u+0.5)*WGD->dy-building_cent_y;
                        // rotated relative coordinate of u-face              
                        x_u = x_p*cos(upwind_dir)+y_p*sin(upwind_dir);
                        y_u = -x_p*sin(upwind_dir)+y_p*cos(upwind_dir);
                        
                        if(std::abs(y_u) > std::abs(y_norm)) {
                            break;
                        } else {
                            x_wall_u=0;
                        }
                        
                        //adjusted downstream value
                        x_u -= x_wall_u;
                       
                        if(std::abs(y_u) < std::abs(y_norm) && std::abs(y_norm) > epsilon && 
                           H > epsilon) {
                            dn_u=H;
                        } else {
                            dn_u = 0.0;
                        }
                        
                        if(x_u > wake_stream_coef*dn_u)
                            u_vegwake_flag = 0;
                        
                        // linearized indexes
                        icell_cent = i_u + j_u*(WGD->nx-1)+k*(WGD->nx-1)*(WGD->ny-1);
                        icell_face = i_u + j_u*WGD->nx+k*WGD->nx*WGD->ny;
                        
                        if(dn_u > 0.0 && u_vegwake_flag == 1 && 
                           WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
                            
                            // polar coordinate in the wake
                            r_center=sqrt(pow(z_c,2)+pow(y_u,2));
                            theta=atan2(z_c,y_u);
                            
                            // FM - ellipse equation:
                            B_h=Bfunc(x_u/H);
                            delta=(B_h-1.15)/sqrt(1-(1-pow((B_h-1.15)/(B_h+1.15),2))*pow(cos(theta),2))*H;

                            // check if within the wake
                            if(r_center<0.5*delta) {
                                // get velocity deficit
                                u_c=ucfunc(x_u/H,ustar_wake);                    
                                u_defect=u_c*(exp(-(r_center*r_center)/(lambda_sq*delta*delta)));
                                //std::cout << r_center << " " << delta << " " << ustar_wake << " " << u_c << std::endl;
                                // apply parametrization
                                WGD->u0[icell_face]*=(1. - std::abs(u_defect)*cos(upwind_dir));
                            } //if (r_center<delta/1)
                        }
                    }
                    
                    // wake v-values
                    // ij coord of v-face
                    int i_v = ((x_c+x_wall)*cos(upwind_dir)-y_c*sin(upwind_dir)+building_cent_x)/WGD->dx;
                    int j_v = std::round(((x_c+x_wall)*sin(upwind_dir)+y_c*cos(upwind_dir)+building_cent_y)/WGD->dy);
                    if (i_v<WGD->nx-1 && i_v>0 && j_v<WGD->ny-1 && j_v>0) {
                        // not rotated relative coordinate of v-face
                        x_p = (i_v+0.5)*WGD->dx-building_cent_x;
                        y_p = j_v*WGD->dy-building_cent_y;
                        // rotated relative coordinate of u-face
                        x_v = x_p*cos(upwind_dir)+y_p*sin(upwind_dir);
                        y_v = -x_p*sin(upwind_dir)+y_p*cos(upwind_dir);
                        
                        if(std::abs(y_v) > std::abs(y_norm)) {
                            break;
                        } else {
                            x_wall_v=0;
                        }
                        
                        //adjusted downstream value
                        x_v -= x_wall_v;
                        
                        if(std::abs(y_v) < std::abs(y_norm) && std::abs(y_norm) > epsilon && 
                           H > epsilon) {
                            dn_v=H;
                        } else {
                            dn_v = 0.0;
                        }
                        
                        if(x_v > wake_stream_coef*dn_v) 
                            v_vegwake_flag = 0;
                        
                        // linearized indexes
                        icell_cent = i_u + j_u*(WGD->nx-1)+k*(WGD->nx-1)*(WGD->ny-1);
                        icell_face = i_v + j_v*WGD->nx+k*WGD->nx*WGD->ny;
                        
                        if(dn_v > 0.0 && v_vegwake_flag == 1 && 
                           WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
                            
                            // polar coordinate in the wake
                            r_center=sqrt(pow(z_c,2)+pow(y_v,2));
                            theta=atan2(z_c,y_v);
                            
                            // FM - ellipse equation:
                            B_h=Bfunc(x_v/H);
                            delta=(B_h-1.15)/sqrt(1-(1-pow((B_h-1.15)/(B_h+1.15),2))*pow(cos(theta),2))*H;

                            // check if within the wake
                            if(r_center<0.5*delta) {
                                // get velocity deficit
                                u_c=ucfunc(x_v/H,ustar_wake);                    
                                u_defect=u_c*(exp(-(r_center*r_center)/(lambda_sq*delta*delta)));
                                // apply parametrization
                                WGD->v0[icell_face]*=(1. - std::abs(u_defect)*cos(upwind_dir));
                            } //if (r_center<delta/1)
                        }
                    }
                
                    // wake celltype w-values
                    // ij coord of cell-center
                    int i_w = ceil(((x_c+x_wall)*cos(upwind_dir)-y_c*sin(upwind_dir)+building_cent_x)/WGD->dx);
                    int j_w = ceil(((x_c+x_wall)*sin(upwind_dir)+y_c*cos(upwind_dir)+building_cent_y)/WGD->dy);

                    if (i_w<WGD->nx-1 && i_w>0 && j_w<WGD->ny-1 && j_w>0) {
                        // not rotated relative coordinate of cell-center
                        x_p = (i_w+0.5)*WGD->dx-building_cent_x;
                        y_p = (j_w+0.5)*WGD->dy-building_cent_y;
                        // rotated relative coordinate of cell-center
                        x_w = x_p*cos(upwind_dir)+y_p*sin(upwind_dir);
                        y_w = -x_p*sin(upwind_dir)+y_p*cos(upwind_dir);
                        
                        if(std::abs(y_w) > std::abs(y_norm)) {
                            break;
                        } else {
                            x_wall_w=0;
                        }
                        
                        //adjusted downstream value
                        x_w -= x_wall_w;
                        
                        if(std::abs(y_w) < std::abs(y_norm) && std::abs(y_norm) > epsilon && 
                           H > epsilon) {
                            dn_w=H;
                        } else {
                            dn_w = 0.0;
                        }
                        
                        if(x_w > wake_stream_coef*dn_w) 
                            w_vegwake_flag = 0;
                        
                        // linearized indexes
                        icell_cent = i_w + j_w*(WGD->nx-1)+k*(WGD->nx-1)*(WGD->ny-1);
                        //icell_face = i_v + j_v*WGD->nx+k*WGD->nx*WGD->ny;
                        
                        if(dn_w > 0.0 && w_vegwake_flag == 1 && 
                           WGD->icellflag[icell_cent] != 0 && WGD->icellflag[icell_cent] != 2) {
                            
                            // polar coordinate in the wake
                            r_center=sqrt(pow(z_c,2)+pow(y_w,2));
                            theta=atan2(z_c,y_w);
                            
                            // FM - ellipse equation:
                            B_h=Bfunc(x_w/H);
                            delta=(B_h-1.15)/sqrt(1-(1-pow((B_h-1.15)/(B_h+1.15),2))*pow(cos(theta),2))*H;
                            
                            // check if within the wake
                            if(r_center<0.5*delta) {
                                // get velocity deficit
                                u_c=ucfunc(x_w/H,ustar_wake);                    
                                u_defect=u_c*(exp(-(r_center*r_center)/(lambda_sq*delta*delta)));
                                // apply parametrization
                                WGD->icellflag[icell_cent]=cellFlagWake;
                            } //if (r_center<delta/1)
                        }
                    }
                    // if u,v, and w are done -> exit x-loop 
                    if (u_vegwake_flag == 0 && v_vegwake_flag == 0 && w_vegwake_flag == 0)
                        break;
                    // END OF WAKE VELOCITY PARAMETRIZATION
                } 
            } // end of x-loop (stream-wise)
        } // end of y-loop (span-wise)
    } // end of z-loop

    return;
}

