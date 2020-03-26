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

    float ustar=0,ustarH=0,ustarV=0;
    int id=1;
    
    
    /*
    for(int i=i_start;i<i_end-1+(int)3.0*Lr/UGD->dx;++i) {
        for(int j=j_start;j<j_end-1;++j) {
            for(int k=k_start;k<k_end;++k) {
                int cell = i + j*(nx-1) + k*(ny-1)*(nx-1);                
                if(TGD->iturbflag[cell] != 0) {
                    TGD->iturbflag[cell] = 10;
                }
            }
        }
    }
    */
    
    float vel_building_cent;
    float u_h,v_h,w_h;
    int index_building_face = i_building_cent + j_building_cent*nx + (k_end+1)*nx*ny;

    int building_id=1;
    
    u_h=0.5*(UGD->u[index_building_face]+UGD->u[index_building_face+1]); 
    v_h=0.5*(UGD->v[index_building_face]+UGD->v[index_building_face+nx]); 
    w_h=0.5*(UGD->w[index_building_face]+UGD->w[index_building_face+nx*ny]);
    vel_building_cent=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);
    

    std::cout << "upwind dir: " << upwind_dir << " wind velocity: "<< vel_building_cent << std::endl;
    

    for(int k=k_start;k<k_end;++k) {
        int cell = i_building_cent + j_building_cent*(nx-1) + k*(ny-1)*(nx-1);
        TGD->iturbflag[cell]=11;
    }
    
    int cell = i_building_cent + j_building_cent*(nx-1) + (k_end+1)*(ny-1)*(nx-1);
    TGD->iturbflag[cell]=11;
    
    /*
    if(ustarH>ustarV){
        ustar=ustarH;
        Lm=wth.at(ibld);
    }
    else{
        ustar=ustarV;
        Lm=0.75*hgt.at(ibld);
    }
    */
    
    /*
    float ustar2 = ustar*ustar;
    TGD->tau11[id] = sigUConst*sigUConst*ustar2;
    TGD->tau22[id] = sigVConst*sigVConst*ustar2;
    TGD->tau33[id] = sigWConst*sigWConst*ustar2;
    if(ustarH>ustarV) {
        ustar=ustarH;
        float ustar2 = ustar*ustar;
        
        TGD->tau11[id] = sigUConst*sigUConst*ustar2;
        TGD->tau22[id] = sigVConst*sigVConst*ustar2;
        TGD->tau33[id] = sigWConst*sigWConst*ustar2;
        
        TGD->tau12[id] = ustar2; //to finish (y-ya)/yW/2+
        TGD->tau23[id] = 0.0; 
        TGD->tau13[id] = 0.0;
        
        TGD->Lm[id] = 0; //effective width
        
    } else {
        ustar=ustarV;
        float ustar2 = ustar*ustar;
        
        TGD->tau11[id] = sigUConst*sigUConst*ustar2;
        TGD->tau22[id] = sigVConst*sigVConst*ustar2;
        TGD->tau33[id] = sigWConst*sigWConst*ustar2;
        
        TGD->tau12[id] = 0;
        TGD->tau23[id] = -ustar2;//to finish (projection with wind dir)
        TGD->tau13[id] = -ustar2;//to finish (projection with wind dir)
        
        TGD->Lm[id] = 0; //height
    }

    TGD->tke[id]=0.5*(TGD->tau11[id]+TGD->tau22[id]+TGD->tau33[id]);
    TGD->CoEps[id]=5.7* pow(ustar,3.0)/(TGD->Lm[id]); 
    */

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
            int cell = i_r + j_r*(nx-1) + k*(ny-1)*(nx-1);
            TGD->iturbflag[cell]=12;
        }
        
        float x_l, y_l;
        x_l = cos(upwind_dir)*xp_l - sin(upwind_dir)*yp_l + building_cent_x;
        y_l = sin(upwind_dir)*xp_l + cos(upwind_dir)*yp_l + building_cent_y;
        
        int i_l = floor(x_l/UGD->dx);
        int j_l = floor(y_l/UGD->dy);
        
        for(int k=k_start;k<k_end;++k) {
            int cell = i_l + j_l*(nx-1) + k*(ny-1)*(nx-1);
            TGD->iturbflag[cell]=12;
        }
    }

    ///////
    float xp_ref_r, yp_ref_r;
    xp_ref_r=xp_r;
    yp_ref_r=yp_r-3.0*dxy;
    
    float x_ref_r, y_ref_r;
    x_ref_r = cos(upwind_dir)*xp_ref_r - sin(upwind_dir)*yp_ref_r + building_cent_x;
    y_ref_r = sin(upwind_dir)*xp_ref_r + cos(upwind_dir)*yp_ref_r + building_cent_y;

    int i_ref_r = floor(x_ref_r/UGD->dx);
    int j_ref_r = floor(y_ref_r/UGD->dy);
    
    for(int k=k_start;k<k_end;++k) {
        int cell = i_ref_r + j_ref_r*(nx-1) + k*(ny-1)*(nx-1);
        TGD->iturbflag[cell]=12;
    }
    
    ///////
    float xp_ref_l, yp_ref_l;
    xp_ref_l=xp_l;
    yp_ref_l=yp_l+3.0*dxy;
    
    float x_ref_l, y_ref_l;
    x_ref_l = cos(upwind_dir)*xp_ref_l - sin(upwind_dir)*yp_ref_l + building_cent_x;
    y_ref_l = sin(upwind_dir)*xp_ref_l + cos(upwind_dir)*yp_ref_l + building_cent_y;

    int i_ref_l = floor(x_ref_l/UGD->dx);
    int j_ref_l = floor(y_ref_l/UGD->dy);
    
    for(int k=k_start;k<k_end;++k) {
        int cell = i_ref_l + j_ref_l*(nx-1) + k*(ny-1)*(nx-1);
        TGD->iturbflag[cell]=12;
    }
    
    float z_build;                  // z value of each building point from its base height
    int k_bottom, k_top;
    int kk;

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
        
        float U_ref_r,U_ref_l;
        int id_face; 
        
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
            float xp,yp;
            
            xp=xp_d+dxy+x_ds*0.5*dxy;
            yp=0.0;
            
            int i = (cos(upwind_dir)*xp - sin(upwind_dir)*yp + building_cent_x)/UGD->dx;
            int j = (sin(upwind_dir)*xp + cos(upwind_dir)*yp + building_cent_y)/UGD->dy;
            int cell = i + j*(nx-1) + k*(ny-1)*(nx-1);    
            int id_face = i + j*nx + k*ny*nx;
            float U_a=0,ustar_h=0;
            
            if ( i > UGD->nx-2 && i < 0 && j > UGD->ny-2 && j < 0) {
                break;
            }
            if (TGD->iturbflag[cell] != 0 && TGD->iturbflag[cell] != 10) {
                
                u_h=0.5*(UGD->u[id_face]+UGD->u[id_face+1]); 
                v_h=0.5*(UGD->v[id_face]+UGD->v[id_face+nx]); 
                w_h=0.5*(UGD->w[id_face]+UGD->w[id_face+nx*ny]);
                U_a=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);
                
                ustar_h = max(kvonk*abs(U_ref_l-U_a),kvonk*abs(U_ref_r-U_a));

                TGD->iturbflag[cell]=10;
                
            } 
            
            //left main wake
            for(auto y_sl=0; y_sl < 2*ceil(abs(yp_l)/dxy); ++y_sl) {
                yp= y_sl*0.5*dxy;
                
                int i = (cos(upwind_dir)*xp - sin(upwind_dir)*yp + building_cent_x)/UGD->dx;
                int j = (sin(upwind_dir)*xp + cos(upwind_dir)*yp + building_cent_y)/UGD->dy;
                int cell = i + j*(nx-1) + k*(ny-1)*(nx-1);    
                
                if ( i > UGD->nx-2 && i < 0 && j > UGD->ny-2 && j < 0) { 
                    break;
                }
                if (TGD->iturbflag[cell] != 0 && TGD->iturbflag[cell] != 10) {
                    TGD->iturbflag[cell]=10;
                }
            }

            //right main wake
            for(auto y_sr=0; y_sr < 2*ceil(abs(yp_r)/dxy); ++y_sr) {
                yp= -y_sr*0.5*dxy;
                
                int i = (cos(upwind_dir)*xp - sin(upwind_dir)*yp + building_cent_x)/UGD->dx;
                int j = (sin(upwind_dir)*xp + cos(upwind_dir)*yp + building_cent_y)/UGD->dy;
                int cell = i + j*(nx-1) + k*(ny-1)*(nx-1);    
                
                if ( i > UGD->nx-2 && i < 0 && j > UGD->ny-2 && j < 0) {
                    break;
                }
                if (TGD->iturbflag[cell] != 0 && TGD->iturbflag[cell] != 10) {
                    TGD->iturbflag[cell]=10;
                }
            }
            
        }
        
        float xp,yp;
            
        xp=xp_d+dxy;
        yp=0.0;
        
        int i = (cos(upwind_dir)*xp - sin(upwind_dir)*yp + building_cent_x)/UGD->dx;
        int j = (sin(upwind_dir)*xp + cos(upwind_dir)*yp + building_cent_y)/UGD->dy;
        id_face = i + j*nx + k*ny*nx;
        float U_a=0,ustar_h=0;
        
        u_h=0.5*(UGD->u[id_face]+UGD->u[id_face+1]); 
        v_h=0.5*(UGD->v[id_face]+UGD->v[id_face+nx]); 
        w_h=0.5*(UGD->w[id_face]+UGD->w[id_face+nx*ny]);
        U_a=sqrt(u_h*u_h + v_h*v_h + w_h*w_h);
        ustar_h = max(kvonk*abs(U_ref_l-U_a),kvonk*abs(U_ref_r-U_a));
        
        for (auto x_ds=0; x_ds <= 2*ceil(abs(xp_d-xp_l)/dxy); x_ds++) {    
            float xp,yp;
            
            xp=xp_l+x_ds*0.5*dxy;
            
            //left side wake
            for(auto y_sl=0; y_sl < 2*ceil(abs(yp_l)/dxy); ++y_sl) {
                yp= y_sl*0.5*dxy;
                
                int i = (cos(upwind_dir)*xp - sin(upwind_dir)*yp + building_cent_x)/UGD->dx;
                int j = (sin(upwind_dir)*xp + cos(upwind_dir)*yp + building_cent_y)/UGD->dy;
                int cell = i + j*(nx-1) + k*(ny-1)*(nx-1);    
                
                if ( i > UGD->nx-2 && i < 0 && j > UGD->ny-2 && j < 0) {
                    break;
                }
                if (TGD->iturbflag[cell] != 0 && TGD->iturbflag[cell] != 10) {
                    TGD->iturbflag[cell]=10;
                } 
            }
        }
        for (auto x_ds=0; x_ds <= 2*ceil(abs(xp_d-xp_r)/dxy); x_ds++) {    
            float xp,yp;
            
            xp=xp_r+x_ds*0.5*dxy;
            
            //right side wake
            for(auto y_sr=0; y_sr < 2*ceil(abs(yp_r)/dxy); ++y_sr) {
                yp=-y_sr*0.5*dxy;
                
                int i = (cos(upwind_dir)*xp - sin(upwind_dir)*yp + building_cent_x)/UGD->dx;
                int j = (sin(upwind_dir)*xp + cos(upwind_dir)*yp + building_cent_y)/UGD->dy;
                int cell = i + j*(nx-1) + k*(ny-1)*(nx-1);    
                
                if ( i > UGD->nx-2 && i < 0 && j > UGD->ny-2 && j < 0) {
                    break;
                }
                if (TGD->iturbflag[cell] != 0 && TGD->iturbflag[cell] != 10) {
                    TGD->iturbflag[cell]=10;
                }
            }   
        }
    }   
    
    return;

    std::vector<float> Lr_face, Lr_node;
    std::vector<int> perpendicular_flag;
    Lr_face.resize (polygonVertices.size(), -1.0);       // Length of wake for each face
    Lr_node.resize (polygonVertices.size(), 0.0);       // Length of wake for each node
    perpendicular_flag.resize (polygonVertices.size(), 0);
    upwind_rel_dir.resize (polygonVertices.size(), 0.0);      // Upwind reletive direction for each face
    
    float yc, xc;
    float Lr_local, Lr_local_u, Lr_local_v, Lr_local_w;   // Local length of the wake for each velocity component
    float x_wall, x_wall_u, x_wall_v, x_wall_w;
    float y_norm, canyon_factor;
    int x_id_min;

    float Lr_ave;                         // Average length of Lr
    float total_seg_length;               // Length of each edge
    int index_previous, index_next;       // Indices of previous and next nodes
    int stop_id = 0;
    
    float tol;
    float farwake_exp = 1.5;
    float farwake_factor = 3;
    float epsilon = 10e-10;
    int u_wake_flag, v_wake_flag, w_wake_flag;
    int i_u, j_u, i_v, j_v, i_w, j_w;          // i and j indices for x, y and z directions
    float xp, yp;
    float xu, yu, xv, yv, xw, yw;
    float dn_u, dn_v, dn_w;             // Length of cavity zone
    float farwake_vel;
    std::vector<double> u_temp, v_temp;
    u_temp.resize (UGD->nx*UGD->ny, 0.0);
    v_temp.resize (UGD->nx*UGD->ny, 0.0);
    std::vector<double> u0_modified, v0_modified;
    std::vector<int> u0_mod_id, v0_mod_id;
    float R_scale, R_cx, vd, hd, shell_height;
    

    for (auto id=0; id<polygonVertices.size()-1; id++)
    {
        // Finding faces that are eligible for applying the far-wake parameterizations
        // angle between two points should be in -180 to 0 degree
        if ( abs(upwind_rel_dir[id]) < 0.5*M_PI)
        {
            // Calculate length of the far wake zone for each face
            Lr_face[id] = Lr*cos(upwind_rel_dir[id]);
        }
    }

    Lr_ave = total_seg_length = 0.0;
    // This loop interpolates the value of Lr for eligible faces to nodes of those faces
    for (auto id=0; id<polygonVertices.size()-1; id++)
    {
        // If the face is eligible for parameterization
        if (Lr_face[id] > 0.0)
        {
            index_previous = (id+polygonVertices.size()-2)%(polygonVertices.size()-1);     // Index of previous face
            index_next = (id+1)%(polygonVertices.size()-1);           // Index of next face
            if (Lr_face[index_previous] < 0.0 && Lr_face[index_next] < 0.0)
            {
                Lr_node[id] = Lr_face[id];
                Lr_node[id+1] = Lr_face[id];
            }
            else if (Lr_face[index_previous] < 0.0)
            {
                Lr_node[id] = Lr_face[id];
                Lr_node[id+1] = ((yi[index_next]-yi[index_next+1])*Lr_face[index_next]+(yi[id]-yi[index_next])*Lr_face[id])/(yi[id]-yi[index_next+1]);
            }
            else if (Lr_face[index_next] < 0.0)
            {
                Lr_node[id] = ((yi[id]-yi[index_next])*Lr_face[id]+(yi[index_previous]-yi[id])*Lr_face[index_previous])/(yi[index_previous]-yi[index_next]);
                Lr_node[id+1] = Lr_face[id];
            }
            else
            {
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
    for (auto k = 1; k <= k_start; k++)
    {
        k_bottom = k;
        if (base_height <= UGD->z[k])
        {
            break;
        }
    }

    for (auto k = k_start; k < UGD->nz-1; k++)
    {
        k_top = k;
        if (height_eff < UGD->z[k+1])
        {
            break;
        }
    }

    for (auto k = k_start; k < k_end; k++)
    {
        kk = k;
        if (0.75*H+base_height <= UGD->z[k])
        {
            break;
        }
    }

    for (auto k=k_top; k>=k_bottom; k--)
    {
        z_build = UGD->z[k] - base_height;
        for (auto id=0; id<=stop_id; id++)
        {
            if (abs(upwind_rel_dir[id]) < 0.5*M_PI)
            {
                if (abs(upwind_rel_dir[id]) < tol)
                {
                    perpendicular_flag[id]= 1;
                    x_wall = xi[id];
                }
                for (auto y_id=0; y_id <= 2*ceil(abs(yi[id]-yi[id+1])/UGD->dxy); y_id++)
                {
                    yc = yi[id]-0.5*y_id*UGD->dxy;
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
                    for (auto x_id=1; x_id <= ceil(Lr_local/UGD->dxy); x_id++)
                    {
                        xc = x_id*UGD->dxy;
                        int i = ((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/UGD->dx;
                        int j = ((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/UGD->dy;
                        if ( i >= UGD->nx-2 && i <= 0 && j >= UGD->ny-2 && j <= 0)
                        {
                            break;
                        }
                        int icell_cent = i+j*(UGD->nx-1)+kk*(UGD->nx-1)*(UGD->ny-1);
                        if ( UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2 && x_id_min < 0)
                        {
                            x_id_min = x_id;
                        }
                        if ( (UGD->icellflag[icell_cent] == 0 || UGD->icellflag[icell_cent] == 2) && x_id_min > 0)
                        {
                            canyon_factor = xc/Lr;

                            break;
                        }
                    }
                    x_id_min = -1;
                    for (auto x_id=1; x_id <= 2*ceil(farwake_factor*Lr_local/UGD->dxy); x_id++)
                    {
                        u_wake_flag = 1;
                        v_wake_flag = 1;
                        w_wake_flag = 1;
                        xc = 0.5*x_id*UGD->dxy;
                        int i = ((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/UGD->dx;
                        int j = ((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/UGD->dy;
                        if (i >= UGD->nx-2 && i <= 0 && j >= UGD->ny-2 && j <= 0)
                        {
                            break;
                        }
                        icell_cent = i+j*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
                        if (UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2 && x_id_min < 0)
                        {
                            x_id_min = x_id;
                        }
                        if (UGD->icellflag[icell_cent] == 0 || UGD->icellflag[icell_cent] == 2)
                        {
                            if (x_id_min >= 0)
                            {
                                if (UGD->ibuilding_flag[icell_cent] == building_id)
                                {
                                    x_id_min = -1;
                                }
                                else if (canyon_factor < 1.0)
                                {
                                    break;
                                }
                                else if (UGD->icellflag[i+j*(UGD->nx-1)+kk*(UGD->nx-1)*(UGD->ny-1)] == 0 || 
                                         UGD->icellflag[i+j*(UGD->nx-1)+kk*(UGD->nx-1)*(UGD->ny-1)] == 2)
                                {
                                    break;
                                }

                            }
                        }

                        if (UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
                        {
                            i_u = std::round(((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/UGD->dx);
                            j_u = ((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/UGD->dy;
                            if (i_u < UGD->nx-1 && i_u > 0 && j_u < UGD->ny-1 && j_u > 0)
                            {
                                xp = i_u*UGD->dx-building_cent_x;
                                yp = (j_u+0.5)*UGD->dy-building_cent_y;
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
                                icell_cent = i_u + j_u*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
                                icell_face = i_u + j_u*UGD->nx+k*UGD->nx*UGD->ny;
                                if (dn_u > 0.0 && u_wake_flag == 1 && yu <= yi[id] && yu >= yi[id+1] && 
                                    UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
                                {
                                    // Far wake zone
                                    if (xu > dn_u)
                                    {
                                        farwake_vel = UGD->u0[icell_face]*(1.0-pow((dn_u/(xu+UGD->wake_factor*dn_u)),farwake_exp));
                                        if (canyon_factor == 1.0)
                                        {
                                            
                                        }
                                    }
                                    // Cavity zone
                                    else
                                    {
                                        
                                    }
                                }
                            }

                            i_v = ((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/UGD->dx;
                            j_v = std::round(((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/UGD->dy);
                            if (i_v<UGD->nx-1 && i_v>0 && j_v<UGD->ny-1 && j_v>0)
                            {
                                xp = (i_v+0.5)*UGD->dx-building_cent_x;
                                yp = j_v*UGD->dy-building_cent_y;
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
                                    //dn_v = Lr;
                                }
                                else
                                {
                                    dn_v = 0.0;
                                }
                                if (xv > farwake_factor*dn_v)
                                {
                                    v_wake_flag = 0;
                                }
                                icell_cent = i_v + j_v*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
                                icell_face = i_v + j_v*UGD->nx+k*UGD->nx*UGD->ny;
                                if (dn_v > 0.0 && v_wake_flag == 1 && yv <= yi[id] && yv >= yi[id+1] && 
                                    UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
                                {
                                    // Far wake zone
                                    /*if (xv > dn_v)
                                    {
                                        farwake_vel = UGD->v0[icell_face]*(1.0-pow((dn_v/(xv+UGD->wake_factor*dn_v)),farwake_exp));
                                        if (canyon_factor == 1)
                                        {
                                            //v0_modified.push_back(farwake_vel);
                                            //v0_mod_id.push_back(icell_face);
                                            //UGD->w0[i+j*UGD->nx+k*UGD->nx*UGD->ny] = 0.0;
                                        }
                                    }
                                    // Cavity zone
                                    else
                                    {
                                        
                                    }*/
                                }
                            }

                            i_w = ceil(((xc+x_wall)*cos(upwind_dir)-yc*sin(upwind_dir)+building_cent_x)/UGD->dx)-1;
                            j_w = ceil(((xc+x_wall)*sin(upwind_dir)+yc*cos(upwind_dir)+building_cent_y)/UGD->dy)-1;
                            if (i_w<UGD->nx-2 && i_w>0 && j_w<UGD->ny-2 && j_w>0)
                            {
                                xp = (i_w+0.5)*UGD->dx-building_cent_x;
                                yp = (j_w+0.5)*UGD->dy-building_cent_y;
                                xw = xp*cos(upwind_dir)+yp*sin(upwind_dir);
                                yw = -xp*sin(upwind_dir)+yp*cos(upwind_dir);
                                Lr_local_w = Lr_node[id]+(yw-yi[id])*(Lr_node[id+1]-Lr_node[id])/(yi[id+1]-yi[id]);
                                if (perpendicular_flag[id] > 0)
                                {
                                    x_wall_w = xi[id];
                                }
                                else
                                {
                                    x_wall_w = ((xi[id+1]-xi[id])/(yi[id+1]-yi[id]))*(yw-yi[id]) + xi[id];
                                }
                                xw -= x_wall_w;
                                if (abs(yw) < abs(y_norm) && abs(y_norm) > epsilon && z_build < height_eff && height_eff > epsilon)
                                {
                                    dn_w = sqrt((1.0-pow(yw/y_norm, 2.0))*(1.0-pow(z_build/height_eff,2.0))*pow(canyon_factor*Lr_local_w,2.0));
                                    //dn_w = Lr;
                                }
                                else
                                {
                                    dn_w = 0.0;
                                }

                                if (xw > farwake_factor*dn_w)
                                {
                                    w_wake_flag = 0;
                                }
                                icell_cent = i_w + j_w*(UGD->nx-1)+k*(UGD->nx-1)*(UGD->ny-1);
                                icell_face = i_w + j_w*UGD->nx+k*UGD->nx*UGD->ny;
                                if (dn_w > 0.0 && w_wake_flag == 1 && yw <= yi[id] && yw >= yi[id+1] && 
                                    UGD->icellflag[icell_cent] != 0 && UGD->icellflag[icell_cent] != 2)
                                {
                                    //TGD->iturbflag[icell_cent] = 10;
                                }
                                if (u_wake_flag == 0 && v_wake_flag == 0 && w_wake_flag == 0)
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    
    return;
    
}
