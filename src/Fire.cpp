/**
*  Fire.cpp
*  
*  This class models fire spread rate in QUIC using Balbi (2019), Baum & McCaffrey (1988), level set: Sethian
*
*  Created by Matthew Moody, Jeremy Gibbs on 12/27/18.
*/

#include "Fire.hpp"

using namespace std;


float PI = 3.14159265359;
Fire :: Fire(URBInputData* UID, URBGeneralData* UGD, Output* output) {
    
    // get domain information
    nx = UGD->nx;
    ny = UGD->ny;
    nz = UGD->nz;
    
    dx = UGD->dx;
    dy = UGD->dy;
    dz = UGD->dz;

    // set-up the mapper array - cell centered
    fire_cells.resize((nx-1)*(ny-1));
    burn_flag.resize((nx-1)*(ny-1));
    burn_out.resize((nx-1)*(ny-1));
    front_map.resize((nx-1)*(ny-1));
    del_plus.resize((nx-1)*(ny-1));
    del_min.resize((nx-1)*(ny-1));
    xNorm.resize((nx-1)*(ny-1));
    yNorm.resize((nx-1)*(ny-1));
    Force.resize((nx-1)*(ny-1));

    // set-up potential field array - cell centered
    Pot_u.resize((nx-1)*(ny-1)*(nz-2));
    Pot_v.resize((nx-1)*(ny-1)*(nz-2));
    Pot_w.resize((nx-1)*(ny-1)*(nz-2));


    /**
     * Set initial fire info
     */
    x_start    = UID->fires->xStart;
    y_start    = UID->fires->yStart;
    H          = UID->fires->height;
    L          = UID->fires->length;
    W          = UID->fires->width;
    baseHeight = UID->fires->baseHeight;
    fuel_type  = UID->fires->fuelType;
    courant    = UID->fires->courant;
    
    // Set fuel properties for domain
    for (int j = 0; j < ny-1; j++){
        for (int i = 0; i < nx-1; i++){
            
	  int idx = i + j*(nx-1);
            
            if (fuel_type==1)  fire_cells[idx].fuel = new ShortGrass();
            if (fuel_type==2)  fire_cells[idx].fuel = new TimberGrass();
            if (fuel_type==3)  fire_cells[idx].fuel = new TallGrass();
            if (fuel_type==4)  fire_cells[idx].fuel = new Chaparral();
            if (fuel_type==5)  fire_cells[idx].fuel = new Brush();
            if (fuel_type==6)  fire_cells[idx].fuel = new DormantBrush();
            if (fuel_type==7)  fire_cells[idx].fuel = new SouthernRough();
            if (fuel_type==8)  fire_cells[idx].fuel = new TimberClosedLitter();
            if (fuel_type==9)  fire_cells[idx].fuel = new HarwoodLitter();
            if (fuel_type==10) fire_cells[idx].fuel = new TimberLitter();
            if (fuel_type==11) fire_cells[idx].fuel = new LoggingSlashLight();
            if (fuel_type==12) fire_cells[idx].fuel = new LoggingSlashMedium();
            if (fuel_type==13) fire_cells[idx].fuel = new LoggingSlashHeavy();
        }
    }
    
    // get grid info of fire
    i_start = std::round(x_start/dx);       
    i_end   = std::round((x_start+L)/dx);     
    j_start = std::round(y_start/dy);       
    j_end   = std::round((y_start+W)/dy);      
    k_start = std::round((H+baseHeight)/dz);                
    k_end   = std::round((H+baseHeight)/dz)+1;
    
    /**
     * Set-up initial fire state
     */
    for (int j = j_start; j < j_end; j++){
        for (int i = i_start; i < i_end; i++){
	  int idx = i + j*(nx-1);
	    	fire_cells[idx].state.burn_flag = 1;
            fire_cells[idx].state.front_flag = 1;
        }
    }
    
    /** 
     *  Set up burn flag field
     */
    for (int j = 0; j < ny-1; j++){
        for (int i = 0; i < nx-1; i++){
	  int idx = i + j*(nx-1);       
            burn_flag[idx] = fire_cells[idx].state.burn_flag;
        }
    }
    
    /**
     * Set up initial level set. Use signed distance function: swap to fast marching method in future.
     */
    float sdf, sdf_min;
    for (int j = 0; j < ny-1; j++){
        for (int i = 0; i < nx-1; i++){
	  int idx = i + j*(nx-1);
            if (fire_cells[idx].state.front_flag == 1){
                front_map[idx] = 0;                
            }
            else {
                sdf = 1000;
                for (int jj = 0; jj < ny-1; jj++){
                    for (int ii = 0; ii < nx-1; ii++){
		      int idx2 = ii + jj*(nx-1);
                        if (fire_cells[idx2].state.front_flag == 1){
                            sdf_min = sqrt ((ii-i)*(ii-i) + (jj-j)*(jj-j));
                        }
                        else{
                            sdf_min = 1000;
                        }
                        
                        sdf = sdf_min < sdf ? sdf_min : sdf;
                    }
                }
                front_map[idx] = sdf;                
            }
        }
    }

    // set output fields
    output_fields = UID->fileOptions->outputFields;
    
    if (output_fields[0]=="all") {
        output_fields.erase(output_fields.begin());
        output_fields = {"time","burn"/*,"ros"*/};
    }
    
    // set cell-centered dimensions
    const std::string tname = "t";
    NcDim t_dim = output->getDimension(tname);
    NcDim z_dim = output->getDimension("z");
    NcDim y_dim = output->getDimension("y");
    NcDim x_dim = output->getDimension("x");

    
    dim_scalar_1.push_back(t_dim);
    
    dim_scalar_3.push_back(t_dim);
    dim_scalar_3.push_back(y_dim);
    dim_scalar_3.push_back(x_dim);

    dim_scalar_4.push_back(t_dim);
    dim_scalar_4.push_back(z_dim);
    dim_scalar_4.push_back(y_dim);
    dim_scalar_4.push_back(x_dim);


    // create attributes
    AttScalarDbl att_t = {&time,      "time", "time[s]",         "--", dim_scalar_1};
    AttVectorDbl att_b = {&burn_out, "burn", "burn flag value", "--", dim_scalar_3};
    //AttVectorDbl att_w = {&Pot_w, "wp", "potential w", "--", dim_scalar_4};



    // map the name to attributes
    map_att_scalar_dbl.emplace("time", att_t);
    map_att_vector_dbl.emplace("burn", att_b);
    //map_att_vector_dbl.emplace("wp", att_w);
    
    // create list of fields to save
    for (int i=0; i<output_fields.size(); i++) {
        std::string key = output_fields[i];
        if (map_att_scalar_dbl.count(key)) {
            output_scalar_dbl.push_back(map_att_scalar_dbl[key]);
        } else if (map_att_vector_dbl.count(key)) {
            output_vector_dbl.push_back(map_att_vector_dbl[key]);
        } else if(map_att_vector_int.count(key)) {
            output_vector_int.push_back(map_att_vector_int[key]);
        }
    }
    
    // add scalar double fields
    for (int i=0; i<output_scalar_dbl.size(); i++) {
        AttScalarDbl att = output_scalar_dbl[i];
        output->addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
    }
    // add vector double fields
    for (int i=0; i<output_vector_dbl.size(); i++) {
        AttVectorDbl att = output_vector_dbl[i];
        output->addField(att.name, att.units, att.long_name, att.dimensions, ncDouble);
    }
    // add vector int fields
    for (int i=0; i<output_vector_int.size(); i++) {
        AttVectorInt att = output_vector_int[i];
        output->addField(att.name, att.units, att.long_name, att.dimensions, ncInt);
    }    
}

/**
 * Compute adaptive time step. Based on Courant criteria.
 */
float Fire :: computeTimeStep() {
    
    // spread rates
    float r, r_max;
    
    // get max spread rate
    for (int j = 0; j < ny-1; j++){
        for (int i = 0; i < nx-1; i++){
	  int idx = i + j*(nx-1);
            r = fire_cells[idx].properties.r;       
            r_max   = r > r_max ? r : r_max;
        }
    }
    return courant * dx / r_max;
}

/**
 * Compute fire spread for burning cells
 */
void Fire :: run(Solver* solver, URBGeneralData* UGD) {


    /**
     * Calculate level set gradient and norm (Chapter 6, Sethian 2008)
     */
    float dmx, dpx, dmy, dpy, n_star_x, n_star_y;
    for (int j = 1; j < ny-2; j++){
        for (int i = 1; i < nx-2; i++){
	  int idx = i + j*(nx-1);       
	  int idxjp = i + (j+1)*(nx-1);
	  int idxjm = i + (j-1)*(nx-1);
            dmy = (front_map[idx] - front_map[idxjm])/dx;
            dpy = (front_map[idxjp] - front_map[idx])/dx;
            dmx = (front_map[idx] - front_map[idx-1])/dy;
            dpx = (front_map[idx+1] - front_map[idx])/dy;
            del_plus[idx] = sqrt(fmax(dmx,0)*fmax(dmx,0) + fmin(dpx,0)*fmin(dpx,0) + fmax(dmy,0)*fmax(dmy,0) + fmin(dpy,0)*fmin(dpy,0));
            del_min[idx] = sqrt(fmax(dpx,0)*fmax(dpx,0) + fmin(dmx,0)*fmin(dmx,0) + fmax(dpy,0)*fmax(dpy,0) + fmin(dmy,0)*fmin(dmy,0));
            n_star_x = dpx/sqrt(dpx*dpx + dpy*dpy) + dmx/sqrt(dmx*dmx + dpy*dpy) + dpx/sqrt(dpx*dpx + dmy*dmy) + dmx/sqrt(dmx*dmx + dmy*dmy);
            n_star_y = dpy/sqrt(dpx*dpx + dpy*dpy) + dpy/sqrt(dmx*dmx + dpy*dpy) + dmy/sqrt(dpx*dpx + dmy*dmy) + dmy/sqrt(dmx*dmx + dmy*dmy);
            xNorm[idx] = n_star_x/sqrt(n_star_x*n_star_x + n_star_y*n_star_y);
            yNorm[idx] = n_star_y/sqrt(n_star_x*n_star_x + n_star_y*n_star_y);
        }
    }

    /**
     * Calculate Forcing Function (Balbi model at mid-flame height or first grid cell if no fire)
     */
    for (int j=0; j < ny-1; j++){
        for (int i=0; i < nx-1; i++){
	    int idx = i + j*(nx-1);
            // get fuel properties at this location
            struct FuelProperties* fuel = fire_cells[idx].fuel;
            // calculate mid-flame height
            int kh   = 0;
            float H = fire_cells[idx].properties.h;
            float T = UGD->terrain[idx];
            float D = fuel->fueldepthm;
            float FD = H + T + D;
        
            if (H==0) {
                kh = 1;
            } else {
                kh = std::round(FD/dz);
            }

            // call u and v from URB General Data
	    int cell_face = i + j*nx + kh*nx*ny;
            float u = 0.5*(UGD->u[cell_face] + UGD->u[cell_face+1]);
            float v = 0.5*(UGD->v[cell_face] + UGD->v[cell_face+nx]);
            // call norm for cells
            float x_norm = xNorm[idx];
            float y_norm = yNorm[idx];
            // run Balbi model
            struct FireProperties fp = balbi(fuel,u,v,x_norm,y_norm,0.0,0.0650);
            fire_cells[idx].properties = fp;
            Force[idx] = fp.r;
        }
    }
    // compute time step
    dt = computeTimeStep();
    
    // indices for burning cells
    std::vector<int> cells_burning;
    
    // search predicate for burn state
    struct find_burn : std::unary_function<FireCell, bool> {
        float burn;
        find_burn(int burn):burn(burn) { }
        bool operator()(FireCell const& f) const {
            return f.state.burn_flag == burn;
        }
    };


    // get indices of burning cells
    std::vector<FireCell>::iterator it = std::find_if(fire_cells.begin(),fire_cells.end(),find_burn(1));
    while ( it != fire_cells.end()) {
        
        if (it!=fire_cells.end()) {
            cells_burning.push_back(std::distance(fire_cells.begin(), it));
        }
        
        it = std::find_if (++it, fire_cells.end(),find_burn(1)); 
    }
    
    // loop through burning cells
    for (int i=0; i<cells_burning.size();i++) {
        
        // get index burning cell
        int id = cells_burning[i];
        
        // get fuel properties at this location
        struct FuelProperties* fuel = fire_cells[id].fuel;
        
        // get vertical index of mid-flame height
        int kh   = 0;                      ///< mid-flame height
	int maxkh = 0;                     ///< max flame height
        //modify flame height by time on fire (assume linear functionality)
        float H = fire_cells[id].properties.h*(1-(fire_cells[id].state.burn_time/fire_cells[id].properties.tau));
	float maxH = fire_cells[id].properties.h;       
	float T = UGD->terrain[id];
        float D = fuel->fueldepthm;
        
        float FD = H/2.0 + T + D;
	float MFD = maxH + T + D;
       

        if (H==0) {
            kh = 1;
        } else {
            kh = std::round(FD/dz);
        }
	if (maxH==0) {
	  maxkh = 1;
	} else {
	  maxkh = std::round(MFD/dz);
	}
                
        // convert flat index to i, j at cell center
        int ii  = id % (nx-1);
        int jj  = (id / (nx-1)) % (ny-1);
                        
        // get horizontal wind at flame height
	int cell_face = ii + jj*nx + kh*ny*nx;
        float u = 0.5*(UGD->u[cell_face] + UGD->u[cell_face+1]);
        float v = 0.5*(UGD->v[cell_face] + UGD->v[cell_face+nx]);
        
        // run Balbi model
        float x_norm = xNorm[id];
        float y_norm = yNorm[id];
        float burnTime = fire_cells[id].state.burn_time;
        struct FireProperties fp = balbi(fuel,u,v,x_norm,y_norm,0.0,0.0650);
        fire_cells[id].properties = fp;
        
	// update icell value for flame
	for (int k=1; k<= maxkh; k++){
	  int icell_cent = ii + jj*(nx-1) + (k-1)*(nx-1)*(ny-1);
          if (k <= 2*kh){
	    UGD->icellflag[icell_cent] = 12;
	  } else {
	    UGD->icellflag[icell_cent] = 1;
	  }
	}




        // modify w0 in solver (adjust to use faces)- bottom cell only - Balbi velocity
        /*
	for (int k=0; k<1; k++) {
            
	  int idf  = ii + (jj)*(nx) + (k+1)*(nx)*(ny);
            int idxF = idf+1;
            int idxB = idf-1;
            int idyF = idf+(nx);
            int idyB = idf-(nx);
            
            double K = fire_cells[id].properties.K;            
            double u = UGD->u0[idf];
            double v = UGD->v0[idf];
            double u_uw, v_uw;
              
            UGD->w0[idf] = fp.w;
	    
        }
	*/
   
    }



    
}

/**
 * Calcluate potential field
 */

void Fire :: potential(URBGeneralData* UGD){
    /**
     * Read netCDF file for Potential Field from heat release
     **/
    // Open netCDF for Potential Field as read only
    NcFile Potential("../data/HeatPot.nc", NcFile::read);

    // Get size of netCDF data
    int pot_z = Potential.getVar("u_r").getDim(0).getSize();
    int pot_r = Potential.getVar("u_r").getDim(1).getSize();
    int pot_G = Potential.getVar("G").getDim(0).getSize();
    int pot_rStar = Potential.getVar("rStar").getDim(0).getSize();
    int pot_zStar = Potential.getVar("zStar").getDim(0).getSize();
    // Allocate variable arrays
    std::vector<float> u_r(pot_z * pot_r);
    std::vector<float> u_z(pot_z * pot_r);
    std::vector<float> G(pot_G);
    std::vector<float> Gprime(pot_G);
    std::vector<float> rStar(pot_rStar);
    std::vector<float> zStar(pot_zStar);
    // Read start index and length to read
    std::vector<size_t> startIdxField = {0,0};
    std::vector<size_t> countsField = {static_cast<unsigned long>(pot_z),
                                       static_cast<unsigned long>(pot_r)};


    // Get variables from netCDF file
    Potential.getVar("u_r").getVar(startIdxField, countsField, u_r.data());
    Potential.getVar("u_z").getVar(startIdxField, countsField, u_z.data());

    Potential.getVar("G").getVar({0}, {pot_G}, G.data());
    Potential.getVar("Gprime").getVar({0}, {pot_G}, Gprime.data());
    Potential.getVar("rStar").getVar({0}, {pot_rStar}, rStar.data());
    Potential.getVar("zStar").getVar({0}, {pot_zStar}, zStar.data());

    // dr and dz, assume linear spacing between
    float drStar = rStar[1]-rStar[0];
    float dzStar = zStar[1]-zStar[0];
    
    // reset potential fields
    std::fill (Pot_u.begin(),Pot_u.end(),0);
    std::fill (Pot_v.begin(),Pot_v.end(),0);
    std::fill (Pot_w.begin(),Pot_w.end(),0);
	/**
	 * Calculate Potential field based on heat release
	 * Baum and McCaffrey plume model
	**/

    for (int ii = 0; ii < nx-1; ii++){
	for (int jj = 0; jj < ny-1; jj++){
  	int id = ii + jj*(nx-1);
	if (burn_flag[id] == 1){
	struct FireProperties fp = fire_cells[id].properties;
	// Loop through horizontal domain
	float ur, uz; 
	float u_p; 	 					///< u velocity from potential field in target cell
	float v_p; 						///< v velocity from potential field in target cell
	float w_p;						///< w velocity from potential field in target cell

	for (int ipot = 0; ipot<nx-1; ipot++) {
	  for (int jpot = 0; jpot<ny-1; jpot++){
	    float deltaX = (ipot-ii)*dx/fp.L_c;                      ///< distance between fire cell and target cell k in x direction
	    float deltaY = (jpot-jj)*dy/fp.L_c;                      ///< distance between fire cell and target cell k in y direction
	    float h_k = sqrt(deltaX*deltaX + deltaY*deltaY);         ///< radial distance from fire cell and target cell k in horizontal
	    // Loop through vertical cells
	    for (int kpot = 1; kpot<nz-2; kpot++) {
	      float z_k = (kpot)*dz/fp.L_c;                            ///< vertical distance between fire cell and target cell k
	      // if radius = 0
	      if (h_k < 0.00001){

		float zMinIdx = floor(z_k/dzStar);
		float zMaxIdx = ceil(z_k/dzStar);

		ur = 0.5*(u_r[zMinIdx*pot_r]+u_r[zMaxIdx*pot_r]);
		uz = 0.5*(u_z[zMinIdx*pot_r]+u_z[zMaxIdx*pot_r]);
		u_p = fp.U_c*ur;                         
		v_p = fp.U_c*ur;                         
		w_p = fp.U_c*uz;                         
	      }
	      // if in potential field lookup, r*(h_k) < 30 and z*(z_k) < 60
	      else if (z_k < 60 && h_k < 30){ 
		// indices for lookup
		float rMinIdx = floor(h_k/drStar);
		float rMaxIdx = ceil(h_k/drStar);
		
		float zMinIdx = floor(z_k/dzStar);
		float zMaxIdx = ceil(z_k/dzStar);
		
		//ur = 0.25*(u_r[rMinIdx+zMinIdx*pot_r]+u_r[rMinIdx+zMaxIdx*pot_r]+u_r[rMaxIdx+zMinIdx*pot_r]+u_r[rMaxIdx+zMaxIdx*pot_r]); //lookup from u_r, linear interpolate between values

		ur = 0.5*(u_r[rMinIdx + zMinIdx*pot_r]+u_r[rMaxIdx + zMinIdx*pot_r]);

		//uz = 0.25*(u_z[rMinIdx+zMinIdx*pot_r]+u_z[rMinIdx+zMaxIdx*pot_r]+u_z[rMaxIdx+zMinIdx*pot_r]+u_z[rMaxIdx+zMaxIdx*pot_r]); //lookup from u_z, linear interpolate between values

		uz = 0.5*(u_z[rMinIdx + zMinIdx*pot_r]+u_z[rMinIdx + zMaxIdx*pot_r]);
		
		u_p = fp.U_c*ur*deltaX/h_k;          
		v_p = fp.U_c*ur*deltaY/h_k;          
		w_p = fp.U_c*uz;                         
		
	      }
	      // if outside potential field lookup use asymptotic functions for potential field
	      else {
		float zeta = sqrt(h_k*h_k + z_k*z_k);
		float x1 = (1+cos(atan(h_k/z_k)))/2.0;
		// lookup indices for G(x) and G'(x) - spans 0.5 to 1.0
		int gMinIdx = floor(pot_G*(x1-.5)/.5);
		int gMaxIdx = ceil(pot_G*(x1-.5)/.5);
		// values for G and G'
                float g_x = 0.5*(G[gMinIdx]+G[gMaxIdx]);
		float gprime_x = 0.5*(G[gMinIdx]+G[gMaxIdx]);

		ur = h_k/(2*PI*pow(zeta,(3/2.0))) + pow(zeta,(-1/3.0))*((5/6.0)*(1-2*x1)/sqrt(x1*(1-x1))*g_x - sqrt(x1*(1-x1))*gprime_x);
		uz = z_k/(2*PI*pow(zeta,(3/2.0))) + pow(zeta,(-1/3.0))*((5/3.0)*g_x + (1-2*x1)/2.0*gprime_x);
		u_p = fp.U_c*ur*deltaX/h_k;            
		v_p = fp.U_c*ur*deltaY/h_k;            
		w_p = fp.U_c*uz;
		                      
	      }		
	      
	      // modify potential fields
	      
	      int cellCentPot = ipot + jpot*(nx-1) + (kpot-1)*(nx-1)*(ny-1);
	      Pot_u[cellCentPot] += u_p;
	      Pot_v[cellCentPot] += v_p;
	      Pot_w[cellCentPot] += w_p;
	    }
	  }
	}
    }
}
}

    // Modify u,v,w in solver - superimpose Potential field onto velocity field (interpolate from potential cell centered values)
    for (int iadd=1; iadd<nx-1; iadd++){
      for (int jadd=1; jadd<ny-1; jadd++){
	for (int kadd=1; kadd<nz-2; kadd++){
	  int cell_face = iadd + jadd*nx + (kadd-1)*nx*ny;
	  int cell_cent = iadd + jadd*(nx-1) + (kadd-1)*(nx-1)*(ny-1);
	  UGD->u0[cell_face] = UGD->u0[cell_face] + 0.5*(Pot_u[cell_cent]+Pot_u[cell_cent+1]);
	  UGD->v0[cell_face] = UGD->v0[cell_face] + 0.5*(Pot_v[cell_cent]+Pot_v[cell_cent+(nx-1)]);
	  UGD->w0[cell_face] = UGD->w0[cell_face] + Pot_w[cell_cent];
	  
	}
      }
    }
}

/** 
 * Compute fire spread. Advance level set.
 */
void Fire :: move(Solver* solver, URBGeneralData* UGD){
    for (int j=1; j < ny-2; j++){
        for (int i=1; i < nx-2; i++){
	  int idx = i + j*(nx-1);
	  
            // get fire properties at this location
            struct FireProperties fp = fire_cells[idx].properties; 
	  
            // advance level set
            front_map[idx] = front_map[idx] - dt*(fmax(Force[idx],0)*del_plus[idx] + fmin(Force[idx],0)*del_min[idx]);
            // if level set <= 1, set burn_flag to 0.5 - L.S. for preheating
            if (front_map[idx] <= 1 && burn_flag[idx] < 1){
                fire_cells[idx].state.burn_flag = 0.5;
            }
            // if level set < threshold, set burn flag to 1
            if (front_map[idx] <= 0.01 && burn_flag[idx] < 1){
                fire_cells[idx].state.burn_flag = 1;
            }
            // if burn flag = 1, update burn time
            if (burn_flag[idx] == 1){
                fire_cells[idx].state.burn_time += dt;
            }
            // set burn flag to 2 (burned) if residence time exceeded and update z0 to bare soil
            if (fire_cells[idx].state.burn_time >= fp.tau) {
                fire_cells[idx].state.burn_flag = 2;
		UGD->z0_domain[idx] = 0.01;
            }
            // update burn flag field
            burn_flag[idx] = fire_cells[idx].state.burn_flag;
	    burn_out[idx] = burn_flag[idx];	    
	    
        }
    }
}


// Rothermel (1972) flame propgation model used for initial guess to Balbi
float Fire :: rothermel(FuelProperties* fuel, float max_wind, float tanphi, float fmc_g) {
    
    // fuel properties
    int savr          = fuel->savr;
    int fueldens      = fuel->fueldens;
    float st         = fuel->st;
    float se         = fuel->se;
    float fgi        = fuel->fgi;        
    float fuelmce    = fuel->fuelmce;
    float fuelheat   = fuel->fuelheat;
    float fueldepthm = fuel->fueldepthm;
    
    // local fire variables
    float bmst      = fmc_g/(1+fmc_g);
    float fuelloadm = (1.-bmst)*fgi;
    float fuelload  = fuelloadm*(pow(.3048, 2.0))*2.205;               // convert fuel load to lb/ft^2
    float fueldepth = fueldepthm/0.3048;                               // to ft
    float betafl    = fuelload/(fueldepth * fueldens);                 // packing ratio  jm: lb/ft^2/(ft * lb*ft^3) = 1
    float betaop    = 3.348 * pow(savr, -0.8189);                      // optimum packing ratio jm: units?? 
    float qig       = 250. + 1116.*fmc_g;                              // heat of preignition, btu/lb
    float epsilon   = exp(-138./savr );                                // effective heating number
    float rhob      = fuelload/fueldepth;                              // ovendry bulk density, lb/ft^3
    float rtemp2    = pow(savr, 1.5);
    float gammax    = rtemp2/(495. + 0.0594*rtemp2);                   // maximum rxn vel, 1/min
    float ar        = 1./(4.774 * (pow(savr, 0.1)) - 7.27);            // coef for optimum rxn vel
    float ratio     = betafl/betaop;   
    float gamma     = gammax*((pow(ratio, ar))*(exp(ar*(1.-ratio))));  // optimum rxn vel, 1/min
    float wn        = fuelload/(1 + st);                               // net fuel loading, lb/ft^2
    float rtemp1    = fmc_g/fuelmce;
    float etam      = 1.-2.59*rtemp1 +5.11*(pow(rtemp1, 2)) -3.52*(pow(rtemp1, 3));  // moist damp coef
    float etas      = 0.174* pow(se, -0.19);                           // mineral damping coef
    float ir        = gamma * wn * fuelheat * etam * etas;             // rxn intensity,btu/ft^2 min
    float xifr      = exp((0.792 + 0.681*(pow(savr, 0.5)))*(betafl+0.1))/(192.+0.2595*savr);// propagating flux ratio   
    float rothR0    = ir*xifr/(rhob*epsilon*qig);                      // SPREAD RATE [ft/s]
    float R0        = rothR0 * .005080;                                // SPREAD RATE [m/s]
    
    return R0;

}

// Balbi (2019) fire propagation model
struct Fire::FireProperties Fire :: balbi(FuelProperties* fuel,float u_mid, float v_mid, 
                                          float x_norm, float y_norm, float tanphi, float fmc_g) {
        
    // fuel properties
    float fgi        = fuel->fgi;              ///< initial total mass of surface fuel [kg/m**2]
    float fueldepthm = fuel->fueldepthm;       ///< fuel depth [m]
    int savr          = fuel->savr;             ///< fuel particle surface-area-to-volume ratio, [1/ft]
    int cmbcnst       = fuel->cmbcnst;          ///< joules per kg of dry fuel [J/kg]
    
    // universal constants
    float g        = 9.81;                     ///< gravity 
    float pi       = 3.14159265358979323846;   ///< pi
    float s        = 17;                       ///< stoichiometric constant - Balbi 2018
    float Chi_0    = 0.3;                      ///< thin flame radiant fraction - Balbi 2009
    float B        = 5.67e-8;                  ///< Stefan-Boltzman 
    float Deltah_v = 2.257e6;                  ///< water evap enthalpy [J/kg]
    float C_p      = 2e3;                      ///< calorific capacity [J/kg] - Balbi 2009  
    float C_pa     = 1150;                     ///< specific heat of air [J/Kg/K]
    float tau_0    = 75591;                    ///< residence time coefficient - Anderson 196?
    float tau      = 75591/(savr/0.3048);

    // fuel constants
    float m        = fmc_g;                    ///< fuel particle moisture content [0-1]
    float sigma    = fgi;                      ///< dead fuel load [kg/m^2]
    float sigmaT   = sigma;                    ///< total fuel load [kg/m^2]
    float rhoFuel  = 1500;                     ///< fuel density [kg/m^3]
    float T_i      = 600;                      ///< ignition temp [k]
    
    // model parameters
    float beta  = sigma/(fueldepthm*rhoFuel);  ///< packing ratio of dead fuel [eq.1]
    float betaT = sigmaT/(fueldepthm*rhoFuel); ///< total packing ratio [eq.2]
    float SAV   = savr/0.3048;                 ///< surface area to volume ratio [m^2/m^3]
    float lai   = (SAV*fueldepthm*beta)/2;     ///< leaf Area Index for dead fuel [eq.3]
    float nu    = fmin(2*lai,2*pi*beta/betaT); ///< absorption coefficient [eq.5]
    float lv    = fueldepthm;                  ///< fuel length [m] ?? need better parameterization here
    float K1    = 100;                         ///< drag force coefficient: 100 for field, 1400 for lab 
    float r_00  = 2.5e-5;                      ///< model parameter ??
    
    // Environmental Constants
    float rhoAir = 1.125;                      ///< air Density [Kg/m^3]
    float T_a    = 293.15;                     ///< air Temp [K]
    float alpha = atan(tanphi);               ///< slope angle [rad]
    float psi    = 0;                          ///< angle between wind and flame front [rad]
    float phi    = 0;                          ///< angle between flame front vector and slope vector [rad]

    float cos_psi = (u_mid*x_norm + v_mid*y_norm)/(sqrt(u_mid*u_mid + v_mid*v_mid)*sqrt(x_norm*x_norm + y_norm*y_norm));
   
    float KDrag = K1*betaT*fmin(fueldepthm/lv,1);  ///< Drag force coefficient [eq.7]

    float q = C_p*(T_i - T_a) + m*Deltah_v;        ///< Activation energy [eq.14]

    float A = fmin(SAV/(2*pi),beta/betaT)*Chi_0*cmbcnst/(4*q);     ///< Radiant coefficient [eq.13]
    
    // Initial guess = Rothermel ROS 
    float R = rothermel(fuel,max(u_mid,v_mid),tanphi,fmc_g);       ///< Total Rate of Spread (ROS) [m/s]
    
    // Initial tilt angle guess = slope angle
    float gamma  = alpha;    ///< Flame tilt angle
    float maxIter = 100;
    float R_tol   = 1e-5;
    float iter    = 1;
    float error   = 1;
    float R_old   = R;
    
    // find spread rates
    float Chi;                     ///< Radiative fraction [-]
    float TFlame;                  ///< Flame Temp [K]
    float u0;                      ///< Upward gas velocity [m/s] 
    float H;                       ///< Flame height [m]
    float b;                       ///< Convective coefficient [-]
    float ROSBase;                 ///< ROS from base radiation [m/s]
    float ROSFlame;                ///< ROS from flame radiation [m/s]
    float ROSConv;                 ///< ROS from convection [m/s] 
    
    float V_mid = sqrt(u_mid*u_mid + v_mid*v_mid);         ///< Midflame Wind Velocity [m/s]
    while (iter < maxIter && error > R_tol){
        Chi = Chi_0/(1 + R*cos(gamma)/(SAV*r_00));          //[eq.20]
        
        TFlame = T_a + cmbcnst*(1 - Chi)/((s+1)*C_pa);      //[eq.16]
        
        u0 = 2*nu*((s+1)/tau_0)*(rhoFuel/rhoAir)*(TFlame/T_a);   //[eq.19]
        
        
        gamma = atan(tan(alpha)*cos(phi)+V_mid*cos_psi/u0);    
        
        H = u0*u0/(g*(TFlame/T_a - 1)*cos(alpha)*cos(alpha));   //[eq.17]
        
        b = 1/(q*tau_0*u0*betaT)*Deltah_v*nu*fmin(s/30,1);      //[eq.8]
        
        /** 
         * Compute ROS
        */
         
        // ROS from base radiation
        ROSBase = fmin(SAV*fueldepthm*betaT/pi,1)*(beta/betaT)*(beta/betaT)*(B*TFlame*TFlame*TFlame*TFlame)/(beta*rhoFuel*q);
        
        // ROS from flame radiation [eq.11]
        ROSFlame = A*R*(1+sin(gamma) - cos (gamma))/(1 + R*cos(gamma)/(SAV*r_00));
        
        // ROS from convection
        ROSConv = b*(tan(alpha) + 2*V_mid/u0*exp(-KDrag*R));
        
        // Total ROS 
        R    = ROSBase + ROSFlame + ROSConv;
        error = std::abs(R-R_old);
        R_old = R;
    }
        
    // calculate flame depth
    float L = R*tau;
    // calculate heat release
    float Q0 = cmbcnst*fgi*dx*dy/tau;
    float H0 = (1-Chi)*Q0;
    // calculate plume centerline characteristic velocity and length
    float U_c = pow(g*g*H0/rhoAir/T_a/C_p, 1/5);
    float L_c = pow(H0/rhoAir/C_p/T_a/pow(g, 1/2), 2/5);

    // struct to hold computed fire properties
    struct FireProperties fp;
    
    // set fire properties
    fp.w    = u0;
    fp.h    = H;
    fp.d    = L;
    fp.r    = R;
    fp.T    = TFlame;
    fp.tau  = tau;
    fp.K    = KDrag;
    fp.H0   = H0;
    fp.U_c  = U_c;
    fp.L_c  = L_c;

    return fp;
}


// Save output at cell-centered values
void Fire :: save(Output* output) {
    
    // output size and location
    std::vector<size_t> scalar_index;
    std::vector<size_t> scalar_size;
    std::vector<size_t> vector_index;
    std::vector<size_t> vector_size;
    
    scalar_index = {static_cast<unsigned long>(output_counter)};
    scalar_size  = {1};
    vector_index = {static_cast<size_t>(output_counter), 0, 0};
    vector_size  = {1, static_cast<unsigned long>(ny-1), static_cast<unsigned long>(nx-1)};
    
    // set time 
    time += dt;
	
    // loop through 1D fields to save
    for (int i=0; i<output_scalar_dbl.size(); i++) {
        output->saveField1D(output_scalar_dbl[i].name, scalar_index, output_scalar_dbl[i].data);
    }
    // loop through 2D double fields to save
    for (int i=0; i<output_vector_dbl.size(); i++) {
        
        output->saveField2D(output_vector_dbl[i].name, vector_index,
                                vector_size, *output_vector_dbl[i].data);
    }
    // loop through 2D int fields to save
    for (int i=0; i<output_vector_int.size(); i++) { 
        output->saveField2D(output_vector_int[i].name, vector_index,
                            vector_size, *output_vector_int[i].data);
    }
    
    // increment for next time insertion
    output_counter +=1;
}
