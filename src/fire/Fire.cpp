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
Fire :: Fire(WINDSInputData* WID, WINDSGeneralData* WGD) {
	// get domain information
	nx = WGD->nx;
	ny = WGD->ny;
	nz = WGD->nz;    
	dx = WGD->dx;
	dy = WGD->dy;
	dz = WGD->dz;
	FFII_flag = WID->fires->fieldFlag;
	std::cout<<"Field flag ["<<FFII_flag<<"]"<<std::endl;
	fmc = WID->fires->fmc;

	/**
	* Set-up the mapper array - cell centered
	**/
	fire_cells.resize((nx-1)*(ny-1));
	fuel_map.resize((nx-1)*(ny-1));
	burn_flag.resize((nx-1)*(ny-1));
	burn_out.resize((nx-1)*(ny-1));
	front_map.resize((nx-1)*(ny-1));
	del_plus.resize((nx-1)*(ny-1));
	del_min.resize((nx-1)*(ny-1));
	xNorm.resize((nx-1)*(ny-1));
	yNorm.resize((nx-1)*(ny-1));
	slope_x.resize((nx-1)*(ny-1));
	slope_y.resize((nx-1)*(ny-1));
	Force.resize((nx-1)*(ny-1));
	z_mix.resize((nx-1)*(ny-1));
	z_mix_old.resize((nx-1)*(ny-1));
	/**
	* Read Potential Field
	**/
	// set-up potential field array - cell centered
	Pot_u.resize((nx-1)*(ny-1)*(nz-2));
	Pot_v.resize((nx-1)*(ny-1)*(nz-2));
	Pot_w.resize((nx-1)*(ny-1)*(nz-2));

	// Open netCDF for Potential Field as read only
	NcFile Potential("../data/FireFiles/HeatPot.nc", NcFile::read);

	// Get size of netCDF data
	pot_z = Potential.getVar("u_r").getDim(0).getSize();
	pot_r = Potential.getVar("u_r").getDim(1).getSize();
	pot_G = Potential.getVar("G").getDim(0).getSize();
	pot_rStar = Potential.getVar("rStar").getDim(0).getSize();
	pot_zStar = Potential.getVar("zStar").getDim(0).getSize();
	// Allocate variable arrays
	u_r.resize(pot_z * pot_r);
	u_z.resize(pot_z * pot_r);
	G.resize(pot_G);
	Gprime.resize(pot_G);
	rStar.resize(pot_rStar);
	zStar.resize(pot_zStar);
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
	/**
	* Set initial fire info
	*/
	if(FFII_flag == 1){
		// Open netCDF for fire times
		//std::cout<<"nc file open"<<std::endl;
		NcFile FireTime("../data/FireFiles/FFII.nc", NcFile::read);
		//std::cout<<"nc file read"<<std::endl;
		// Get size of netCDF data
		SFT_time = FireTime.getVar("time").getDim(0).getSize();
		// Allocate variable arrays
		FT_time.resize(SFT_time);
		FT_x1.resize(SFT_time);
		FT_y1.resize(SFT_time);
		FT_x2.resize(SFT_time);
		FT_y2.resize(SFT_time);

		// Get variables from netCDF
    
		FireTime.getVar("time").getVar({0},{SFT_time},FT_time.data());
 		FireTime.getVar("y1").getVar({0},{SFT_time},FT_x1.data());
 		FireTime.getVar("x1").getVar({0},{SFT_time},FT_y1.data());
 		FireTime.getVar("y2").getVar({0},{SFT_time},FT_x2.data());
 		FireTime.getVar("x2").getVar({0},{SFT_time},FT_y2.data());
		std::cout<<"FFII burn times read"<<std::endl;    
	}
	if(FFII_flag == 2){
		// Open netCDF for fire times
		//std::cout<<"nc file open"<<std::endl;
		NcFile FireTime("../data/FireFiles/RxCadreL2F.nc", NcFile::read);
		//std::cout<<"nc file read"<<std::endl;
		// Get size of netCDF data
		SFT_time = FireTime.getVar("time").getDim(0).getSize();
		// Allocate variable arrays
		FT_time.resize(SFT_time);
		FT_x1.resize(SFT_time);
		FT_y1.resize(SFT_time);
		FT_x2.resize(SFT_time);
		FT_y2.resize(SFT_time);
		FT_x3.resize(SFT_time);
		FT_y3.resize(SFT_time);

		// Get variables from netCDF
    
		FireTime.getVar("time").getVar({0},{SFT_time},FT_time.data());
		FireTime.getVar("x1").getVar({0},{SFT_time},FT_x1.data());
		FireTime.getVar("y1").getVar({0},{SFT_time},FT_y1.data());
		FireTime.getVar("x2").getVar({0},{SFT_time},FT_x2.data());
		FireTime.getVar("y2").getVar({0},{SFT_time},FT_y2.data());
		FireTime.getVar("x3").getVar({0},{SFT_time},FT_x3.data());
		FireTime.getVar("y3").getVar({0},{SFT_time},FT_y3.data());
		std::cout<<"L2F burn times read"<<std::endl;
    
	}

	fuel_type  = WID->fires->fuelType;
	courant    = WID->fires->courant;
    
	// Set fuel properties for domain
	std::string fuelFile = WID->fires->fuelFile;
	FuelRead* fuelField = nullptr;
	Mesh* fuelMesh;

	Vector3<int>* domain;
	Vector3<float>* grid;
	domain = WID->simParams->domain;
	grid = WID->simParams->grid;
	if (fuelFile != ""){
		std::cout<<"Extracting fuel data from "<< fuelFile << std::endl;
		fuelField = new FuelRead(fuelFile,(*(grid))[0],(*(grid))[1] );	
		assert(fuelField);	
		//fuelField->setDomain(domain, grid);
		fuelMesh = new Mesh(fuelField->getTris());
    
		for (int j = 0; j < ny-1; j++){
			for (int i = 0; i < nx-1; i++){
				int idx = i+j*(nx-1);
				//float fID = fuelMesh->getHeight(i*dx+dx*0.5f,j*dy+dy*0.5f);
				float fID = fuelMesh->getHeight(i*dx,j*dy);				
				//std::cout<<"fuelMesh ["<<i<<"]["<<j<<"] = "<<fID<<std::endl;
				fuel_map[idx] = round(fID);
				int fuelID = round(fID);
					
				if (fuelID==1)  	fire_cells[idx].fuel = new ShortGrass();
				else if (fuelID==2)	fire_cells[idx].fuel = new TimberGrass();
				else if (fuelID==3) fire_cells[idx].fuel = new TallGrass();
				else if (fuelID==4)	fire_cells[idx].fuel = new Chaparral();
				else if (fuelID==5) fire_cells[idx].fuel = new Brush();
				else if (fuelID==6) fire_cells[idx].fuel = new DormantBrush();
				else if (fuelID==7)  fire_cells[idx].fuel = new SouthernRough();
				else if (fuelID==8)  fire_cells[idx].fuel = new TimberClosedLitter();
				else if (fuelID==9)  fire_cells[idx].fuel = new HarwoodLitter();
				else if (fuelID==10) fire_cells[idx].fuel = new TimberLitter();
				else if (fuelID==11) fire_cells[idx].fuel = new LoggingSlashLight();
				else if (fuelID==12) fire_cells[idx].fuel = new LoggingSlashMedium();
				else if (fuelID==13) fire_cells[idx].fuel = new LoggingSlashHeavy();
				else if (fuelID==91) fire_cells[idx].fuel = new Urban();
				else if (fuelID==92) fire_cells[idx].fuel = new Snow();
				else if (fuelID==93) fire_cells[idx].fuel = new Agricultural();
				else if (fuelID==98) fire_cells[idx].fuel = new Water();
				else if (fuelID==99) fire_cells[idx].fuel = new Bare();
				else if (fuelID==101) fire_cells[idx].fuel = new GR1();
				else if (fuelID==102) fire_cells[idx].fuel = new GR2();
				else if (fuelID==103) fire_cells[idx].fuel = new GR3();
				else if (fuelID==104) fire_cells[idx].fuel = new GR4();
				else if (fuelID==105) fire_cells[idx].fuel = new GR5();
				else if (fuelID==106) fire_cells[idx].fuel = new GR6();
				else if (fuelID==107) fire_cells[idx].fuel = new GR7();
				else if (fuelID==108) fire_cells[idx].fuel = new GR8();
				else if (fuelID==109) fire_cells[idx].fuel = new GR9();
				else if (fuelID==121) fire_cells[idx].fuel = new GS1();
				else if (fuelID==122) fire_cells[idx].fuel = new GS2();
				else if (fuelID==123) fire_cells[idx].fuel = new GS3();
				else if (fuelID==124) fire_cells[idx].fuel = new GS4();
				else if (fuelID==141) fire_cells[idx].fuel = new SH1();
				else if (fuelID==142) fire_cells[idx].fuel = new SH2();
				else if (fuelID==143) fire_cells[idx].fuel = new SH3();
				else if (fuelID==144) fire_cells[idx].fuel = new SH4();
				else if (fuelID==145) fire_cells[idx].fuel = new SH5();
				else if (fuelID==146) fire_cells[idx].fuel = new SH6();
				else if (fuelID==147) fire_cells[idx].fuel = new SH7();
				else if (fuelID==148) fire_cells[idx].fuel = new SH8();
				else if (fuelID==149) fire_cells[idx].fuel = new SH9();
				else if (fuelID==161) fire_cells[idx].fuel = new TU1();
				else if (fuelID==162) fire_cells[idx].fuel = new TU2();
				else if (fuelID==163) fire_cells[idx].fuel = new TU3();
				else if (fuelID==164) fire_cells[idx].fuel = new TU4();
				else if (fuelID==165) fire_cells[idx].fuel = new TU5();
				else if (fuelID==181) fire_cells[idx].fuel = new TL1();
				else if (fuelID==182) fire_cells[idx].fuel = new TL2();
				else if (fuelID==183) fire_cells[idx].fuel = new TL3();
				else if (fuelID==184) fire_cells[idx].fuel = new TL4();
				else if (fuelID==185) fire_cells[idx].fuel = new TL5();
				else if (fuelID==186) fire_cells[idx].fuel = new TL6();
				else if (fuelID==187) fire_cells[idx].fuel = new TL7();
				else if (fuelID==188) fire_cells[idx].fuel = new TL8();
				else if (fuelID==189) fire_cells[idx].fuel = new TL9();
				else if (fuelID==201) fire_cells[idx].fuel = new SB1();
				else if (fuelID==202) fire_cells[idx].fuel = new SB2();
				else if (fuelID==203) fire_cells[idx].fuel = new SB3();
				else if (fuelID==204) fire_cells[idx].fuel = new SB4();
				else fire_cells[idx].fuel = new SH1();
				
			}
		}
	std::cout<<"fuel set"<<std::endl;								
	} else {
		for (int j = 0; j < ny-1; j++){
			for (int i = 0; i < nx-1; i++){
				int idx = i + j*(nx-1);
				fuel_map[idx] = fuel_type;            
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
				if (fuel_type==91) fire_cells[idx].fuel = new Urban();
				if (fuel_type==92) fire_cells[idx].fuel = new Snow();
				if (fuel_type==93) fire_cells[idx].fuel = new Agricultural();
				if (fuel_type==98) fire_cells[idx].fuel = new Water();
				if (fuel_type==99) fire_cells[idx].fuel = new Bare();
				if (fuel_type==101) fire_cells[idx].fuel = new GR1();
				if (fuel_type==102) fire_cells[idx].fuel = new GR2();
				if (fuel_type==103) fire_cells[idx].fuel = new GR3();
				if (fuel_type==104) fire_cells[idx].fuel = new GR4();
				if (fuel_type==105) fire_cells[idx].fuel = new GR5();
				if (fuel_type==106) fire_cells[idx].fuel = new GR6();
				if (fuel_type==107) fire_cells[idx].fuel = new GR7();
				if (fuel_type==108) fire_cells[idx].fuel = new GR8();
				if (fuel_type==109) fire_cells[idx].fuel = new GR9();
				if (fuel_type==121) fire_cells[idx].fuel = new GS1();
				if (fuel_type==122) fire_cells[idx].fuel = new GS2();
				if (fuel_type==123) fire_cells[idx].fuel = new GS3();
				if (fuel_type==124) fire_cells[idx].fuel = new GS4();
				if (fuel_type==141) fire_cells[idx].fuel = new SH1();
				if (fuel_type==142) fire_cells[idx].fuel = new SH2();
				if (fuel_type==143) fire_cells[idx].fuel = new SH3();
				if (fuel_type==144) fire_cells[idx].fuel = new SH4();
				if (fuel_type==145) fire_cells[idx].fuel = new SH5();
				if (fuel_type==146) fire_cells[idx].fuel = new SH6();
				if (fuel_type==147) fire_cells[idx].fuel = new SH7();
				if (fuel_type==148) fire_cells[idx].fuel = new SH8();
				if (fuel_type==149) fire_cells[idx].fuel = new SH9();
				if (fuel_type==161) fire_cells[idx].fuel = new TU1();
				if (fuel_type==162) fire_cells[idx].fuel = new TU2();
				if (fuel_type==163) fire_cells[idx].fuel = new TU3();
				if (fuel_type==164) fire_cells[idx].fuel = new TU4();
				if (fuel_type==165) fire_cells[idx].fuel = new TU5();
				if (fuel_type==181) fire_cells[idx].fuel = new TL1();
				if (fuel_type==182) fire_cells[idx].fuel = new TL2();
				if (fuel_type==183) fire_cells[idx].fuel = new TL3();
				if (fuel_type==184) fire_cells[idx].fuel = new TL4();
				if (fuel_type==185) fire_cells[idx].fuel = new TL5();
				if (fuel_type==186) fire_cells[idx].fuel = new TL6();
				if (fuel_type==187) fire_cells[idx].fuel = new TL7();
				if (fuel_type==188) fire_cells[idx].fuel = new TL8();
				if (fuel_type==189) fire_cells[idx].fuel = new TL9();
				if (fuel_type==201) fire_cells[idx].fuel = new SB1();
				if (fuel_type==202) fire_cells[idx].fuel = new SB2();
				if (fuel_type==203) fire_cells[idx].fuel = new SB3();
				if (fuel_type==204) fire_cells[idx].fuel = new SB4();		
			}
		}
	std::cout<<"fuel set constant = "<<fuel_type<<std::endl;		
	}
	
	for (int fidx = 0; fidx < WID->fires->IG.size(); fidx++){
	std::cout<<"ignition ["<<fidx<<"]"<<std::endl;
	x_start    = WID->fires->IG[fidx]->xStart;
	y_start    = WID->fires->IG[fidx]->yStart;
	H          = WID->fires->IG[fidx]->height;
	L          = WID->fires->IG[fidx]->length;
	W          = WID->fires->IG[fidx]->width;
	baseHeight = WID->fires->IG[fidx]->baseHeight;
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
	std::cout<<"burn initialized"<<std::endl;
	/**
	* Set up initial level set. Use signed distance function: swap to fast marching method in future.
	*/
	float sdf, sdf_min;
	for (int j = 0; j < ny-1; j++){
		for (int i = 0; i < nx-1; i++){
			int idx = i + j*(nx-1);
			if (fire_cells[idx].state.front_flag == 1){
				front_map[idx] = 0;                
			} else {
				sdf = 1000;
				for (int jj = 0; jj < ny-1; jj++){
					for (int ii = 0; ii < nx-1; ii++){
						int idx2 = ii + jj*(nx-1);
						if (fire_cells[idx2].state.front_flag == 1){
							sdf_min = sqrt ((ii-i)*(ii-i) + (jj-j)*(jj-j));
						} else{
							sdf_min = 1000;
						}
						sdf = sdf_min < sdf ? sdf_min : sdf;
					}
				}
				front_map[idx] = sdf;                
			}
		}
	}
	std::cout<<"level set initialized"<<std::endl;
	/**
	* Calculate slope at each terrain cell
	*/
	for (int j=1; j < ny-2; j++){
		for (int i=1; i < nx-2; i++){
			int id = i + j*(nx-1);
			int idxp = (i+1) + j*(nx-1);
			int idxm = (i-1) + j*(nx-1);
			int idyp = i + (j+1)*(nx-1);
			int idym = i + (j-1)*(nx-1);
			float delzx = WGD->terrain[idxp]-WGD->terrain[idxm];
			float delzy = WGD->terrain[idyp]-WGD->terrain[idym];
			slope_x[id] = (delzx/(2*dx))/sqrt(delzx*delzx + 2*dx*2*dx);
			slope_y[id] = (delzy/(2*dy))/sqrt(delzy*delzy + 2*dy*2*dy);	    
		}
	}
	std::cout<<"slope calculated"<<std::endl;
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
	std::cout<<"max ROS = "<<r_max<<std::endl;
	if (r_max < 0.3){
		r_max = 0.3;
	}
	float dt = courant * dx / r_max;
	std::cout<<"dt = "<<dt<<" s"<<std::endl;
	return courant * dx / r_max;	
}

/**
 * Compute fire spread for burning cells
 */
void Fire :: run(Solver* solver, WINDSGeneralData* WGD) {


    /**
     * Calculate level set gradient and norm (Chapter 6, Sethian 2008)
     */
	float dmx, dpx, dmy, dpy, n_star_x, n_star_y;
	float sdmx, sdpx, sdmy, sdpy, sn_star_x,sn_star_y;
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
	std::cout<<"level set calculated"<<std::endl;
	/**
	* Reset forcing function for level set
	*/
	std::fill (Force.begin(),Force.end(),0);

	/**
	* Calculate Forcing Function (Balbi model at mid-flame height or first grid cell if no fire)
	*/
    
	for (int j=1; j < ny-2; j++){
		for (int i=1; i < nx-2; i++){
			int idx = i + j*(nx-1);
			// get fuel properties at this location
			struct FuelProperties* fuel = fire_cells[idx].fuel;
			// calculate mid-flame height
			int kh   = 0;
			float H = fire_cells[idx].properties.h;
			float T = WGD->terrain[idx];
			float D = fuel->fuelDepth*0.3048;
			float FD = H + T + D;
        
			if (H==0) {
				kh = std::round(T/dz);
			} else {
				kh = std::round(FD/dz);
			}
			// call u and v from WINDS General Data
			int cell_face = i + j*nx + kh*nx*ny;
			float u = 0.5*(WGD->u[cell_face] + WGD->u[cell_face+1]);
			float v = 0.5*(WGD->v[cell_face] + WGD->v[cell_face+nx]);
			// run Balbi model
			struct FireProperties fp = balbi(fuel,u,v,xNorm[idx],yNorm[idx],slope_x[idx],slope_y[idx],fmc);
			fire_cells[idx].properties = fp;
			Force[idx] = fp.r;
		}
	}
	std::cout<<"Level set forcing calculated"<<std::endl;
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
       
		// convert flat index to i, j at cell center
		int ii  = id % (nx-1);
		int jj  = (id / (nx-1)) % (ny-1);
		int idx = ii+jj*nx;	
		float T = WGD->terrain[idx];
		float D = fuel->fuelDepth*0.3048;
		int TID = std::round(T/dz);
		float FD = H/2.0 + T + D;
		float MFD = maxH + T + D;
		if (H==0) {
			kh = std::round(T/dz);
		} else {
			kh = std::round(FD/dz);
		}
		if (maxH==0) {
			maxkh = std::round(T/dz);
		} else {
			maxkh = std::round(MFD/dz);
		}
               
		// get horizontal wind at flame height
		int cell_face = ii + jj*nx + kh*ny*nx;
		float u = 0.5*(WGD->u[cell_face] + WGD->u[cell_face+1]);
		float v = 0.5*(WGD->v[cell_face] + WGD->v[cell_face+nx]);
		// run Balbi model
		float burnTime = fire_cells[id].state.burn_time;
		struct FireProperties fp = balbi(fuel,u,v,xNorm[id],yNorm[id],slope_x[id],slope_y[id],fmc);
		fire_cells[id].properties = fp;
		Force[id] = fp.r;
		// update icell value for flame
		for (int k=TID; k<= maxkh; k++){
			int icell_cent = ii + jj*(nx-1) + (k)*(nx-1)*(ny-1);
			WGD->icellflag[icell_cent] = 12;
		}
	}
	// compute time step
	dt = computeTimeStep();
	std::vector<int>().swap(cells_burning);
}

/**
 * Calcluate potential field
 */

void Fire :: potential(WINDSGeneralData* WGD){
	// dr and dz, assume linear spacing between
	float drStar = rStar[1]-rStar[0];
	float dzStar = zStar[1]-zStar[0];

    // reset potential fields
    std::fill (Pot_u.begin(),Pot_u.end(),0);
    std::fill (Pot_v.begin(),Pot_v.end(),0);
    std::fill (Pot_w.begin(),Pot_w.end(),0);

    // set z_mix to terrain height
    for (int i = 0; i<nx-1; i++){
      	for (int j = 0; j<ny-1; j++){
			int id = i+j*(nx-1);
			z_mix[id] = WGD->terrain_id[id];
      	}
    }
    /**
     * Calculate Potential field based on heat release
     * Baum and McCaffrey plume model
     **/
    
    // loop through burning cells to get heat release 

    float H0 = 0;                                   ///< heat release
    float fa = 0;                                   ///< fire area
    int cent = 0;
    int icent = 0;
    int jcent = 0;
    int counter = 0;
    int firei;				///< index center of merged fire sources (i)
    int firej;				///< index center of merged fire sources (j)
    int firek;				///< index average height of terrain of merged fires (k)
    
	for (int ii = 0; ii < nx-1; ii++){
		for (int jj = 0; jj < ny-1; jj++){
			int id = ii + jj*(nx-1);
			if (burn_flag[id] == 1){
				counter += 1;
				icent += ii;
				jcent += jj;
				H0 += fire_cells[id].properties.H0;
				// Hard code heat release		
				//H0 += 1.80357e6*dx*dy/25/25;
				fa += dx*dy;
			}
		}
	}
    
	if(H0 != 0){
		float g = 9.81;
		float rhoAir = 1.125;
		float C_pa = 1150;
		//float C_p = 2000;
		float T_a = 293.15;
		float U_c = pow(g*g*H0/rhoAir/T_a/C_pa, 1.0/5.0);
		float L_c = pow(H0/rhoAir/C_pa/T_a/pow(g, 1.0/2.0), 2.0/5.0);
		float ur, uz; 
		float u_p; 	 					///< u velocity from potential field in target cell
		float v_p; 						///< v velocity from potential field in target cell
		float w_p;						///< w velocity from potential field in target cell
		float alpha_e = 0.09;                               ///< entrainment constant (Kaye & Linden 2004)
		float lambda_mix = 1/alpha_e*sqrt(25.0/132.0);      ///< nondimensional plume mixing height
		float kmax=0;                                    ///< plume mixing height
		int XIDX;
		int YIDX;
		int ZIDX = 0;
		int filt = 0;
		float k_fire = 0;					///< terrain index for plume merge
		float k_fire_old = 0;
		float mixIDX = 0;
		float mixIDX_old;

		while (filt < nx-1){
			filt = pow(2.0,ZIDX);
			//std::cout<<"Filter = "<<filt<<std::endl;
			ZIDX += 1;
			z_mix_old = z_mix;
			XIDX = 0;
			while (XIDX < nx-1-filt){
				YIDX = 0;
				while (YIDX < ny-1-filt){
					H = 0;
					k_fire = 0;
					k_fire_old = 0;
					icent = 0;
					jcent = 0;
					counter = 0;
					for (int ii = XIDX; ii < XIDX+filt; ii++){
						for (int jj = YIDX; jj < YIDX+filt; jj++){
							int id = ii + jj*(nx-1);
							if (burn_flag[id] == 1){
								struct FireProperties fp = fire_cells[id].properties;
								counter += 1;
			  				icent += ii;
			  				jcent += jj;
			  				H += fp.H0;
	    		  		// Hard code heat flux for WRF simulation burn
								//H += 1.80357e6*dx*dy/25/25;
								k_fire += WGD->terrain[id];
								k_fire_old += z_mix_old[id];
							}
						}
					}
		
					if (H != 0){
						firei = icent/counter;
						firej = jcent/counter;
						U_c = pow(g*g*H/rhoAir/T_a/C_pa, 1.0/5.0);
						L_c = pow(H/rhoAir/C_pa/T_a/pow(g, 1.0/2.0), 2.0/5.0);
						firek = k_fire/counter;
						mixIDX_old = floor(k_fire_old/counter);
						mixIDX = (lambda_mix*dx*filt);
						for (int ii = XIDX; ii < XIDX+filt; ii++){
							for (int jj = YIDX; jj < YIDX+filt; jj++){
								int id = ii + jj*(nx-1);
								z_mix[id] = mixIDX;
							}
						}
						//std::cout<<"Fire center = ["<<firei<<"]["<<firej<<"]"<<std::endl;
						//std::cout<<"Merge terrain index = "<<firek<<std::endl;
						//std::cout<<"Mix height index = "<<mixIDX<<std::endl;
						kmax = nz-3 > mixIDX ? mixIDX : nz-3;
						// Loop through vertical levels
						for (int kpot = mixIDX_old; kpot<kmax; kpot++) {
							// Calculate virtual origin
							float z_v = dx*filt*0.124/alpha_e+firek;				  ///< virtual origin for merged plumes
							float z_k = (kpot+z_v)*dz/L_c;                            ///< non-dim vertical distance between fire cell and target cell k
							if (z_k < 0){
								z_k = 0;
							}
							float zeta = 0;
							float x1 = 0;
							// Loop through horizontal domain
							for (int ipot = 0; ipot<nx-1; ipot++) {
								for (int jpot = 0; jpot<ny-1; jpot++){
									float deltaX = (ipot-firei)*dx/L_c;                      ///< non-dim distance between fire cell and target cell k in x direction
									float deltaY = (jpot-firej)*dy/L_c;                      ///< non-dim distance between fire cell and target cell k in y direction
									float h_k = sqrt(deltaX*deltaX + deltaY*deltaY);         ///< non-dim radial distance from fire cell and target cell k in horizontal
									u_p = 0;
									v_p = 0;
									w_p = 0;
									// if radius = 0
									if (h_k < 0.00001 && z_k < 60){
	
										float zMinIdx = floor(z_k/dzStar);
										float zMaxIdx = ceil(z_k/dzStar);		  
										//ur = 0.5*(u_r[zMinIdx*pot_r]+u_r[zMaxIdx*pot_r]);
										ur = 0.0;		
										//uz = 0.5*(u_z[zMinIdx*pot_r]+u_z[zMaxIdx*pot_r]);
										uz = u_z[zMinIdx*pot_r];
										u_p = U_c*ur;                         
										v_p = U_c*ur;                         
										w_p = U_c*uz;
									}
									// if in potential field lookup, r*(h_k) < 30 and z*(z_k) < 60
									else if (z_k < 60 && h_k < 30){ 
										// indices for lookup
										float rMinIdx = floor(h_k/drStar);
										float rMaxIdx = ceil(h_k/drStar);
										float zMinIdx = floor(z_k/dzStar);
										float zMaxIdx = ceil(z_k/dzStar);
										//ur = 0.25*(u_r[rMinIdx+zMinIdx*pot_r]+u_r[rMinIdx+zMaxIdx*pot_r]+u_r[rMaxIdx+zMinIdx*pot_r]+u_r[rMaxIdx+zMaxIdx*pot_r]); //lookup from u_r, linear interpolate between values
										//ur = 0.5*(u_r[rMinIdx + zMinIdx*pot_r]+u_r[rMaxIdx + zMinIdx*pot_r]);
										ur = u_r[rMinIdx + zMinIdx*pot_r];

										//uz = 0.25*(u_z[rMinIdx+zMinIdx*pot_r]+u_z[rMinIdx+zMaxIdx*pot_r]+u_z[rMaxIdx+zMinIdx*pot_r]+u_z[rMaxIdx+zMaxIdx*pot_r]); //lookup from u_z, linear interpolate between values
										//uz = 0.5*(u_z[rMinIdx + zMinIdx*pot_r]+u_z[rMinIdx + zMaxIdx*pot_r]);
										uz = u_z[rMinIdx + zMinIdx*pot_r];
		
										u_p = U_c*ur*deltaX/h_k;          
										v_p = U_c*ur*deltaY/h_k;          
										w_p = U_c*uz;                         
									} else {
			  							zeta = sqrt(h_k*h_k + z_k*z_k);
			  							x1 = (1+cos(atan(h_k/z_k)))/2.0;
			  							if(x1<0.5){
											std::cout<<"x1<0.5"<<std::endl;
											x1 = 0.5;
			  							} else if(isnan(x1)){
											std::cout<<"x1 is nan"<<std::endl;
			  							}
			  							// lookup indices for G(x) and G'(x) - spans 0.5 to 1.0
				  						int gMinIdx = floor(pot_G*(x1-.5)/.5);
				  						if(gMinIdx<0){
											std::cout<<"gMin error"<<std::endl;
				  						} else if(gMinIdx>pot_G){
											std::cout<<"gMin error"<<std::endl;
				  						} else if(isnan(gMinIdx)){
											std::cout<<"gMin NaN"<<std::endl;
				  						}
				  						int gMaxIdx = ceil(pot_G*(x1-.5)/.5);
			          					if(gMaxIdx>pot_G){
											std::cout<<"gMax error"<<std::endl;
				  						} else if (gMaxIdx < 0){
											std::cout<<"gMax error"<<std::endl;
				  						} else if(isnan(gMaxIdx)){
											std::cout<<"gMax NaN"<<std::endl;
				  						}
			 	  						// values for G and G'
				  
				  						float g_x = 0.5*(G[gMinIdx]+G[gMaxIdx]);
				  						float gprime_x = 0.5*(Gprime[gMinIdx]+Gprime[gMaxIdx]);
	
				  						ur = h_k/(2*PI*pow(h_k*h_k+z_k*z_k,(3/2.0))) + pow(zeta,(-1/3.0))*((5/6.0)*(1-2*x1)/sqrt(x1*(1-x1))*g_x - sqrt(x1*(1-x1))*gprime_x);
				  						uz = z_k/(2*PI*pow(h_k*h_k+z_k*z_k,(3/2.0))) + pow(zeta,(-1/3.0))*((5/3.0)*g_x + (1-2*x1)/2.0*gprime_x);
				  						if(h_k != 0){
			  							u_p = U_c*ur*deltaX/h_k;            
			  							v_p = U_c*ur*deltaY/h_k;
			  						}
			  						w_p = U_c*uz;

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
				YIDX += filt;
				}//while YIDX
			XIDX += filt;
        	}//while XIDX
		}//kfilt
	}//H0!=0
    
    
    // Modify u,v,w in solver - superimpose Potential field onto velocity field (interpolate from potential cell centered values)
    for (int iadd=1; iadd<nx-1; iadd++){
      	for (int jadd=1; jadd<ny-1; jadd++){
			for (int kadd=1; kadd<nz-2; kadd++){
	  			int cell_face = iadd + jadd*nx + (kadd-1)*nx*ny;
	  			int cell_cent = iadd + jadd*(nx-1) + (kadd-1)*(nx-1)*(ny-1);
	  			if (WGD->icellflag[cell_cent] == 12 or WGD->icellflag[cell_cent] == 1){
	    			WGD->u0[cell_face] = WGD->u0[cell_face] + 0.5*(Pot_u[cell_cent]+Pot_u[cell_cent+1]);
	    			WGD->v0[cell_face] = WGD->v0[cell_face] + 0.5*(Pot_v[cell_cent]+Pot_v[cell_cent+(nx-1)]);
	    			WGD->w0[cell_face] = WGD->w0[cell_face] + 0.5*(Pot_w[cell_cent]+Pot_w[cell_cent+(nx-1)*(ny-1)]);
	  			}
			}
      	}
    } 
}

/** 
 * Compute fire spread. Advance level set.
 */
void Fire :: move(Solver* solver, WINDSGeneralData* WGD){
    
	if (FFII_flag == 1){
		int FT_idx1 = 0;
    	int FT_idx2 = 0;
    	float FT = ceil(time)+FT_time[0];
    	int it;
		for (int IDX=0; IDX < FT_time.size(); IDX++){
			if (FT == FT_time[IDX]){
				it = IDX;
				break;
			}
		}
		int nx1 = round(FT_x1[it]/dx);
		int ny1 = round((750-FT_y1[it])/dy);
		FT_idx1 = nx1 + ny1*(nx-1);
		int nx2 = round(FT_x2[it]/dx);
		int ny2 = round((750-FT_y2[it])/dy);
		FT_idx2 = nx2 + ny2*(nx-1);
 		if (burn_flag[FT_idx1]<2){
	    	front_map[FT_idx1] = 0;
	    	fire_cells[FT_idx1].state.burn_flag = 1;
		}
		if (burn_flag[FT_idx2]<2){
	    	front_map[FT_idx2] = 0;
	    	fire_cells[FT_idx2].state.burn_flag = 1;
		}
    }
    if (FFII_flag == 2){
		int FT_idx1 = 0;
		int FT_idx2 = 0;
		int FT_idx3 = 0;
		float FT = ceil(time)+FT_time[0];
		int it=-1;
		for (int IDX=0; IDX < FT_time.size(); IDX++){
	    	if (FT == FT_time[IDX]){
	        	it = IDX;
	     		break;
	    	}
    	}
		if (it >= 0){
			int nx1 = round(FT_x1[it]/dx);
			int ny1 = round((FT_y1[it])/dy);
			FT_idx1 = nx1 + ny1*(nx-1);
			int nx2 = round(FT_x2[it]/dx);
			int ny2 = round((FT_y2[it])/dy);
			FT_idx2 = nx2 + ny2*(nx-1);
			int nx3 = round(FT_x3[it]/dx);
			int ny3 = round((FT_y3[it])/dy);
			FT_idx3 = nx3 + ny3*(nx-1);
			if (burn_flag[FT_idx1]<2){
				front_map[FT_idx1] = 0;
				fire_cells[FT_idx1].state.burn_flag = 1;
			}
			if (burn_flag[FT_idx2]<2){
				front_map[FT_idx2] = 0;
				fire_cells[FT_idx2].state.burn_flag = 1;
			}
			if (burn_flag[FT_idx3]<2){
				front_map[FT_idx3] = 0;
				fire_cells[FT_idx3].state.burn_flag = 1;
			}
		}
    }
    for (int j=1; j < ny-2; j++){
        for (int i=1; i < nx-2; i++){
	  		int idx = i + j*(nx-1);
          	// get fire properties at this location
        	struct FireProperties fp = fire_cells[idx].properties; 
	  		struct FuelProperties* fuel = fire_cells[idx].fuel;
        	float H = fire_cells[idx].properties.h*(1-(fire_cells[idx].state.burn_time/fire_cells[idx].properties.tau));
	  		float maxH = fire_cells[idx].properties.h;       
        	float T = WGD->terrain[idx];
        	float D = fuel->fuelDepth*0.3048;
        	int TID = std::round(T/dz);
        	float FD = H/2.0 + T + D;
	  		float MFD = maxH + T + D;
       		int kh = 0;
	  		int maxkh = 0;

        	if (H==0) {
        		kh = std::round(T/dz);
        	} else {
            	kh = std::round(FD/dz);
        	}
	  		if (maxH==0) {
	    		maxkh = std::round(T/dz);
	  		} else {
	    		maxkh = std::round(MFD/dz);
	  		}	  

            // if burn flag = 1, update burn time
            if (burn_flag[idx] == 1){
                fire_cells[idx].state.burn_time += dt;
            }	            
            // set burn flag to 2 (burned) if residence time exceeded, set Forcing function to 0, and update z0 to bare soil
            if (fire_cells[idx].state.burn_time >= fp.tau) {
                fire_cells[idx].state.burn_flag = 2;
				Force[idx] = 0;

				// Need to fix where z0 is reset MM
				//WGD->z0_domain[idx] = 0.01;
            }  

            // advance level set

          	front_map[idx] = front_map[idx] - dt*(fmax(Force[idx],0)*del_plus[idx] + fmin(Force[idx],0)*del_min[idx]);
           	// if level set <= 1, set burn_flag to 0.5 - L.S. for preheating
           	if (front_map[idx] <= 1 && burn_flag[idx] < 1){
               	fire_cells[idx].state.burn_flag = 0.5;
           	}
           	// if level set < threshold, set burn flag to 1
           	if (front_map[idx] <= 0.1 && burn_flag[idx] < 1){
               	fire_cells[idx].state.burn_flag = 1;
           	}
			// update burn flag field
			burn_flag[idx] = fire_cells[idx].state.burn_flag;
			burn_out[idx] = burn_flag[idx];	    

		}
	}
	// advance time
	time += dt;	
}

/**
 * Rothermel (1972) flame propgation model used for initial guess to Balbi
 */
float Fire :: rothermel(FuelProperties* fuel, float max_wind, float tanphi, float fmc_g) {
    
    // fuel properties
	float oneHour = fuel->oneHour; 					///< one hour fuel load [t/ac]
	float tenHour = fuel->tenHour;					///< ten hour fuel load [t/ac]
	float hundredHour = fuel->hundredHour;			///< hundred hour fuel load [t/ac]
	float liveHerb = fuel->liveHerb;				///< live herb fuel load [t/ac]
	float liveWoody = fuel->liveWoody; 				///< live woody fuel load [t/ac]
    float savOneHour        = fuel->savOneHour;		///< surface area to volume ratio of one hour fuel load
	float savTenHour 		= 109;					///< surface area to volume ratio of ten hour fuel load
	float savHundredHour 	= 30;					///< surface area to volume ratio of hundred hour fuel load
	float savHerb 			= fuel->savHerb;		///< surface area to volume ratio of herbacious fuel load
	float savWoody			= fuel->savWoody;		///< surface area to volume ratio of live woody fuel load
    float fueldens      	= fuel->fuelDensity;	///< ovendry fuel particle density [lb/ft^2]
    float st         		= 0.0555;				///< Total fuel mineral content
    float se         		= 0.0100;				///< Silica free mineral content
    float fgi = (oneHour+liveHerb+liveWoody)*0.2471;///< Initial fine fuel load [kg/m^2]
    float fuelmce    		= fuel->fuelmce/100;	///< fuel moisture content of extinction[%]
    float fuelheat   		= fuel->heatContent;	///< heat content of fuel [BTU/lb]
    float fueldepth  		= fuel->fuelDepth;		///< fuel bed depth [ft]
    
	float savr = (savOneHour*oneHour+savTenHour*tenHour+savHundredHour*hundredHour+savHerb*liveHerb+savWoody*liveWoody)
					/(oneHour+tenHour+hundredHour+liveHerb+liveWoody);		///< Characteristic fine fuel load surface area to volume ratio
    // local fire variables
    float bmst      = fmc_g/(1+fmc_g);
    float fuelloadm = (1.-bmst)*fgi;
    float fuelload  = fuelloadm*(pow(.3048, 2.0))*2.205;               // convert fuel load to lb/ft^2
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
                                          float x_norm, float y_norm, float x_slope, float y_slope, float fmc_g) {


    // struct to hold computed fire properties
    struct FireProperties fp;
   
    // fuel properties
	/*
    float fgi        = fuel->fgi;              ///< initial total mass of surface fuel [kg/m**2]
    float fueldepthm = fuel->fueldepthm;       ///< fuel depth [m]
    float savr          = fuel->savr;             ///< fuel particle surface-area-to-volume ratio, [1/ft]
    float cmbcnst       = fuel->cmbcnst;          ///< joules per kg of dry fuel [J/kg]
    */
	// Fuel Properties
	float oneHour = fuel->oneHour; 		///< one hour fuel load [t/ac]
	float tenHour = fuel->tenHour;		///< ten hour fuel load [t/ac]
	float hundredHour = fuel->hundredHour;	///< hundred hour fuel load [t/ac]
	float liveHerb = fuel->liveHerb;		///< live herb fuel load [t/ac]
	float liveWoody = fuel->liveWoody; 	///< live woody fuel load [t/ac]
	float savOneHour = fuel->savOneHour;	///< surface area to volume ratio of one hour fuels
	float savHerb = fuel->savHerb; 		///< surface area to volume ratio of live herbacious fuel load
	float savWoody = fuel->savWoody;		///< surface area to volume ratio of live woody fuel load
	float fueldepthm = fuel->fuelDepth*0.3048; ///< fuel bed depth [m]
	float cmbcnst = fuel->heatContent*2326; ///< heat content [J/kg]
	float savTenHour = 109;					///< surface area to volume ratio of ten hour fuel load
	float savHundredHour = 30;				///< surface area to volume ratio of hundred hour fuel load
	float rhoFuel = fuel->fuelDensity*16.0185; 		///< ovendry fuel particle density [kg/m^3]
	float fgi = (oneHour+liveHerb+liveWoody)*0.2471; 	///< Initial fine fuel load [kg/m^2]
	float savr = (savOneHour*oneHour+savTenHour*tenHour+savHundredHour*hundredHour+savHerb*liveHerb+savWoody*liveWoody)
					/(oneHour+tenHour+hundredHour+liveHerb+liveWoody);		///< Characteristic fine fuel load surface area to volume ratio


	if (fgi < 0.00001){

      // set fire properties
      fp.w    = 0;
      fp.h    = 0;
      fp.d    = 0;
      fp.r    = 0;
      fp.T    = 0;
      fp.tau  = 5000;
      fp.K    = 0;
      fp.H0   = 0;
    } else {

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
      float tau      = tau_0/(savr/0.3048);

      // fuel constants
      float m        = fmc_g;                   ///< fuel particle moisture content [0-1]
      float sigma    = oneHour*0.2471;		///< dead fine fuel load [kg/m^2]
      float sigmaT   = fgi;                    	///< total fine fuel load [kg/m^2]
      //float rhoFuel  = 1500;                    ///< fuel density [kg/m^3]
      float T_i      = 600;                     ///< ignition temp [k]
    
      // model parameters
      float beta  = sigma/(fueldepthm*rhoFuel);  ///< packing ratio of dead fuel [eq.1]
      float betaT = sigmaT/(fueldepthm*rhoFuel); ///< total packing ratio [eq.2]
      float SAV   = savr/0.3048;                 ///< surface area to volume ratio [m^2/m^3]
      float lai   = (SAV*fueldepthm*beta)/2;     ///< leaf Area Index for dead fuel [eq.3]
      float nu    = fmin(2*lai,2*pi*beta/betaT); ///< absorption coefficient [eq.5]
      float lv    = fueldepthm;                  ///< fuel length [m] ?? need better parameterization here
      float K1    = 100;                         ///< drag force coefficient: 100 for field, 1400 for lab 
      float r_00  = 2.5e-5;                      ///< model parameter 
    
      // Environmental Constants
      float rhoAir = 1.125;                      ///< air Density [Kg/m^3]
      float T_a    = 289.15;                     ///< air Temp [K]
      float alpha  = sqrt(x_slope*x_slope + y_slope*y_slope);		               ///< slope angle [rad]
      float psi    = 0;                          ///< angle between wind and flame front [rad]
      float phi    = 0;                          ///< angle between flame front vector and slope vector [rad]

      int s_c = 0;
      int n_c = 0;
      int v_c = 0;
      if (abs(x_slope) < 0.001 and abs(y_slope) < 0.001){
				s_c = 1;
      }
      if (abs(x_norm) < 0.001 and abs(y_norm) < 0.001){
				n_c = 1;
      }
      if (abs(u_mid) < 0.001 and abs(v_mid) < 0.001){
				v_c = 1;
      }
      if (s_c == 1 or n_c == 1){
			 	phi = 0;
      } else {
				phi = acos((x_slope*x_norm + y_slope*y_norm)/(sqrt(x_slope*x_slope + y_slope*y_slope)*sqrt(x_norm*x_norm + y_norm*y_norm)));
      }
      if (v_c == 1 or n_c == 1){
				psi = 0;
      } else {
				psi = acos((u_mid*x_norm + v_mid*y_norm)/(sqrt(u_mid*u_mid + v_mid*v_mid)*sqrt(x_norm*x_norm + y_norm*y_norm)));
      }
			if (phi > 0.785){
				alpha = -alpha;
			} 
      float KDrag = K1*betaT*fmin(fueldepthm/lv,1);  ///< Drag force coefficient [eq.7]
  
      float q = C_p*(T_i - T_a) + m*Deltah_v;        ///< Activation energy [eq.14]

      float A = fmin(SAV/(2*pi),beta/betaT)*Chi_0*cmbcnst/(4*q);     ///< Radiant coefficient [eq.13]
    
      // Initial guess = Rothermel ROS 
      float R = rothermel(fuel,max(u_mid,v_mid),alpha,fmc_g);       ///< Total Rate of Spread (ROS) [m/s]
	//float R = 0.1;
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
        
        
        gamma = atan(tan(alpha)*cos(phi)+V_mid*cos(psi)/u0);    
        
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
        ROSConv = b*(tan(alpha) + 2*V_mid/u0*exp(-KDrag*R))*cos(psi);
        
        // Total ROS 
        R    = ROSBase + ROSFlame + ROSConv;
	if(R < ROSBase){
		//std::cout<<"ROS Base = "<<ROSBase<<": ROS Flame = "<<ROSFlame<<": ROS Conv = "<<ROSConv<<": gamma = "<<gamma<<": psi = "<<psi<<std::endl;
		R = ROSBase;
	}
        error = std::abs(R-R_old);
        
	iter++;
	/*
	if(iter == 99){
		std::cout<<"Balbi iteration reached: R = "<<R<<": R old = "<<R_old<<": one hour =  "<<oneHour<<std::endl;
		std::cout<<"u = "<<u_mid<<": v = "<<v_mid<<": xnorm = "<<x_norm<<": ynorm = "<<y_norm<<": xslope = "<<x_slope<<": yslope = "<<y_slope<<std::endl;
	}
	*/
	R_old = R;
      }
        

      // calculate flame depth
      float L = R*tau;
      if (isnan(L)){
        L = dx;
      }
 
      float Q0 = cmbcnst*fgi*dx*dy/tau;
      float H0 = (1.0-Chi)*Q0;



    
      // set fire properties
      fp.w    = u0;
      fp.h    = H;
      fp.d    = L;
      fp.r    = R;
      fp.T    = TFlame;
      fp.tau  = tau;
      fp.K    = KDrag;
      fp.H0   = H0;
    }
    return fp;
}

