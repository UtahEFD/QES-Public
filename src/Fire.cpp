//
//  Fire.cpp
//  
//  This class models fire spread rate using Balbi (2019)
//
//  Created by Matthew Moody, Jeremy Gibbs on 12/27/18.
//

#include "Fire.hpp"

using namespace std;

Fire :: Fire(URBInputData* UID, Output* output) {
    
    // get domain information
    Vector3<int> domainInfo;
    domainInfo = *(UID->simParams->domain);
    nx = domainInfo[0];
    ny = domainInfo[1];
    nz = domainInfo[2];
    
    // get grid information
    Vector3<float> gridInfo;
    gridInfo = *(UID->simParams->grid);
    dx = gridInfo[0];
    dy = gridInfo[1];
    dz = gridInfo[2];
    
    // set-up the mapper array
    fire_cells.resize(nx*ny);
    burn_flag.resize(nx*ny);
    
    // get initial fire info
    x_start    = UID->fires->xStart;
    y_start    = UID->fires->yStart;
    H          = UID->fires->height;
    L          = UID->fires->length;
    W          = UID->fires->width;
    baseHeight = UID->fires->baseHeight;
    fuel_type  = UID->fires->fuelType;
    
    // set fuel properties
    for (int j = 0; j < ny; j++){
        for (int i = 0; i < nx; i++){
            
            int idx = i + j*nx;
            
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
    
    // set-up initial fire state
    for (int j = j_start; j < j_end; j++){
        for (int i = i_start; i < i_end; i++){
            int idx = i + j*nx;
	    	fire_cells[idx].state.burn_flag = 1;
        }
    }
    
    // set up burn flag field
    for (int j = 0; j < ny; j++){
        for (int i = 0; i < nx; i++){
            
            int idx = i + j*nx;       
            burn_flag[idx] = fire_cells[idx].state.burn_flag;
        }
    }
    
    // set output fields
    output_fields = UID->fileOptions->outputFields;
    
    if (output_fields[0]=="all") {
        output_fields.erase(output_fields.begin());
        output_fields = {"time","burn"};
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

    // create attributes
    AttScalarDbl att_t = {&time,      "time", "time[s]",         "--", dim_scalar_1};
    AttVectorDbl att_b = {&burn_flag, "burn", "burn flag value", "--", dim_scalar_3};
    
    // map the name to attributes
    map_att_scalar_dbl.emplace("time", att_t);
    map_att_vector_dbl.emplace("burn", att_b);
    
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

// compute adaptive time step
double Fire :: computeTimeStep(Solver* solver) {
    
    // get max wind speed
    double u,v,w,u_mag;
    double u_max = 0;
    double c = 1.0;
    for (int k = 1; k < nz+1; k++){
        for (int j = 0; j < ny+1; j++){
            for (int i = 0; i < nx+1; i++){
                
                int idx = i + j*(nx+1) + k*(ny+1)*(nx+1);
                
                u = solver->u[idx];
                v = solver->v[idx];
                w = solver->w[idx];
                
                u_mag = std::sqrt(std::pow(u,2)+std::pow(v,2)+std::pow(w,2));
                u_max = u_mag>u_max ? u_mag : u_max;    
            }
        }
    }
    
    return c * dx / u_max;
}

// compute fire spread for burning cells
void Fire :: run(Solver* solver) {
    
    // compute time step
    dt = computeTimeStep(solver);
    
    // indices for burning cells
    std::vector<int> cells_burning;
    
    // search predicate for burn state
    struct find_burn : std::unary_function<FireCell, bool> {
        double burn;
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
        int kh   = 0;
        double H = fire_cells[id].properties.h;
        double T = solver->terrain[id];
        double D = fuel->fueldepthm;
        
        double FD = H + T + D;
        
        if (H==0) {
            kh = 1;
        } else {
            kh = std::round(FD/dz);
        }
                
        // convert flat index to i, j at cell center
        int ii  = id % nx;
        int jj  = (id / nx) % ny;
        int iiF = ii+1;
        int iiB = ii-1;
        int jjF = jj+1;
        int jjB = jj-1;
                        
        // get horizontal wind at flame height
        double u = solver->u[ii + jj*(nx+1) + kh*(ny+1)*(nx+1)];
        double v = solver->v[ii + jj*(nx+1) + kh*(ny+1)*(nx+1)];
        
        // run Balbi model
        struct FireProperties fp = balbi(fuel,u,v,0.0,0.0650);
        fire_cells[id].properties = fp;
        
        // modify w0 in solver (adjust to use faces)
        for (int k=0; k<=kh; k++) {
            
            int idf  = ii + jj*(nx+1) + (k+2)*(nx+1)*(ny+1);
            int idxF = idf+1;
            int idxB = idf-1;
            int idyF = idf+(nx+1);
            int idyB = idf-(nx+1);
            
            double K = fire_cells[id].properties.K;            
            double u = solver->u0[idf];
            double v = solver->v0[idf];
            double u_uw, v_uw;
            
            if (u>0 && iiB>=0) {
                u_uw = solver->u0[idxB];
                solver->u0[idf] = u_uw * std::exp(-K*dx);
            }
            if (u<0 && iiF <= nx) {
                u_uw = solver->u0[idxF];
                solver->u0[idf] = u_uw * std::exp(-K*dx);
            }
            if (v>0 && jjB>=0) {
                v_uw = solver->v0[idyB];
                solver->v0[idf] = v_uw * std::exp(-K*dy);
            }
            if (v<0 && jjF <= ny) {
                v_uw = solver->v0[idyF];
                solver->v0[idf] = v_uw * std::exp(-K*dy);
            }
            
            solver->w0[ii + jj*(nx+1) + (k+2)*(nx+1)*(ny+1)] = fp.w;
        }
    }
}

// compute fire spread for burning cells
void Fire :: move(Solver* solver) {
    
    // indices for burning cells
    std::vector<int> cells_burning;
    
    // search predicate for burn state
    struct find_burn : std::unary_function<FireCell, bool> {
        double burn;
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
        
        // convert flat index to i, j at cell center
        int ii = id % nx;
        int jj = (id / nx) % ny;
                
        // get fire proiperties at this location
        struct FireProperties fp = fire_cells[id].properties;
        
        // check neighbors
        int idxF = id+1;
        int idxB = id-1;
        int idyF = id+nx;
        int idyB = id-nx;
        int idFF = id+1+nx;
        int idFB = id+1-nx;
        int idBF = id-1+nx;
        int idBB = id-1-nx;
        
        int iiF = ii+1;
        int iiB = ii-1;
        int jjF = jj+1;
        int jjB = jj-1;
                
        // check that x+1 is in-bounds, then compute fraction
        if (iiF <= (nx-1)) {
            double BxF = fire_cells[idxF].state.burn_flag;
            if (BxF != 1 && BxF != 2) {
                double frac = fmin(BxF+fp.rxf*dt/dx,1.0);
                fire_cells[idxF].state.burn_flag = frac;
            }
        }
        
        // check that x-1 is in-bounds, then compute fraction
        if (iiB>=0) {
            double BxB = fire_cells[idxB].state.burn_flag;
            if (BxB != 1 && BxB != 2) {
                double frac = fmin(BxB+fp.rxb*dt/dx,1.0);
                fire_cells[idxB].state.burn_flag = frac;
            }
        }
        
        // check that y+1 is in-bounds, then compute fraction
        if (jjF<=(ny-1)) {
            double ByF = fire_cells[idyF].state.burn_flag;
            if (ByF != 1 && ByF != 2) {
                double frac = fmin(ByF+fp.ryf*dt/dy,1.0);
                fire_cells[idyF].state.burn_flag = frac;        
            }
        }
        
        // check that y-1 is in-bounds, then compute fraction
        if (jjB>=0) {
            double ByB = fire_cells[idyB].state.burn_flag;
            if (ByB != 1 && ByB != 2) {
                double frac = fmin(ByB+fp.ryb*dt/dy,1.0);
                fire_cells[idyB].state.burn_flag = frac;
            }
        }
        
        // check that x+1, y+1 is in-bounds, then compute fraction
        if (iiF<=(nx-1) && jjF<=(ny-1)) {
            double BdFF = fire_cells[idFF].state.burn_flag;
            if (BdFF != 1 && BdFF != 2) {
                double frac = fmin(BdFF+fp.rxf*dt/dx+fp.ryf*dt/dy,1.0);
                fire_cells[idFF].state.burn_flag = frac;
            }
        }
        
        // check that x+1, y-1 is in-bounds, then compute fraction
        if (iiF<=(nx-1) && jjB>=0) {
            double BdFB = fire_cells[idFB].state.burn_flag;
            if (BdFB != 1 && BdFB != 2) {
                double frac = fmin(BdFB+fp.rxf*dt/dx+fp.ryb*dt/dy,1.0);
                fire_cells[idFB].state.burn_flag = frac;
            }
        }
        
        // check that x-1, y+1 is in-bounds, then compute fraction
        if (iiB>=0 && jjF<=(ny-1)) {
            double BdBF = fire_cells[idBF].state.burn_flag;
            if (BdBF != 1 && BdBF != 2) {
                double frac = fmin(BdBF+fp.rxb*dt/dx+fp.ryf*dt/dy,1.0);
                fire_cells[idBF].state.burn_flag = frac;
            }
        }
        
        // check that x-1, y-1 is in-bounds, then compute fraction
        if (iiB>=0 && jjB>=0) {
            double BdBB = fire_cells[idBB].state.burn_flag;
            if (BdBB != 1 && BdBB != 2) {
                double frac = fmin(BdBB+fp.rxb*dt/dx+fp.ryb*dt/dy,1.0);
                fire_cells[idBB].state.burn_flag = frac;
            }
        }
                        
        // update residence time
        fire_cells[id].state.burn_time += dt;
        
        // set burn flag to 2 (burned) if residence time exceeded
        if (fire_cells[id].state.burn_time >= fp.tau) {
            fire_cells[id].state.burn_flag = 2;
        }
    }
    
    // update burn flag field
    for (int j = 0; j < ny; j++){
        for (int i = 0; i < nx; i++){
            int idx = i + j*nx;       
            burn_flag[idx] = fire_cells[idx].state.burn_flag;
        }
    }
}


// Rothermel (1972) flame propgation model used for initial guess to Balbi
double Fire :: rothermel(FuelProperties* fuel, double max_wind, double tanphi,double fmc_g) {
    
    // fuel properties
    int savr          = fuel->savr;
    int fueldens      = fuel->fueldens;
    double st         = fuel->st;
    double se         = fuel->se;
    double fgi        = fuel->fgi;        
    double fuelmce    = fuel->fuelmce;
    double fuelheat   = fuel->fuelheat;
    double fueldepthm = fuel->fueldepthm;
    
    // local fire variables
    double bmst      = fmc_g/(1+fmc_g);
    double fuelloadm = (1.-bmst)*fgi;
    double fuelload  = fuelloadm*(pow(.3048, 2.0))*2.205;               // convert fuel load to lb/ft^2
    double fueldepth = fueldepthm/0.3048;                               // to ft
    double betafl    = fuelload/(fueldepth * fueldens);                 // packing ratio  jm: lb/ft^2/(ft * lb*ft^3) = 1
    double betaop    = 3.348 * pow(savr, -0.8189);                      // optimum packing ratio jm: units?? 
    double qig       = 250. + 1116.*fmc_g;                              // heat of preignition, btu/lb
    double epsilon   = exp(-138./savr );                                // effective heating number
    double rhob      = fuelload/fueldepth;                              // ovendry bulk density, lb/ft^3
    double rtemp2    = pow(savr, 1.5);
    double gammax    = rtemp2/(495. + 0.0594*rtemp2);                   // maximum rxn vel, 1/min
    double ar        = 1./(4.774 * (pow(savr, 0.1)) - 7.27);            // coef for optimum rxn vel
    double ratio     = betafl/betaop;   
    double gamma     = gammax*((pow(ratio, ar))*(exp(ar*(1.-ratio))));  // optimum rxn vel, 1/min
    double wn        = fuelload/(1 + st);                               // net fuel loading, lb/ft^2
    double rtemp1    = fmc_g/fuelmce;
    double etam      = 1.-2.59*rtemp1 +5.11*(pow(rtemp1, 2)) -3.52*(pow(rtemp1, 3));  // moist damp coef
    double etas      = 0.174* pow(se, -0.19);                           // mineral damping coef
    double ir        = gamma * wn * fuelheat * etam * etas;             // rxn intensity,btu/ft^2 min
    double xifr      = exp((0.792 + 0.681*(pow(savr, 0.5)))*(betafl+0.1))/(192.+0.2595*savr);// propagating flux ratio   
    double rothR0    = ir*xifr/(rhob*epsilon*qig);                      // SPREAD RATE [ft/s]
    double R0        = rothR0 * .005080;                                // SPREAD RATE [m/s]
    
    return R0;

}

// Balbi (2019) fire propagation model
struct Fire::FireProperties Fire :: balbi(FuelProperties* fuel,double u_mid, double v_mid, 
                                          double tanphi,double fmc_g) {
        
    // fuel properties
    double fgi        = fuel->fgi;              // initial total mass of surface fuel (kg/m**2)
    double fueldepthm = fuel->fueldepthm;       // fuel depth (m)
    int savr          = fuel->savr;             // fuel particle surface-area-to-volume ratio, 1/ft
    int cmbcnst       = fuel->cmbcnst;          // joules per kg of dry fuel
    
    // universal constants
    double g        = 9.81;                     // gravity 
    double pi       = 3.14159265358979323846;   // pi
    double s        = 17;                       // stoichiometric constant - Balbi 2018
    double Chi_0    = 0.3;                      // thin flame radiant fraction - ?Balbi 2009?
    double B        = 5.67e-8;                  // Stefan-Boltzman 
    double Deltah_v = 2.257e6;                  // water evap enthalpy [J/kg]
    double C_p      = 2e3;                      // calorific capacity [J/kg] - Balbi 2009  
    double C_pa     = 1150;                     // specific heat of air [J/Kg/K]
    double tau_0    = 75591;                    // residence time coefficient - Anderson 196?
    double tau      = 75591/(savr/0.3048);

    // fuel constants
    double m        = fmc_g;                    // fuel particle moisture content [0-1]
    double sigma    = fgi;                      // dead fuel load [kg/m^2]
    double sigmaT   = sigma;                    // total fuel load [kg/m^2]
    double rhoFuel  = 1500;                     // fuel density [kg/m^3]
    double T_i      = 600;                      // ignition temp [k]
    
    // model parameters
    double beta  = sigma/(fueldepthm*rhoFuel);  // packing ratio of dead fuel [eq.1]
    double betaT = sigmaT/(fueldepthm*rhoFuel); // total packing ratio [eq.2]
    double SAV   = savr/0.3048;                 // surface area to volume ratio [m^2/m^3]
    double lai   = (SAV*fueldepthm*beta)/2;     // leaf Area Index for dead fuel [eq.3]
    double nu    = fmin(2*lai,2*pi*beta/betaT); // absorption coefficient [eq.5]
    double lv    = fueldepthm;                  // fuel length [m] ?? need better parameterization here
    double K1    = 100;                         // drag force coefficient: 100 for field, 1400 for lab 
    double r_00  = 2.5e-5;                      // model parameter ??
    
    // Environmental Constants
    double rhoAir = 1.125;                      // air Density [Kg/m^3]
    double T_a    = 293.15;                     // air Temp [K]
    double alphax = atan(tanphi);               // slope angle [rad]
    double alphay = atan(tanphi);               // slope angle [rad]
    double psi    = 0;                          // angle between wind and flame front, assume parallel
    double phi    = 0;                          // angle between flame front vector and slope vector
    
    // Compute drag force coefficient [eq.7]
    double KDrag = K1*betaT*fmin(fueldepthm/lv,1);

    // Compute activation energy [eq.14]
    double q = C_p*(T_i - T_a) + m*Deltah_v; 

    // Compute radiant coefficient [eq.13]
    double A = fmin(SAV/(2*pi),beta/betaT)*Chi_0*cmbcnst/(4*q);
    
    // Initial quess = Rothermel ROS 
    double R = rothermel(fuel,max(u_mid,v_mid),tanphi,fmc_g);
    
    // Initial tilt angle guess = slope angle
    double gammax  = alphax;
    double gammay  = alphay;
    double maxIter = 100;
    double R_tol   = 1e-5;
    double iter    = 1;
    double error   = 1;
    double R_old   = R;
    
    // find spread rates
    double Chi,TFlame,u0,H,Hx,Hy,b;
    double ROSBase,ROSFlamex,ROSFlamey;
    double ROSConvx,ROSConvy,Rx,Ry;
    while (iter < maxIter && error > R_tol){
        
        // Calculate radiative fraction [eq.20]
        Chi = Chi_0/(1 + R*cos(gammax)/(SAV*r_00));
        
        // Compute flame Temp [eq.16]
        TFlame = T_a + cmbcnst*(1 - Chi)/((s+1)*C_pa);
        
        // Compute upward gas velocity [eq.19]
        u0 = 2*nu*((s+1)/tau_0)*(rhoFuel/rhoAir)*(TFlame/T_a);       
        
        // Calculate flame tilt angle (gammax)
        gammax = atan(tan(alphax)*cos(phi)+u_mid*cos(psi)/u0);
        gammay = atan(tan(alphay)*cos(phi)+v_mid*cos(psi)/u0);
        
        // Compute flame height [eq.17]
        Hx = u0*u0/(g*(TFlame/T_a - 1)*cos(alphax)*cos(alphax));
        Hy = u0*u0/(g*(TFlame/T_a - 1)*cos(alphay)*cos(alphay));
        H = fmax(Hx,Hy);
        
        // Compute convective coefficient [eq.8]
        b = 1/(q*tau_0*u0*betaT)*Deltah_v*nu*fmin(s/30,1);
        
        // Compute ROS
        // ROS from base radiation
        ROSBase = fmin(SAV*fueldepthm*betaT/pi,1)*(beta/betaT)*(beta/betaT)*(B*TFlame*TFlame*TFlame*TFlame)/(beta*rhoFuel*q);
        
        // ROS from flame radiation [eq.11]
        ROSFlamex = A*R*(1+sin(gammax) - cos (gammax))/(1 + R*cos(gammax)/(SAV*r_00));
        ROSFlamey = A*R*(1+sin(gammay) - cos (gammay))/(1 + R*cos(gammay)/(SAV*r_00));
        
        // ROS from convection
        ROSConvx = b*(tan(alphax) + 2*u_mid/u0*exp(-KDrag*R));
        ROSConvy = b*(tan(alphay) + 2*v_mid/u0*exp(-KDrag*R));
        
        // Total ROS 
        Rx    = ROSBase + ROSFlamex + ROSConvx;
        Ry    = ROSBase + ROSFlamey + ROSConvy;
        R     = sqrt(Rx*Rx+Ry*Ry);
        error = std::abs(R-R_old);
        R_old = R;
    }
    
    // define forward, backward spread in x, y diections
    double RxF, RxB, RyF, RyB;
    if (u_mid>0) {
        RxF = Rx;
        RxB = ROSBase;
    } else if (u_mid<0) {
        RxF = ROSBase;
        RxB = Rx;
    } else {
        RxF = Rx;
        RxB = Rx;
    }
    if (v_mid>0) {
        RyF = Ry;
        RyB = ROSBase;
    } else if (v_mid<0) {
        RyF = ROSBase;
        RyB = Ry;
    } else {
        RyF = Ry;
        RyB = Ry;
    }
        
    // calculate flame depth
    double L = R*tau;
    
    // struct to hold computed fire properties
    struct FireProperties fp;
    
    // set fire properties
    fp.w    = u0;
    fp.h    = H;
    fp.d    = L;
    fp.rxf  = RxF;
    fp.rxb  = RxB;
    fp.ryf  = RyF;
    fp.ryb  = RyB;
    fp.T    = TFlame;
    fp.tau  = tau;
    fp.K    = KDrag;
    
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
    vector_size  = {1, static_cast<unsigned long>(ny), static_cast<unsigned long>(nx)};
    
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