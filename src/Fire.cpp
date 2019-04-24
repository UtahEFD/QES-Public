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
    k_start = baseHeight/dz;                
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
    
    //if (output_fields[0]=="all") {
    //    output_fields.erase(output_fields.begin());
    //    output_fields = {"u","v","w"};
    //}
    
    // set cell-centered dimensions
    const std::string tname = "t";
    NcDim t_dim = output->getDimension(tname);
    NcDim z_dim = output->getDimension("z");
    NcDim y_dim = output->getDimension("y");
    NcDim x_dim = output->getDimension("x");
    
    dim_scalar_t.push_back(t_dim);
    dim_scalar_t.push_back(y_dim);
    dim_scalar_t.push_back(x_dim);

    // create attributes
    AttVectorInt att_b = {&burn_flag,  "burn", "burn flag value", "--", dim_scalar_t};
    
    // map the name to attributes
    map_att_vector_int.emplace("burn", att_b);
      
    // we will always save time and grid lengths
    //output_vector_int.push_back(map_att_vector_int["burn"]);
    
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

void Fire :: run(Solver* solver) {
    
    // search predicate for burn state
    struct find_burn : std::unary_function<FireCell, bool> {
        int burn;
        find_burn(int burn):burn(burn) { }
        bool operator()(FireCell const& f) const {
            return f.state.burn_flag == burn;
        }
    };
    
    // get indices of burning cells
    std::vector<FireCell>::iterator it = std::find_if(fire_cells.begin(),fire_cells.end(),find_burn(1));
    
    while ( it != fire_cells.end()) {
        it = std::find_if (++it, fire_cells.end(),find_burn(1)); 
        //std::cout<<std::distance(fire_cells.begin(), it)<<std::endl;
    }
    FuelProperties* test = fire_cells[0].fuel;
    double R0 = rothermel(test, 0.0,0.0,0.0);
    FireProperties tester = balbi(test,0.0,0.0,0.0,0.0);
}

double Fire :: rothermel(FuelProperties* fuel, double max_wind, double tanphi,double fmc_g) {
    
    std::cout<<"Running Rothermel"<<std::endl;
    return 0.0;
}

struct Fire::FireProperties Fire :: balbi(FuelProperties* fuel,double u_mid, double v_mid, 
                                    double tanphi,double fmc_g) {
    
    FireProperties test;
    std::cout<<"Running Balbi"<<std::endl;
    return test;
    
}

//struct FireProperties Fire :: runFire(double u_mid, double v_mid, int type) {
//    
//    // get properties for fuel type
//    FuelProperties f = fuelProp(type);
//    
//    // Calculate ROS for 0 slope, 0 wind using Rothermel (1972)
//    double R0 = Rothermel(f.windrf, f.fgi, f.fueldepthm, f.savr, f.fuelmce, 
//                          f.fueldens, f.st, f.se, f.weight, f.fci_d, 
//                          f.fct, f.ichap, f.fci, f.fcbr, f.hfgl, 
//                          f.cmbcnst, f.fuelheat, f.fuelmc_g, f.fuelmc_c);
//
//    // Calculate ROS fusing Balbi (2009)
//    struct FireProperties fp = Balbi(f.fueldens, f.fueldepthm, f.fgi, f.savr, 
//                                     f.cmbcnst, u_mid, v_mid, f.slope, f.fuelmc_g, R0);
//
//    return fp; 
//}
//
//
//// Calculate ROS for zero slope, zero wind using Rothermel (1972) model
//double Fire :: Rothermel(double windrf, double fgi, double fueldepthm, double savr, double fuelmce, 
//                         double fueldens, double st, double se, double weight, double fci_d, 
//                         double fct, double ichap, double fci, double fcbr, double hfgl, 
//                         double cmbcnst, double fuelheat, double fuelmc_g, double fuelmc_c){
//   
//    double bmst        = fuelmc_g/(1+fuelmc_g);
//    double fuelloadm   = (1.-bmst)*fgi;
//    double fuelload    = fuelloadm*(pow(.3048, 2.0))*2.205;             // convert fuel load to lb/ft^2
//    double fueldepth   = fueldepthm/0.3048;                             // to ft
//    double betafl      = fuelload/(fueldepth * fueldens);               // packing ratio  jm: lb/ft^2/(ft * lb*ft^3) = 1
//    double betaop      = 3.348 * pow(savr, -0.8189);                    // optimum packing ratio jm: units?? 
//    double qig         = 250. + 1116.*fuelmc_g;                         // heat of preignition, btu/lb
//    double epsilon     = exp(-138./savr );                              // effective heating number
//    double rhob        = fuelload/fueldepth;                            // ovendry bulk density, lb/ft^3
//    double rtemp2      = pow(savr, 1.5);
//    double gammax      = rtemp2/(495. + 0.0594*rtemp2);                 // maximum rxn vel, 1/min
//    double ar          = 1./(4.774 * (pow(savr, 0.1)) - 7.27);          // coef for optimum rxn vel
//    double ratio       = betafl/betaop;   
//    double gamma       = gammax*((pow(ratio, ar))*(exp(ar*(1.-ratio))));// optimum rxn vel, 1/min
//    double wn          = fuelload/(1 + st);                             // net fuel loading, lb/ft^2
//    double rtemp1      = fuelmc_g/fuelmce;
//    double etam        = 1.-2.59*rtemp1 +5.11*(pow(rtemp1, 2)) -3.52*(pow(rtemp1, 3));  // moist damp coef
//    double etas        = 0.174* pow(se, -0.19);                         // mineral damping coef
//    double ir          = gamma * wn * fuelheat * etam * etas;           // rxn intensity,btu/ft^2 min
//    double xifr        = exp((0.792 + 0.681*(pow(savr, 0.5))) * (betafl+0.1))/(192. + 0.2595*savr);   // propagating flux ratio   
//    
//    double rothR0 = ir*xifr/(rhob*epsilon*qig); // SPREAD RATE [ft/s]
//    double R0 = rothR0 * .005080;               // SPREAD RATE [m/s]
//
//    return R0;
//}
//
//struct FireProperties Fire :: Balbi(double fueldens, double fueldepthm, double fgi, double savr, 
//                                    double cmbcnst, double u_mid, double v_mid, double slope, double fuelmc_g, double R0){
//    
//    struct FireProperties fp;
//    
//    // Universal constants
//    double g = 9.81;                       // Gravity 
//    double pi = 3.14159265358979323846;    // pi
//    //double s = 9;                          // Stoichiometric constant - Balbi 2009
//    double s = 17;                         // Stoichiometric constant - Balbi 2018
//    double Chi_0 = 0.3;                    // Thin Flame Radiant Fraction - ?Balbi 2009?
//    double a = 0.05;                       // Constant from Balbi 2008
//    double A_0 = 2.25;                     // Constant from Balbi 2008
//    double eps = 0.2;                      // Pastor 2002
//    double B = 5.67e-8;                    // Stefan-Boltzman 
//    double Deltah_v = 2.257e6;             // Water Evap Enthalpy [J/kg]
//    double C_p = 2e3;                      // Calorific Capacity [J/kg] - Balbi 2009  
//    double C_pa = 1150;                    // Specific heat of air [J/Kg/K]
//    double tau_0 = 75591;                  // Residence time coefficient - Anderson 196?
//    double tau = 75591/(savr/0.3048);
//
//    // Fuel Constants
//    double m = fuelmc_g;                   // FUEL PARTICLE MOISTURE CONTENT [0-1]
//    double rho_v = fueldens*16.0185;       // FUEL Particle Density [Kg/m^3]
//    double sigma = fgi;                    // Dead fuel load [Kg/m^2]
//    double sigmaT = sigma;                 // Total fuel load [Kg/m^2]
//    double e_delta = (savr/0.3048)*sigma/(4*rho_v);
//    double rhoFuel = 1500;                 // Fuel Density [Kg/m^3]
//    double rhoFlame = 0.25;                // Gas Flame Density [kg/m^3]
//    double T_i = 600;                      // Ignition temp [K]
//    
//    // Model Parameters
//    double beta = sigma/(fueldepthm*rhoFuel);     // Packing ratio of dead fuel [eq.1]
//    double betaT = sigmaT/(fueldepthm*rhoFuel);      // Total packing ratio [eq.2]
//    double SAV = savr/0.3048;                       // Surface area to volume ratio [m^2/m^3]
//    double lai = (SAV*fueldepthm*beta)/2;           // Leaf Area Index for dead fuel [eq.3]
//    double laiT = (SAV*fueldepthm*betaT)/2;         // Total fuel LAI [eq.4]
//    double nu = fmin(2*lai,2*pi*beta/betaT);        // Absorption coefficient [eq.5]
//    double lv = fueldepthm;                         // fuel length [m] ?? need better parameterization here
//    double K1 = 100;                                // drag force coefficient: 100 for field, 1400 for lab 
//    double r_00 = 2.5e-5;                           // Model parameter ??
//    
//    // Environmental Constants
//    double rhoAir = 1.125;                  // Air Density [Kg/m^3]
//    double T_a = 293.15;                    // Air Temp [K]
//    double alphax = atan(slope);             // Slope angle [rad]
//    double alphay = atan(slope);
//    double psi = 0;                         // Angle between wind and flame front, assume parallel
//    double phi = 0;                         // Angle between flame front vector and slope vector
//
//    // Compute drag force coefficient [eq.7]
//    double KDrag = K1*betaT*fmin(fueldepthm/lv,1);
//
//    // Compute activation energy [eq.14]
//    double q = C_p*(T_i - T_a) + m*Deltah_v; 
//
//    // Compute radiant coefficient [eq.13]
//    double A = fmin(SAV/(2*pi),beta/betaT)*Chi_0*cmbcnst/(4*q);
//    
//    // Initial quess = Rothermel ROS 
//    double R = R0;
//    double Rx, Ry;
//    
//    // Initial tilt angle guess = slope angle
//    double gammax = alphax;
//    double gammay = alphay;
//    double maxIter = 100;
//    double R_tol = 1e-5;
//    double iter = 1;
//    double error = 1;
//    double R_old = R;
//    double Chi,TFlame,u0,H,Hx,Hy;
//    while (iter < maxIter && error > R_tol){
//        // Calculate radiative fraction [eq.20]
//        double Chi = Chi_0/(1 + R*cos(gammax)/(SAV*r_00));
//        // Compute flame Temp [eq.16]
//        double TFlame = T_a + cmbcnst*(1 - Chi)/((s+1)*C_pa);
//        // Compute upward gas velocity [eq.19]
//        double u0 = 2*nu*((s+1)/tau_0)*(rhoFuel/rhoAir)*(TFlame/T_a);       
//        // Calculate flame tilt angle (gammax)
//        double gammax = atan(tan(alphax)*cos(phi)+u_mid*cos(psi)/u0);
//        double gammay = atan(tan(alphay)*cos(phi)+v_mid*cos(psi)/u0);
//        // Compute flame height [eq.17]
//        Hx = u0*u0/(g*(TFlame/T_a - 1)*cos(alphax)*cos(alphax));
//        Hy = u0*u0/(g*(TFlame/T_a - 1)*cos(alphay)*cos(alphay));
//        H = fmax(Hx,Hy);
//        // Compute convective coefficient [eq.8]
//        double b = 1/(q*tau_0*u0*betaT)*Deltah_v*nu*fmin(s/30,1);
//        // Compute ROS
//        // ROS from base radiation
//        double ROSBase = fmin(SAV*fueldepthm*betaT/pi,1)*(beta/betaT)*(beta/betaT)*(B*TFlame*TFlame*TFlame*TFlame)/(beta*rhoFuel*q);
//        // ROS from flame radiation [eq.11]
//        double ROSFlamex = A*R*(1+sin(gammax) - cos (gammax))/(1 + R*cos(gammax)/(SAV*r_00));
//        double ROSFlamey = A*R*(1+sin(gammay) - cos (gammay))/(1 + R*cos(gammay)/(SAV*r_00));
//        // ROS from convection
//        double ROSConvx = b*(tan(alphax) + 2*u_mid/u0*exp(-KDrag*R));
//        double ROSConvy = b*(tan(alphay) + 2*v_mid/u0*exp(-KDrag*R));
//        // Total ROS 
//        Rx = ROSBase + ROSFlamex + ROSConvx;
//        Ry = ROSBase + ROSFlamey + ROSConvy;
//        R = sqrt(Rx*Rx+Ry*Ry);
//        error = std::abs(R-R_old);
//        R_old = R;
//    }
//    
//    // Calculate Flame Depth
//    double L = R*tau;
//    
//    fp.w   = u0;
//    fp.h   = H;
//    fp.rx  = Rx;
//    fp.ry  = Ry;
//    fp.T   = TFlame;
//    fp.tau = tau;
//    
//    return fp;
//}

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
    time = (double)output_counter;
	
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