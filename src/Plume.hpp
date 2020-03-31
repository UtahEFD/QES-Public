//
//  Plume.hpp
//  
//  This class handles plume model
//

#ifndef PLUME_H
#define PLUME_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>

#include "util/calcTime.h"


//#include "NetCDFOutputGeneric.h"
#include "PlumeOutputLagrToEul.h"
#include "PlumeOutputLagrangian.h"

#include "PlumeInputData.hpp"
#include "Urb.hpp"
#include "Turb.hpp"
#include "Eulerian.h"
#include "Dispersion.h"


// LA do we need these here???
using namespace netCDF;
using namespace netCDF::exceptions;



class Plume {
  
    public:


        // constructor
        // first makes a copy of the urb grid number of values and the domain size as determined by dispersion
        // then sets up the concentration sampling box information for output
        // next copies important input time values and calculates needed time information 
        // lastly sets up the boundary condition functions and checks to make sure input BC's are valid
        Plume( PlumeInputData* PID,Urb* urb_ptr,Turb* turb_ptr,Eulerian* eul_ptr,Dispersion* dis_ptr, const bool& doLagrDataOutput_val,
               const bool& outputSimInfoFile_val,const std::string& outputFolder_val,const std::string& caseBaseName_val, const bool& debug_val);


        // this is the plume solver. It performs a time integration of the particle positions and particle velocity fluctuations
        // with calculations done on a per particle basis. During each iteration, temporary single value particle information
        // is taken from the overall particle list to do the calculations, with an update to the overall list particle information 
        // using the single value particle information at the end of the calculations.
        // Note that command line outputs are setup to not give all the information for each particle and timestep unless errors occur
        //  but are setup to be user controlled at input to limit the information to a sample of the particle information.
        // LA note: Outputting particle information for each particle for each timestep overwhelms the command line output file readers
        //  cause storing the command line output gets ridiculous.
        // LA future work: Need to add a CFL condition where the user specifies a courant number that varies from 0 to 1
        //  that is used to do an additional time remainder time integration loop for each particle, forcing particles to only
        //  move one cell at a time.
        // LA future work: Need to update the boundary condition functions again to be on a per cell basis if we want reflections
        //  reflections off of buildings and terrain. Could just use a ton of if statements, but it might be better to setup
        //  an array of boundary condition functions that are set at constructor time and accessed just by knowing the current
        //  and last particle indices of the Eulerian grid. Would also need another for loop for some of these reflective BCs
        //  to iterate until a distX has been completely travelled.
        void run(PlumeOutputLagrToEul* lagrToEulOutput,PlumeOutputLagrangian* lagrOutput);


    private:


        // types needed for implementing domainEdgeBC functions of varying functions
        typedef void (Plume::*xDomainEdgeBCptrFunction)( double& distX, double& distX_inc, double& xPos, const double& xPos_old, double& uFluct, double& uFluct_old, bool& isActive );
        typedef void (Plume::*yDomainEdgeBCptrFunction)( double& distY, double& distY_inc, double& yPos, const double& yPos_old, double& vFluct, double& vFluct_old, bool& isActive );
        typedef void (Plume::*zDomainEdgeBCptrFunction)( double& distZ, double& distZ_inc, double& zPos, const double& zPos_old, double& wFluct, double& wFluct_old, bool& isActive );

        // type needed for vector of pointer functions
        typedef void (Plume::*BCptrFunction)( double& distX, double& distY, double& distZ,
                                              double& distX_inc, double& distY_inc, double& distZ_inc,
                                              double& xPos, double& yPos, double& zPos, 
                                              const double& xPos_old, const double& yPos_old, const double& zPos_old, 
                                              double& uFluct, double& vFluct, double& wFluct, 
                                              double& uFluct_old, double& vFluct_old, double& wFluct_old, 
                                              bool& isActive, 
                                              xDomainEdgeBCptrFunction xDomainEdgeBC, yDomainEdgeBCptrFunction yDomainEdgeBC, 
                                              zDomainEdgeBCptrFunction zDomainEdgeBC );



        // pointers to the input classes to be set at constructor time
        // so they don't have to get passed in at the function run() anymore
        Urb* urb;
        Turb* turb;
        Eulerian* eul;
        Dispersion* dis;


        // Eulerian grid information
        // LA future work: this currently assumes urb and turb have the same grid and that will need to change quite soon.
        int nx;       // a copy of the urb grid nx value
        int ny;       // a copy of the urb grid ny value
        int nz;       // a copy of the urb grid nz value
        double dx;    // a copy of the urb grid dx value, eventually could become an array
        double dy;    // a copy of the urb grid dy value, eventually could become an array
        double dz;    // a copy of the urb grid dz value, eventually could become an array


        // these values are calculated from the urb and turb grids by dispersion
        // they are used for applying boundary conditions at the walls of the domain
        double domainXstart;    // the domain starting x value, a copy of the value found by dispersion
        double domainXend;      // the domain ending x value, a copy of the value found by dispersion
        double domainYstart;    // the domain starting y value, a copy of the value found by dispersion
        double domainYend;      // the domain ending y value, a copy of the value found by dispersion
        double domainZstart;    // the domain starting z value, a copy of the value found by dispersion
        double domainZend;      // the domain ending z value, a copy of the value found by dispersion


        // important time variables copied from dispersion (and some of them copied to dispersion from input)
        double sim_dt;       // the simulation timestep
        double simDur;   // the total amount of time to run the simulation for
        int nSimTimes; // this is the number of timesteps of the simulation, the calculated size of times
        std::vector<double> simTimes;  // this is the list of times for the simulation

        // important time variables not copied from dispersion
        double CourantNum;  // the Courant number, used to know how to divide up the simulation timestep into smaller per particle timesteps. Copied from input
        
        // copy of dispersion number of pars to release for each timestep,
        // used for updating the particle loop counter in Plume
        std::vector<int> nParsToRelease;


        double invarianceTol; // this is the tolerance used to determine whether makeRealizeable should be run on the stress tensor for a particle
        double C_0;           // used to separate out CoEps into its separate parts when doing debug output
        int updateFrequency_timeLoop;     // used to know how frequently to print out information during the time loop of the solver
        int updateFrequency_particleLoop; // used to know how frequently to print out information during the particle loop of the solver


        // some vars that are normally inside the particle and time loops, but are placed as data members
        // to allow access to them from any function inside the class
        // these include updateFrequency output as well as some of the loop variables themselves
        bool updateFrequency_timeLoop_output;   // if( (sim_tIdx+1) % updateFrequency_timeLoop == 0 || sim_tIdx == 0 || sim_tIdx == nSimTimes-2 )
        bool updateFrequency_particleLoop_output;   // if( parIdx % updateFrequency_particleLoop == 0 || parIdx == dis->pointList.size()-1 )
        int sim_tIdx;
        int parIdx;
        double timeRemainder;
        double par_time;
        bool extraDebug;

        // timer class useful for debugging and timing different operations
        calcTime timers;

        // copies of debug related information from the input arguments
        bool doLagrDataOutput;
        bool outputSimInfoFile;
        std::string outputFolder;
        std::string caseBaseName;
        bool debug;


        // function for calculating the individual particle timestep from the courant number, the current velocity fluctuations,
        // and the grid size. Forces particles to always move only at one timestep at at time.
        // Uses timeRemainder as the timestep if it is smaller than the one calculated from the Courant number
        double calcCourantTimestep(const double& uFluct,const double& vFluct,const double& wFluct,const double& timeRemainder);
        

        // utility functions for the plume solver
        // LA note: hm, this is the one place where it may be helpful to bring back in the complex data types
        //  at the same time, I kind of hate those complex datatypes, and it appears to be working
        void calcInvariants( const double& txx,const double& txy,const double& txz,
                             const double& tyy,const double& tyz,const double& tzz,
                             double& invar_xx,double& invar_yy,double& invar_zz);

        void makeRealizable(double& txx,double& txy,double& txz,double& tyy,double& tyz,double& tzz);

        void invert3( double& A_11,double& A_12,double& A_13,double& A_21,double& A_22,
                      double& A_23,double& A_31,double& A_32,double& A_33);
        void matmult( const double& A_11,const double& A_12,const double& A_13,
                      const double& A_21,const double& A_22,const double& A_23,
                      const double& A_31,const double& A_32,const double& A_33,
                      const double& b_11,const double& b_21,const double& b_31,
                      double& x_11, double& x_21, double& x_31);

        
        // BC function from input vars
        // set these during setBCfunctions
        bool doDepositions; // boolean for whether to do deposiitons or no
        
        // a function used at constructor time to set the pointer functions to the desired BC types
        void setBCfunctions( PlumeInputData* PID );

        // the overall vector of pointer functions set by setBCfunctions at constructor time
        std::vector<BCptrFunction> BCpointerFunctions;

        // the domain edge vectors of pointer functions set by setBCfunctions at constructor time
        // and called by each individual BCpointerFunction in BCpointerFunctions at plume run time
        std::vector<xDomainEdgeBCptrFunction> xDomainEdgePointerFunctions;
        std::vector<yDomainEdgeBCptrFunction> yDomainEdgePointerFunctions;
        std::vector<zDomainEdgeBCptrFunction> zDomainEdgePointerFunctions;

        
        // the domain start and end boundary condition functions to be pointed to by a given xDomainEdgeBCptrFunction variable
        // also made an empty one for use by cells that are not at the domain edges
        void xNotDomainEdgeBC( double& distX, double& distX_inc, double& xPos, const double& xPos_old, double& uFluct, double& uFluct_old, bool& isActive )
        {
            std::cerr << "ERROR (Plume::xNotDomainEdgeBC): this is a xNotDomainEdgeBC function that should not be used by anything that points to it. exiting program!" << std::endl;
            exit(EXIT_FAILURE);
        }
        // also made one for when particles move along a line along the domain wall plane
        void xDomainWallLineBC_passthrough( double& distX, double& distX_inc, double& xPos, const double& xPos_old, double& uFluct, double& uFluct_old, bool& isActive );
        void xDomainStartBC_exiting( double& distX, double& distX_inc, double& xPos, const double& xPos_old, double& uFluct, double& uFluct_old, bool& isActive );
        void xDomainEndBC_exiting( double& distX, double& distX_inc, double& xPos, const double& xPos_old, double& uFluct, double& uFluct_old, bool& isActive );
        void xDomainStartBC_periodic( double& distX, double& distX_inc, double& xPos, const double& xPos_old, double& uFluct, double& uFluct_old, bool& isActive );
        void xDomainEndBC_periodic( double& distX, double& distX_inc, double& xPos, const double& xPos_old, double& uFluct, double& uFluct_old, bool& isActive );
        void xDomainStartBC_reflection( double& distX, double& distX_inc, double& xPos, const double& xPos_old, double& uFluct, double& uFluct_old, bool& isActive );
        void xDomainEndBC_reflection( double& distX, double& distX_inc, double& xPos, const double& xPos_old, double& uFluct, double& uFluct_old, bool& isActive );
        // the domain start and end boundary condition functions to be pointed to by a given yDomainEdgeBCptrFunction variable
        // also made an empty one for use by cells that are not at the domain edges
        void yNotDomainEdgeBC( double& distY, double& distY_inc, double& yPos, const double& yPos_old, double& vFluct, double& vFluct_old, bool& isActive )
        {
            std::cerr << "ERROR (Plume::yNotDomainEdgeBC): this is a yNotDomainEdgeBC function that should not be used by anything that points to it. exiting program!" << std::endl;
            exit(EXIT_FAILURE);
        }
        // also made one for when particles move along a line along the domain wall plane
        void yDomainWallLineBC_passthrough( double& distY, double& distY_inc, double& yPos, const double& yPos_old, double& vFluct, double& vFluct_old, bool& isActive );
        void yDomainStartBC_exiting( double& distY, double& distY_inc, double& yPos, const double& yPos_old, double& vFluct, double& vFluct_old, bool& isActive );
        void yDomainEndBC_exiting( double& distY, double& distY_inc, double& yPos, const double& yPos_old, double& vFluct, double& vFluct_old, bool& isActive );
        void yDomainStartBC_periodic( double& distY, double& distY_inc, double& yPos, const double& yPos_old, double& vFluct, double& vFluct_old, bool& isActive );
        void yDomainEndBC_periodic( double& distY, double& distY_inc, double& yPos, const double& yPos_old, double& vFluct, double& vFluct_old, bool& isActive );
        void yDomainStartBC_reflection( double& distY, double& distY_inc, double& yPos, const double& yPos_old, double& vFluct, double& vFluct_old, bool& isActive );
        void yDomainEndBC_reflection( double& distY, double& distY_inc, double& yPos, const double& yPos_old, double& vFluct, double& vFluct_old, bool& isActive );
        // the domain start and end boundary condition functions to be pointed to by a given zDomainEdgeBCptrFunction variable
        // also made an empty one for use by cells that are not at the domain edges
        void zNotDomainEdgeBC( double& distZ, double& distZ_inc, double& zPos, const double& zPos_old, double& wFluct, double& wFluct_old, bool& isActive )
        {
            std::cerr << "ERROR (Plume::zNotDomainEdgeBC): this is a zNotDomainEdgeBC function that should not be used by anything that points to it. exiting program!" << std::endl;
            exit(EXIT_FAILURE);
        }
        // also made one for when particles move along a line along the domain wall plane
        void zDomainWallLineBC_passthrough( double& distZ, double& distZ_inc, double& zPos, const double& zPos_old, double& wFluct, double& wFluct_old, bool& isActive );
        void zDomainStartBC_exiting( double& distZ, double& distZ_inc, double& zPos, const double& zPos_old, double& wFluct, double& wFluct_old, bool& isActive );
        void zDomainEndBC_exiting( double& distZ, double& distZ_inc, double& zPos, const double& zPos_old, double& wFluct, double& wFluct_old, bool& isActive );
        void zDomainStartBC_periodic( double& distZ, double& distZ_inc, double& zPos, const double& zPos_old, double& wFluct, double& wFluct_old, bool& isActive );
        void zDomainEndBC_periodic( double& distZ, double& distZ_inc, double& zPos, const double& zPos_old, double& wFluct, double& wFluct_old, bool& isActive );
        void zDomainStartBC_reflection( double& distZ, double& distZ_inc, double& zPos, const double& zPos_old, double& wFluct, double& wFluct_old, bool& isActive );
        void zDomainEndBC_reflection( double& distZ, double& distZ_inc, double& zPos, const double& zPos_old, double& wFluct, double& wFluct_old, bool& isActive );

        // boundary condition function types to be pointed to by pointer functions in the BCpointerFunctions vector
        // so to be pointed to by a BCptrFunction type variable
        // each cell needs to point to one of these boundary condition functions, and the choice
        //  is determined by the function setBCfunctions() at constructor time.
        // LA-future work: this is starting to look like it would be better to do as dynamic polymorphism stuff
        //  kind of like is done for sources
        void domainEdgeBC( double& distX, double& distY, double& distZ,
                           double& distX_inc, double& distY_inc, double& distZ_inc,
                           double& xPos, double& yPos, double& zPos, 
                           const double& xPos_old, const double& yPos_old, const double& zPos_old, 
                           double& uFluct, double& vFluct, double& wFluct, 
                           double& uFluct_old, double& vFluct_old, double& wFluct_old, 
                           bool& isActive, 
                           xDomainEdgeBCptrFunction xDomainEdgeBC, yDomainEdgeBCptrFunction yDomainEdgeBC, 
                           zDomainEdgeBCptrFunction zDomainEdgeBC );
        // now the interior cell boundary condition functions, the ones chosen depending on the icellflag of the given cell
        void innerCellBC_passthrough( double& distX, double& distY, double& distZ,
                                      double& distX_inc, double& distY_inc, double& distZ_inc,
                                      double& xPos, double& yPos, double& zPos, 
                                      const double& xPos_old, const double& yPos_old, const double& zPos_old, 
                                      double& uFluct, double& vFluct, double& wFluct, 
                                      double& uFluct_old, double& vFluct_old, double& wFluct_old, 
                                      bool& isActive, 
                                      xDomainEdgeBCptrFunction xDomainEdgeBC, yDomainEdgeBCptrFunction yDomainEdgeBC, 
                                      zDomainEdgeBCptrFunction zDomainEdgeBC );
        void innerCellBC_simpleStairStepReflection( double& distX, double& distY, double& distZ,
                                                    double& distX_inc, double& distY_inc, double& distZ_inc,
                                                    double& xPos, double& yPos, double& zPos, 
                                                    const double& xPos_old, const double& yPos_old, const double& zPos_old, 
                                                    double& uFluct, double& vFluct, double& wFluct, 
                                                    double& uFluct_old, double& vFluct_old, double& wFluct_old, 
                                                    bool& isActive, 
                                                    xDomainEdgeBCptrFunction xDomainEdgeBC, yDomainEdgeBCptrFunction yDomainEdgeBC, 
                                                    zDomainEdgeBCptrFunction zDomainEdgeBC );
        // more will be coming soon



        // this is called to set the values whenever it is found that a particle is inactive or rogue
        void setFinishedParticleVals( double& xPos,double& yPos,double& zPos,bool& isActive,
                                      const bool& isRogue,
                                      const double& xPos_init, const double& yPos_init, const double& zPos_init);

        // this is for writing an output simulation info file separate from the regular command line output
        void writeSimInfoFile(Dispersion* dis,const double& current_time);


  
};
#endif
