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
#include <list>
#include <cmath>
#include <cstring>

#include "util/calcTime.h"
#include "Vector3.h"
#include "Matrix3.h"
#include "Random.h"

#include "QESNetCDFOutput.h"
#include "PlumeOutputLagrToEul.h"
#include "PlumeOutputLagrangian.h"

#include "Args.hpp"
#include "PlumeInputData.hpp"

#include "URBGeneralData.h"
#include "TURBGeneralData.h"
#include "Eulerian.h"

#include "Particle.hpp"

#include "SourcePoint.hpp"
#include "SourceLine.hpp"
#include "SourceCircle.hpp"
#include "SourceCube.hpp"
#include "SourceFullDomain.hpp"

class Plume {
    
public:
        
    // constructor
    // first makes a copy of the urb grid number of values and the domain size as determined by dispersion
    // then sets up the concentration sampling box information for output
    // next copies important input time values and calculates needed time information 
    // lastly sets up the boundary condition functions and checks to make sure input BC's are valid
    Plume( PlumeInputData*, URBGeneralData*, TURBGeneralData*, Eulerian*, Args*); 

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
    void run(URBGeneralData*,TURBGeneralData*,Eulerian*,std::vector<QESNetCDFOutput*> );
    
    // This the storage for all particles
    // the sources can set these values, then the other values are set using urb and turb info using these values
    std::list<Particle> pointList;
    
    // this is the total number of particles expected to be released during the simulation
    // !!! this has to be calculated carefully inside the getInputSources() function
    int totalParsToRelease;

private:

    // Eulerian grid information
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
    double sim_dt;     // the simulation timestep
    double simDur;     // the total amount of time to run the simulation for
    int nSimTimes;     // this is the number of timesteps of the simulation, the calculated size of times
    std::vector<double> simTimes;  // this is the list of times for the simulation

    // some overall metadata for the set of particles
    int isRogueCount;        // just a total number of rogue particles per time iteration
    int isNotActiveCount;    // just a total number of inactive active particles per time iteration

    // important time variables not copied from dispersion
    double CourantNum;  // the Courant number, used to know how to divide up the simulation timestep into smaller per particle timesteps. Copied from input
    double vel_threshold;
    
    // ALL Sources that will be used 
    std::vector< SourceKind* > allSources;
    // this is the global counter of particles released (used to set particleID)
    int nParsReleased;
    
    double invarianceTol; // this is the tolerance used to determine whether makeRealizeable should be run on the stress tensor for a particle
    double C_0;           // used to separate out CoEps into its separate parts when doing debug output
    int updateFrequency_timeLoop;     // used to know how frequently to print out information during the time loop of the solver
    int updateFrequency_particleLoop; // used to know how frequently to print out information during the particle loop of the solver

    // timer class useful for debugging and timing different operations
    calcTime timers;

    // copies of debug related information from the input arguments
    bool doLagrDataOutput;
    bool outputSimInfoFile;
    std::string outputFolder;
    std::string caseBaseName;
    bool debug;
    
    void setParticleVals(TURBGeneralData* TGD, Eulerian* eul, std::list<Particle>& newParticles);
    // this function gets sources from input data and adds them to the allSources vector
    // this function also calls the many check and calc functions for all the input sources
    // !!! note that these check and calc functions have to be called here 
    //  because each source requires extra data not found in the individual source data
    // !!! totalParsToRelease needs calculated very carefully here using information from each of the sources
    void getInputSources(PlumeInputData* PID);

    // this function generates the list of particle to be released at a given time
    int generateParticleList(float,TURBGeneralData*, Eulerian*);

    // this function scrubs the inactive particle for the particle list (pointList)
    void scrubParticleList();
    

    // this function moves (advects) one particle
    void advectParticle(int&, std::list<Particle>::iterator, URBGeneralData*, TURBGeneralData*, Eulerian*);

    /* reflection functions in WallReflection.cpp */
    // main function pointer
    bool (Plume::*wallReflection)(URBGeneralData* UGD, Eulerian* eul,
                                  double& xPos, double& yPos, double& zPos, 
                                  double& disX, double& disY, double& disZ,
                                  double& uFluct, double& vFluct, double& wFluct);
    // reflection on walls (stair step)
    bool wallReflectionFullStairStep(URBGeneralData* UGD, Eulerian* eul,
                                     double& xPos, double& yPos, double& zPos, 
                                     double& disX, double& disY, double& disZ,
                                     double& uFluct, double& vFluct, double& wFluct);
    // reflection -> set particle inactive when entering a wall
    bool wallReflectionSetToInactive(URBGeneralData* UGD, Eulerian* eul,
                                     double& xPos, double& yPos, double& zPos, 
                                     double& disX, double& disY, double& disZ,
                                     double& uFluct, double& vFluct, double& wFluct);
    // reflection -> this function will do nothing 
    bool wallReflectionDoNothing(URBGeneralData* UGD, Eulerian* eul,
                                 double& xPos, double& yPos, double& zPos, 
                                 double& disX, double& disY, double& disZ,
                                 double& uFluct, double& vFluct, double& wFluct);
    
    
    // function for calculating the individual particle timestep from the courant number, the current velocity fluctuations,
    // and the grid size. Forces particles to always move only at one timestep at at time.
    // Uses timeRemainder as the timestep if it is smaller than the one calculated from the Courant number
    double calcCourantTimestep(const double& uFluct,const double& vFluct,const double& wFluct,const double& timeRemainder);
        

    // utility functions for the plume solver
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
    
    // a function used at constructor time to set the pointer function to the desired BC type
    void setBCfunctions(std::string xBCtype,std::string yBCtype,std::string zBCtype);   

    // A pointer to the wallBC function for the x direction. 
    // Which function it points at is determined by setBCfunctions and the input xBCtype
    void (Plume::*enforceWallBCs_x)( double& pos,double& velFluct,double& velFluct_old,bool& isActive,
                                     const double& domainStart,const double& domainEnd);  
    // A pointer to the wallBC function for the y direction. 
    // Which function it points at is determined by setBCfunctions and the input yBCtype
    void (Plume::*enforceWallBCs_y)( double& pos,double& velFluct,double& velFluct_old,bool& isActive,
                                     const double& domainStart,const double& domainEnd);  
    // A pointer to the wallBC function for the z direction. 
    // Which function it points at is determined by setBCfunctions and the input zBCtype
    void (Plume::*enforceWallBCs_z)( double& pos,double& velFluct,double& velFluct_old,bool& isActive,
                                     const double& domainStart,const double& domainEnd);  
    // Boundary condition functions:
    void enforceWallBCs_exiting( double& pos,double& velFluct,double& velFluct_old,bool& isActive,
                                 const double& domainStart,const double& domainEnd);
    void enforceWallBCs_periodic( double& pos,double& velFluct,double& velFluct_old,bool& isActive,
                                  const double& domainStart,const double& domainEnd);
    void enforceWallBCs_reflection( double& pos,double& velFluct,double& velFluct_old,bool& isActive,
                                    const double& domainStart,const double& domainEnd);
    
    // this is called to set the values whenever it is found that a particle is inactive or rogue
    void setFinishedParticleVals( double& xPos,double& yPos,double& zPos,bool& isActive,
                                  const bool& isRogue,
                                  const double& xPos_init, const double& yPos_init, const double& zPos_init);
    
    // this is for writing an output simulation info file separate from the regular command line output
    void writeSimInfoFile(const double& current_time);

};

#endif

