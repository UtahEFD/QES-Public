//
//  Dispersion.cpp
//  
//  This class handles dispersion information
//

#include "Dispersion.h"
#include <fstream>
#define EPSILON 0.00001   

void dispersion::createDisp(const eulerian& e) {
    eul=e;
    xSrc=eul.utl.xSrc;
    ySrc=eul.utl.ySrc;
    zSrc=eul.utl.zSrc;
    int nx=eul.utl.nx;
    int ny=eul.utl.ny;
    
    numPar=eul.utl.theight*eul.utl.twidth;
    numTimeStep=ceil(eul.utl.dur/eul.utl.timeStep);
    std::cout<<"first resize"<<std::endl;
    timeStepStamp.resize(numTimeStep);
    std::cout<<"dt:"<<std::endl;
    double dt=eul.utl.timeStep;
    std::cout<<"timeStap"<<std::endl;
    
    for(int i=0;i<numTimeStep;++i)
        timeStepStamp.at(i)=i*dt+dt;
        
    double dur=eul.utl.dur;
    pos.resize(numPar);
    prime.resize(numPar);
    int id=int(zSrc)*ny*nx+int(ySrc)*nx+int(xSrc);
    std::cout<<"primes"<<numPar<<std::endl;
    for(int i=0;i<numPar;i++){
        pos.at(i).x=xSrc;
        pos.at(i).y=ySrc;
        pos.at(i).z=zSrc;
    
        double rann=random::norRan();
        prime.at(i).x=eul.sig.at(id).e11 * rann;
        rann=random::norRan();
        prime.at(i).y=eul.sig.at(id).e22 * rann;
        rann=random::norRan();
        prime.at(i).z=eul.sig.at(id).e33 * rann;//id doesnt change here  id=8458
    }
    std::cout<<"primes end"<<std::endl;
    tStrt.resize(numPar);
    
    /*particles released per time step
      (integer-as number of particles cannot be a fraction)*/
    parPerTimestep=numPar*dt/dur; 
                                  
    std::cout<<"Emitting "<<parPerTimestep<< " particles per Time Step"<<std::endl;
    
    int parRel=0; //number of particles released in a particluar time step
    double startTime=dt;
    
    
    for(int i=0;i<numPar;i++){
      /*when number of particles to be released in a particluar time step reaches total 
        number of particles to be released in that time step, then increase the start time 
        by timestep and set parRel=0*/
        if(parRel==parPerTimestep) {
            startTime=startTime+dt;
            parRel=0;
        }
        tStrt.at(i)=startTime;
        ++parRel;
    }
    
    /*
      Checking if the starting time for the last particle is equal to the duration of
      the simulation (for continous release ONLY)
    */
    if (fabs(tStrt.back()-dur)>EPSILON) {
        std::cerr<<" Error, in start time of the particles"<<std::endl;
        exit(1);
    }
}