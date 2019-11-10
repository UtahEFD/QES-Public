
#include "Source_continuous_uniform_point.hpp"


std::vector<pointInfo> Source_continuous_uniform_point::outputPointInfo(const double& dt,const double& simDur)
{

    // create the output storage values
    std::vector<pointInfo> outputVect;



    // setup the number of particles per timestep
    // this should eventually be adjusting this value, not adding to it
    int parPerTimestep = numParticles*dt/simDur;
    std::cout << "[sourcePoint] \t Emitting " << parPerTimestep << " particles per Time Step" << std::endl;

    // counter variables to help get different times for each set of particles
    int parRel = 0;
    double startTime = dt;
    

    // now set output positions and times
    for(int i = 0; i < numParticles; i++)
    {

        // need to create the single pointInfo storage value that will be pointed to
        pointInfo current_pointInfo;

        // set the source positions for each particle
        current_pointInfo.pos.e11 = posX;
        current_pointInfo.pos.e21 = posY;
        current_pointInfo.pos.e31 = posZ;
        
        // set the source times for each particle
        /*when number of particles to be released in a particluar time step reaches total 
        number of particles to be released in that time step, then increase the start time 
        by timestep and set parRel=0*/
        if( parRel == parPerTimestep )
        {
            startTime = startTime + dt;
            parRel = 0;
        }
        current_pointInfo.tStrt = startTime;
        ++parRel;

        
        
        // now that all the values for the current particle have been set, need to set a pointer, and stuff it into the output vector
        // how the heck do I do correct syntax for this? I'm just going to do something, even if it causes memory leak problems, then ask Pete or look up later
        // actually, going to attempt to do this not as a pointer, just stuff it into the vector to see if it works
        outputVect.push_back(current_pointInfo);

    }

    return outputVect;

}