//
//  NetCDFOutput.h
//  
//  This class handles saving output files for Eulerian binned Lagrangian particle data,
//   where this class handles the binning of the Lagrangian particle data
//  This is a specialized output class derived 
//   and inheriting from QESNetCDFOutput.
//
//  Created by Fabien Margairaz on 01/25/20
//  Modified by Loren Atwood 02/08/20
//


#include "PlumeOutput.h"
#include "Plume.hpp"

// note that this sets the output file and the bool for whether to do output, in the netcdf inherited classes
// in this case, output should always be done, so the bool for whether to do output is set to true
PlumeOutput::PlumeOutput(PlumeInputData* PID,WINDSGeneralData* WGD,Plume* plume_ptr,std::string output_file)
  : QESNetCDFOutput(output_file)
{

    std::cout << "[PlumeOutput] set up NetCDF file " << output_file << std::endl;

    // this is the current simulation start time
    // LA note: this would need adjusted if we ever change the 
    //  input simulation parameters to include a simStartTime
    //  instead of just using simDur
    float simStartTime = 0.0;
    
    // setup output frequency control information
    // time to start concentration averaging, not the time to start output.
    //Adjusted if the time averaging duration does not divide evenly by the averaging frequency
    timeAvgStart = PID->colParams->timeAvgStart;
    // time to end concentration averaging and output. Notice that this is now always the simulation end time
    timeAvgEnd = PID->simParams->simDur;
    // time averaging frequency and output frequency
    timeAvgFreq = PID->colParams->timeAvgFreq;          
    
    // !!! Because collection parameters could not know anything about simulation duration at parse time,
    //  need to make this check now
    // Make sure the timeAvgStart is not greater than the simulation end time
    if( timeAvgStart > PID->simParams->simDur ) {
        std::cerr << "[PlumeOutput] ERROR "
                  << "(CollectionParameters checked during PlumeOutput) "
                  << "input timeAvgStart must be smaller than or equal to the input simulation duration!" << std::endl;
        std::cerr << " timeAvgStart = \"" << timeAvgStart << "\", simDur = \"" << PID->simParams->simDur << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }

    // !!! Because collection parameters could not know anything 
    //  about the simulation duration at parse time, need to make this check now
    // Make sure timeAvgFreq is not bigger than the simulation duration
    // LA note: timeAvgFreq can be as big as the collection duration, or even smaller than the collection duration
    //  IF timeAvgFreq is at least the same size or smaller than the simulation duration
    if( timeAvgFreq > PID->simParams->simDur ) {
        std::cerr << "[PlumeOutput] ERROR "
                  << "(CollectionParameters checked during PlumeOutput): "
                  << "input timeAvgFreq must be smaller than or equal to the input simulation duration!" << std::endl;
        std::cerr << " timeAvgFreq = \"" << timeAvgFreq << "\", simDur = \"" << PID->simParams->simDur << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }

    
    // Determine whether timeAvgStart needs adjusted to make the time average duration divide evenly by the averaging frequency
    // This is essentially always keeping the timeAvgEnd at what it is (end of the simulation), and adjusting the timeAvgStart
    // and outputStartTime to avoid slicing off an averaging and output time unless we have to.
    float avgDur = timeAvgEnd - timeAvgStart;
    // if doesn't divide evenly, need to adjust timeAvgStart
    // can determine if it divides evenly by comparing the quotient with the decimal division result
    //  if the values do not match, the division has a remainder
    //  here's hoping numerical error doesn't play a role
    float quotient = std::floor(avgDur/timeAvgFreq);
    float decDivResult = avgDur/timeAvgFreq;
    if( quotient != decDivResult ) {
        // clever algorythm that always gets the exact number of time averages (and outputs) 
        // when the time averaging duration divides evenly by the time averaging frequency
        // and rounds the number of time averages down to what it would be if the start time 
        // were the next smallest evenly dividing number
        int nAvgs = std::floor(avgDur/timeAvgFreq);
    
        // clever algorythm to always calculate the desired averaging start time based off the number of time averages
        // the timeAvgStart if not adjusting nAvgs
        float current_timeAvgStart = timeAvgEnd - timeAvgFreq*(nAvgs);
        // the timeAvgStart if adjusting nAvgs. Note nAvgs has one extra averaging period
        float adjusted_timeAvgStart = timeAvgEnd - timeAvgFreq*(nAvgs+1);
        if( adjusted_timeAvgStart >= simStartTime ) {
            // need to adjust the timeAvgStart to be the adjustedTimeAvgStart
            // warn the user that the timeAvgStart is being adjusted before adjusting timeAvgStart
            std::cout << "[PlumeOutput] "
                      << "adjusting timeAvgStart because time averaging duration did not divide evenly by timeAvgFreq" << std::endl;
            std::cout << "  original timeAvgStart = \"" << timeAvgStart
                      << "\", timeAvgEnd = \"" << timeAvgEnd 
                      << "\", timeAvgFreq = \"" << timeAvgFreq
                      << "\", new timeAvgStart = \"" << adjusted_timeAvgStart << "\"" << std::endl;
            timeAvgStart = adjusted_timeAvgStart;
        } else {
            // need to adjust the timeAvgStart to be the currentTimeAvgStart
            // warn the user that the timeAvgStart is being adjusted before adjusting timeAvgStart
            std::cout << "[PlumeOutput] "
                      << "adjusting timeAvgStart because time averaging duration did not divide evenly by timeAvgFreq" << std::endl;
            std::cout << "  original timeAvgStart = \"" << timeAvgStart
                      << "\", timeAvgEnd = \"" << timeAvgEnd 
                      << "\", timeAvgFreq = \"" << timeAvgFreq
                      << "\", new timeAvgStart = \"" << current_timeAvgStart << "\"" << std::endl;
            timeAvgStart = current_timeAvgStart;
        }
    } // else does divide evenly, no need to adjust anything so no else


    // set the initial next output time value
    nextOutputTime = timeAvgStart + timeAvgFreq;
    
    
    // setup copy of plume pointer so output data can be grabbed directly
    plume = plume_ptr;

    
    // need nx, ny, nz of the domain to make sure the output handles domains that are not three dimensional
    // for now these are a copy of the input urb values
    nx = WGD->nx;
    ny = WGD->ny;
    nz = WGD->nz;
    
    // need the simulation timeStep for use in concentration averaging
    timeStep = PID->simParams->timeStep;
    
    
    // --------------------------------------------------------
    // setup information: sampling box/concentration
    // --------------------------------------------------------
    
    // Sampling box variables for calculating concentration data
    nBoxesX = PID->colParams->nBoxesX;
    nBoxesY = PID->colParams->nBoxesY;
    nBoxesZ = PID->colParams->nBoxesZ;

    lBndx = PID->colParams->boxBoundsX1;
    uBndx = PID->colParams->boxBoundsX2;
    lBndy = PID->colParams->boxBoundsY1;
    uBndy = PID->colParams->boxBoundsY2;
    lBndz = PID->colParams->boxBoundsZ1;
    uBndz = PID->colParams->boxBoundsZ2;

    boxSizeX = (uBndx-lBndx)/(nBoxesX);
    boxSizeY = (uBndy-lBndy)/(nBoxesY);
    boxSizeZ = (uBndz-lBndz)/(nBoxesZ);

    volume = boxSizeX*boxSizeY*boxSizeZ;

    // output concentration storage variables
    xBoxCen.resize(nBoxesX);
    yBoxCen.resize(nBoxesY);
    zBoxCen.resize(nBoxesZ);
    int zR = 0, yR = 0, xR = 0;
    for(int k = 0; k < nBoxesZ; ++k) {
        zBoxCen.at(k) = lBndz + (zR*boxSizeZ) + (boxSizeZ/2.0);
        zR++;
    }
    for(int j = 0; j < nBoxesY; ++j) {
        yBoxCen.at(j) = lBndy + (yR*boxSizeY) + (boxSizeY/2.0);
        yR++;
    }
    for(int i = 0; i <nBoxesX; ++i) {
        xBoxCen.at(i) = lBndx + (xR*boxSizeX) + (boxSizeX/2.0);
        xR++;
    }

    // initialization of the container
    pBox.resize(nBoxesX*nBoxesY*nBoxesZ,0);
    conc.resize(nBoxesX*nBoxesY*nBoxesZ,0.0);

    // --------------------------------------------------------
    // setup information:  
    // --------------------------------------------------------
    
    int nbrFace = WGD->wall_below_indices.size() + WGD->wall_above_indices.size() +
        WGD->wall_back_indices.size() + WGD->wall_front_indices.size() +
        WGD->wall_left_indices.size() + WGD->wall_right_indices.size();
    


    // --------------------------------------------------------
    // setup the netcdf output information storage
    // --------------------------------------------------------

    // setup desired output fields string
    // LA future work: can be added in fileOptions at some point
    output_fields = {"t","x","y","z","pBox","conc"};

    // set data dimensions, which in this case are cell-centered dimensions
    // time dimension
    NcDim NcDim_t = addDimension("t");
    // space dimensions
    NcDim NcDim_x = addDimension("x",nBoxesX);
    NcDim NcDim_y = addDimension("y",nBoxesY);
    NcDim NcDim_z = addDimension("z",nBoxesZ);

    // create attributes for time dimension
    std::vector<NcDim> dim_vect_t;
    dim_vect_t.push_back(NcDim_t);
    createAttScalar("t","time","s",dim_vect_t,&time);

    // create attributes space dimensions
    std::vector<NcDim> dim_vect_x;
    dim_vect_x.push_back(NcDim_x);
    createAttVector("x","x-center collection box","m",dim_vect_x,&xBoxCen);
    std::vector<NcDim> dim_vect_y;
    dim_vect_y.push_back(NcDim_y);
    createAttVector("y","y-center collection box","m",dim_vect_y,&yBoxCen);
    std::vector<NcDim> dim_vect_z;
    dim_vect_z.push_back(NcDim_z);
    createAttVector("z","z-center collection box","m",dim_vect_z,&zBoxCen);

    // create 3D vector and put in the dimensions (nt,nz,ny,nx).
    // !!! make sure the order is specificall nt,nz,ny,nx in this spot,
    //  the order doesn't seem to matter for other spots
    std::vector<NcDim> dim_vect_3d;
    dim_vect_3d.push_back(NcDim_t);
    dim_vect_3d.push_back(NcDim_z);
    dim_vect_3d.push_back(NcDim_y);
    dim_vect_3d.push_back(NcDim_x);
    
    
    // create attributes for all output information
    createAttVector("pBox","number of particle per box","#ofPar",dim_vect_3d,&pBox);
    createAttVector("conc","concentration","g m-3",dim_vect_3d,&conc);    

     // face dimensions
    NcDim NcDim_nFace = addDimension("nFace",nbrFace);
    //NcDim NcDim_x = addDimension("x",nBoxesX);
    //NcDim NcDim_y = addDimension("y",nBoxesY);
    //NcDim NcDim_z = addDimension("z",nBoxesZ);

    // create attributes space dimensions
    std::vector<NcDim> dim_vect_face;
    dim_vect_face.push_back(NcDim_nFace);
    //createAttVector("xface","x-face","m",dim_vect_face,&xBoxCen);
    //createAttVector("yface","y-face","m",dim_vect_face,&xBoxCen);
    //createAttVector("zface","z-face","m",dim_vect_face,&xBoxCen);

    // !!! make sure the order is specificall nt,nz,ny,nx in this spot,
    //  the order doesn't seem to matter for other spots
    dim_vect_face.clear();
    dim_vect_face.push_back(NcDim_t);
    dim_vect_face.push_back(NcDim_nFace);
    
    // create output fields
    addOutputFields();

}

// Save output at cell-centered values
void PlumeOutput::save(float currentTime)
{

    // once past the time to start averaging, calculate the number of particles 
    // for each bin for each time till past the time to stop averaging.
    // LA note: concentration is only calculated at each time to average, 
    //  so while output concentrations are averages over the instantaneous concentrations,
    //  the instantaneous concentrations are never calculated, just the cumulative number of particles
    //  per bin are calculated, reset at each average concentration output.
    // LA note: Because we always want to see concentration averaging done at the last time of a time averaging period
    //  and at that last time of the time averaging period we do a final binning of particle locations,
    //  the first time of each time averaging period should not involve binning.
    //  This means instead of currentTime >= timeAvgStart, we need currentTime > timeAvgStart.
    //  Still need to double check with professors if this is the right way, or whether we should actually 
    //   just bin twice for the timeAvgStart time, once on each time average period that falls on that time.
    if( currentTime > timeAvgStart && currentTime <= timeAvgEnd ) {
        boxCount();
    }

    // only calculate the concentration and do output if it is during the next output time
    //  also reinitialize the counter (cBox) in prep of the next concentration time averaging period
    // LA note: No need to adjust this one to be > instead of >=, just the binning.
    if( currentTime >= nextOutputTime && currentTime <= timeAvgEnd ) {
        // FM, updated by LA, future work, need to adjust: the current output 
        //  is particle per volume, not mass per volume
        //  should be fixed by multiplying by the specific particle density of each particle
        //  which should probably be done at particle bin time because this value is different
        //  for each particle. Eventually the mass per particle will change as well.
        //  cc = dt/timeAvgFreq * mass_per_particle/vol => in kg/m3

        // this is what it used to be, which assumes the mass per particle is 1 per total number of particles to release
        //double cc = (dt)/(timeAvgFreq*volume* dis->particleList.size() );

        // adjusting concentration for averaging time and volume of the box
        // cc = dt/Tavg * 1/vol => in /m3 
        //double cc = (timeStep)/(timeAvgFreq*volume);

        //for( auto id = 0u; id <  conc.size();id++ ) {
        //    conc[id] = conc[id]*cc;
        //}

        // set output time for correct netcdf output
        time = currentTime;

        // save the fields to NetCDF files
        saveOutputFields();

        // reset container for the next averaging period
        for( auto id = 0u; id < pBox.size();id++ ) {
            pBox[id] = 0;
            conc[id] = 0.0;
        }

        // remove x, y, z
        // from output array after first save
        // FM: only remove time dep variables from output array after first save
        // LA note: the output counter is an inherited variable
        if( output_counter == 0 ) {
            rmTimeIndepFields();
        }

        // increment inherited output counter for next time insertion
        output_counter +=1;
        
        // update the next output time value 
        // so averaging and output only happens at the averaging frequency
        nextOutputTime = nextOutputTime + timeAvgFreq;

    }

};

void PlumeOutput::boxCount()
{

    // for all particles see where they are relative to the
    // concentration collection boxes
    for(auto parItr = plume->particleList.begin(); parItr != plume->particleList.end() ; parItr++ ) {
        
        // because particles all start out as active now, need to also check the release time
        if( (*parItr)->isActive == true ) {
            
            // Calculate which collection box this particle is currently in.
            // The method is the same as the setInterp3Dindexing() function in the Eulerian class:
            //  Correct the particle position by the bounding box starting edge
            //  then divide by the dx of the boxes plus a small number, running a floor function on the result
            //  to get the index of the nearest concentration box node in the negative direction.
            //  No need to calculate the fractional distance between nearest nodes since not interpolating.
            // Because the particle position is offset by the bounding box starting edge,
            //  particles in a spot to the left of the box will have a negative index
            //  and particles in a spot to the right of the box will have an index greater than the number of boxes.
            // Because dividing is not just the box size, but is the box size plus a really small number,
            //  particles are considered in a box if they are on the left hand node to the right hand node
            //  so particles go outside the box if their indices are at nx-2, not nx-1.

            // x-direction
            int idx = floor(((*parItr)->xPos-lBndx)/(boxSizeX+1e-9));      
            // y-direction
            int idy = floor(((*parItr)->yPos-lBndy)/(boxSizeY+1e-9));
            // z-direction
            int idz = floor(((*parItr)->zPos-lBndz)/(boxSizeZ+1e-9));
            
            // now, does the particle land in one of the boxes? 
            // if so, add one particle to that box count
            if( idx >= 0 && idx <= nBoxesX-1 && 
                idy >= 0 && idy <= nBoxesY-1 && 
                idz >= 0 && idz <= nBoxesZ-1 )
            {
                int id = idz*nBoxesY*nBoxesX + idy*nBoxesX + idx;
                pBox[id] ++;
                conc[id] = conc[id] + (*parItr)->m*(*parItr)->wdepos*(*parItr)->wdecay*timeStep/(timeAvgFreq*volume);
            }

        }   // is active == true

    }   // particle loop

}
