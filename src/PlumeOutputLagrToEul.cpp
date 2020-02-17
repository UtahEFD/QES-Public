//
//  NetCDFOutputLagrToEul.h
//  
//  This class handles saving output files for Eulerian binned Lagrangian particle data,
//   where this class handles the binning of the Lagrangian particle data
//  This is a specialized output class derived 
//   and inheriting from NetCDFOutputGeneric.
//
//  Created by Fabien Margairaz on 01/25/20
//  Modified by Loren Atwood 02/08/20
//


#include "PlumeOutputLagrToEul.h"


// note that this sets the output file and the bool for whether to do output, in the netcdf inherited classes
// in this case, output should always be done, so the bool for whether to do output is set to true
PlumeOutputLagrToEul::PlumeOutputLagrToEul(PlumeInputData* PID,Urb* urb_ptr,Dispersion* dis_ptr,std::string output_file)
  : NetCDFOutputGeneric(output_file)
{

    std::cout << "[PlumeOutputLagrToEul] set up NetCDF file " << output_file << std::endl;


    // setup output frequency control information
    timeAvgStart = PID->colParams->timeStart;       // time to start concentration averaging and output
    timeAvgEnd = PID->colParams->timeEnd;           // time to end concentration averaging and output
    timeAvgFreq = PID->colParams->timeAvg;          // time averaging frequency and output frequency

    
    // set the initial next averaging time value, which is also the next output time value
    nextAvgTime = timeAvgStart + timeAvgFreq;
    
    
    // setup copy of disp pointer so output data can be grabbed directly
    disp = dis_ptr;


    // need nx, ny, nz of the domain to make sure the output handles domains that are not three dimensional
    // for now these are a copy of the input urb values
    nx = urb_ptr->nx;
    ny = urb_ptr->ny;
    nz = urb_ptr->nz;

    // need the simulation timeStep for use in concentration averaging
    timeStep = PID->simParams->timeStep;


    // --------------------------------------------------------
    // setup the sampling box information 
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
    for(int k = 0; k < nBoxesZ; ++k)
    {
        zBoxCen.at(k) = lBndz + (zR*boxSizeZ) + (boxSizeZ/2.0);
        zR++;
    }
    for(int j = 0; j < nBoxesY; ++j)
    {
        yBoxCen.at(j) = lBndy + (yR*boxSizeY) + (boxSizeY/2.0);
        yR++;
    }
    for(int i = 0; i <nBoxesX; ++i)
    {
        xBoxCen.at(i) = lBndx + (xR*boxSizeX) + (boxSizeX/2.0);
        xR++;
    }

    // initialization of the container
    cBox.resize(nBoxesX*nBoxesY*nBoxesZ,0.0);
    conc.resize(nBoxesX*nBoxesY*nBoxesZ,0.0);


    // --------------------------------------------------------
    // setup the netcdf output information storage
    // --------------------------------------------------------

    // setup desired output fields string
    // LA future work: can be added in fileOptions at some point
    output_fields = {"t","x","y","z","conc"};


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
    createAttVector("x","x-distance","m",dim_vect_x,&xBoxCen);
    std::vector<NcDim> dim_vect_y;
    dim_vect_y.push_back(NcDim_y);
    createAttVector("y","y-distance","m",dim_vect_y,&yBoxCen);
    std::vector<NcDim> dim_vect_z;
    dim_vect_z.push_back(NcDim_z);
    createAttVector("z","z-distance","m",dim_vect_z,&zBoxCen);


    // create 3D vector and put in the dimensions (nt,nz,ny,nx).
    // !!! make sure the order is specificall nt,nz,ny,nx in this spot,
    //  the order doesn't seem to matter for other spots
    std::vector<NcDim> dim_vect_3d;
    dim_vect_3d.push_back(NcDim_t);
    dim_vect_3d.push_back(NcDim_z);
    dim_vect_3d.push_back(NcDim_y);
    dim_vect_3d.push_back(NcDim_x);
    
    
    // create attributes for all output information
    createAttVector("conc","concentration","#ofPar m-3",dim_vect_3d,&conc);


    // create output fields
    addOutputFields();

}

// Save output at cell-centered values
void PlumeOutputLagrToEul::save(float currentTime)
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
    if( currentTime > timeAvgStart && currentTime <= timeAvgEnd )
    {
        boxCount(currentTime);
    }

    // only calculate the concentration and do output if it is during the next averaging (output) time
    //  also reinitialize the counter (cBox) in prep of the next concentration averaging (output) time
    // LA note: No need to adjust this one to be > instead of >=, just the binning.
    if( currentTime >= nextAvgTime && currentTime <= timeAvgEnd )
    {
        // FM, updated by LA, future work, need to adjust: the current output 
        //  is particle per volume, not mass per volume
        //  should be fixed by multiplying by the specific particle density of each particle
        //  which should probably be done at particle bin time because this value is different
        //  for each particle. Eventually the mass per particle will change as well.
        //  cc = dt/timeAvgFreq * mass_per_particle/vol => in kg/m3

        // this is what it used to be, which assumes the mass per particle is 1 per total number of particles to release
        //double cc = (dt)/(timeAvgFreq*volume* dis->pointList.size() );

        // FM - here: number of particles in volume
        // cc = dt/timeAvgFreq * 1/vol => in #/m3 
        double cc = (timeStep)/(timeAvgFreq*volume);

        for( auto id = 0; id <  conc.size();id++ )
        {
            conc.at(id) = cBox.at(id)*cc;
            // notice that the number of particles per box is reset for the next averaging time
            cBox.at(id) = 0.0;
        }


        // set output time for correct netcdf output
        time = currentTime;

        // save the fields to NetCDF files
        saveOutputFields();


        // remove x, y, z
        // from output array after first save
        // FM: only remove time dep variables from output array after first save
        // LA note: the output counter is an inherited variable
        if( output_counter == 0 )
        {
            rmTimeIndepFields();
        }

        // increment inherited output counter for next time insertion
        output_counter +=1;


        // update the next averaging (output) time so averaging (output) only happens at averaging (output) frequency
        nextAvgTime = nextAvgTime + timeAvgFreq;    

    }

};

void PlumeOutputLagrToEul::boxCount(float currentTime)
{

    // for all particles see where they are relative to the
    // concentration collection boxes
    for(int i = 0; i < disp->pointList.size(); i++)
    {
        // because particles all start out as active now, need to also check the release time
        if( currentTime >= disp->pointList.at(i).tStrt && disp->pointList.at(i).isActive == true )
        {

            // get the current position of the particle
            double xPos = disp->pointList.at(i).xPos;
            double yPos = disp->pointList.at(i).yPos;
            double zPos = disp->pointList.at(i).zPos;

            
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
            int idx = floor((xPos-lBndx)/(boxSizeX+1e-9));      
            // y-direction
            int idy = floor((yPos-lBndy)/(boxSizeY+1e-9));
            // z-direction
            int idz = floor((zPos-lBndz)/(boxSizeZ+1e-9));
            
            // still have to correct the indices if the simulation is not three dimensional
            if( nx == 1 )
            {
                idx = 0;
            }
            if( ny == 1 )
            {
                idy = 0;
            }
            if( nz == 1 )
            {
                idz = 0;
            }

            // now, does the particle land in one of the boxes? 
            // if so, add one particle to that box count
            if( idx >= 0 && idx < nBoxesX-1 && 
                idy >= 0 && idy < nBoxesY-1 && 
                idz >= 0 && idz < nBoxesZ-1 )
            {
                int id = idz*nBoxesY*nBoxesX + idy*nBoxesX + idx;
                cBox.at(id) = cBox.at(id) + 1.0;
            }

        }   // is active == true

    }   // particle loop

}
