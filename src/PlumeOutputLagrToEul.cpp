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
PlumeOutputLagrToEul::PlumeOutputLagrToEul(PlumeInputData* PID,Dispersion* dis,std::string output_file)
  : NetCDFOutputGeneric(output_file,true)
{

    std::cout << "[PlumeOutputLagrToEul] set up NetCDF file " << output_file << std::endl;

    // --------------------------------------------------------
    // setup the sampling box concentration information 
    // --------------------------------------------------------

    // time to start averaging the concentration
    sCBoxTime = PID->colParams->timeStart;
    // time to output the concentration (1st output -> updated each time)
    avgOutTime = sCBoxTime + PID->colParams->timeAvg;
    // Copy of the input timeAvg and timeStep
    timeAvg = PID->colParams->timeAvg;
    timeStep = PID->simParams->timeStep;

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

    // copy of disp pointer
    disp = dis;

    // --------------------------------------------------------
    // setup the output information 
    // --------------------------------------------------------

    // setup output fields
    output_fields = {"t","x","y","z","conc"};

    // set cell-centered data dimensions
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

    // create 3D vector (time dep)
    std::vector<NcDim> dim_vect_3d;
    dim_vect_3d.push_back(NcDim_t);
    dim_vect_3d.push_back(NcDim_z);
    dim_vect_3d.push_back(NcDim_y);
    dim_vect_3d.push_back(NcDim_x);
    // create attributes
    createAttVector("conc","concentration","# m-3",dim_vect_3d,&conc);

    // create output fields
    addOutputFields();

}

// Save output at cell-centered values
void PlumeOutputLagrToEul::save(float currentTime)
{

    // if past the time to start averaging => calculate the concentration
    // count particule in sampling box
    if( currentTime >= sCBoxTime )
    {
        boxCount(disp);
    }

    // if the current time = time to output => output concentration and 
    // reinitialze counter (cBox)
    if( currentTime >= avgOutTime )
    {
        // FM -> need adjust - multiply by the mass of the particle 
        // cc = dt/T_avf * mass_of_part/vol => in kg/m3 
        //double cc = (dt)/(avgTime*volume* dis->pointList.size() );

        // FM - here: number of particles in volume
        // cc = dt/T_avf * 1/vol => in #/m3 
        double cc = (timeStep)/(timeAvg*volume);

        for( auto id = 0; id <  conc.size();id++ )
        {
            conc.at(id) = cBox.at(id)*cc;
            cBox.at(id) = 0.0;
        }

        // set output time
        time = currentTime;

        // save the fields to NetCDF files
        saveOutputFields();

        // remove x, y, z
        // from output array after first save
        if( output_counter == 0 )
        {
            rmTimeIndepFields();
        }

        // increment for next time insertion
        output_counter +=1;

        // avgTime - updated to the time for the next average output
        avgOutTime = avgOutTime + timeAvg;    

    }

};

void PlumeOutputLagrToEul::boxCount(const Dispersion* disp)
{

    // for all particles see where they are relative to the
    // concentration collection boxes
    for(int i = 0; i < disp->pointList.size(); i++)
    {
        
        double xPos = disp->pointList.at(i).xPos;
        double yPos = disp->pointList.at(i).yPos;
        double zPos = disp->pointList.at(i).zPos;

        // FM 
        //    -> why check here? sampling box can be smaller 
        //    -> this add a few check for nothting (and complicate the call because need URB/TURB info)
        //    if ( (xPos > 0.0 && yPos > 0.0 && zPos > 0.0) &&
        //    (xPos < (nx*dx)) && (yPos < (ny*dy)) && (zPos < (nz*dz)) ) {
        //

        // ????
        if( zPos == -1 )
        {
            continue;
        }

        // Calculate which collection box this particle is currently in
        // x-direction
        int idx = -1;
        if( xPos >= lBndx && xPos <= uBndx )
        {
            idx = (int)((xPos-lBndx)/boxSizeX);      
        }

        // y-direction
        int idy = -1;
        if( yPos >= lBndy && yPos <= uBndy )
        {
            idy = (int)((yPos-lBndy)/boxSizeY);
        }
        // z-direction
        int idz = -1;
        if( zPos >= lBndz && zPos <= uBndz )
        {
            idz = (int)((zPos-lBndz)/boxSizeZ);
        }

        // FM
        //- a particle has the potential to be on the limit of the last box
        //-> in that case id*=nBoxes* and the particle will not be counted
        //-> how to solve this issue?
        //

        if( idx >= 0 && idx < nBoxesX && 
            idy >= 0 && idy < nBoxesY && 
            idz >= 0 && idz < nBoxesZ )
        {
            int id = idz*nBoxesY*nBoxesX + idy*nBoxesX + idx;
            cBox.at(id) = cBox.at(id) + 1.0;
        }

    }   // particle loop

}
