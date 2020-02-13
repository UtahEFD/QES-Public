//
//  NetCDFOutputLagrangian.cpp
//  
//  This class handles saving output files for Lagrangian particle data
//  This is a specialized output class derived 
//   and inheriting from NetCDFOutputGeneric.
//
//  Created by Fabien Margairaz on 01/25/20
//  Modified by Loren Atwood 02/08/20
//


#include "PlumeOutputLagrangian.h"


// note that this sets the output file and the bool for whether to do output, in the netcdf inherited classes
PlumeOutputLagrangian::PlumeOutputLagrangian(PlumeInputData* PID,Dispersion* dis_ptr,std::string output_file,bool doFileOutput_val)
  : NetCDFOutputGeneric(output_file,doFileOutput_val)
{

    // return not doing anything if this file is not desired
    // in essence acting like an empty constructor
    if( doFileOutput == false )
    {
        return;
    }

    std::cout << "[PlumeOutputLagrangian] set up NetCDF file "<< output_file << std::endl;


    // setup output frequency control information
    // FM future work: need to create dedicated input variables
    //  for now, use the simulation start and end times, as well as the simulation timestep
    // LA for now, use the updateFrequency_timeLoop variable for the outputFrequency.
    //  because the time counter in this class uses time and not a timeIdx, also need to use the timestep
    outputStartTime = 0;     // time to start output
    outputEndTime =  PID->simParams->simDur;      // time to end output
    outputFrequency = PID->simParams->updateFrequency_timeLoop*PID->simParams->timeStep;        // output frequency
    std::cout << "outputFrequency = \"" << outputFrequency << "\"" << std::endl;

    // set the initial next output time value
    nextOutputTime = outputStartTime;


    // setup copy of disp pointer so output data can be grabbed directly
    disp = dis_ptr;


    // --------------------------------------------------------
    // setup the paricle information storage
    // --------------------------------------------------------

    // get total number of particle to be released 
    numPar = disp->totalParsToRelease;
    std::cout << "[PlumeOutputLagrangian] total number of particle to be saved in file " << numPar << std::endl;

    // initialization of the main particle metadata containers, setting initial vals to different noDataVals
    parID.resize(numPar,-999);
    xPos_init.resize(numPar,-999.0);
    yPos_init.resize(numPar,-999.0);
    zPos_init.resize(numPar,-999.0);
    tStrt.resize(numPar,-999.0);
    sourceIdx.resize(numPar,-999);

    // initialization of the other particle data containers, setting initial vals to different noDataVals
    xPos.resize(numPar,-999.0);
    yPos.resize(numPar,-999.0);
    zPos.resize(numPar,-999.0);
    uFluct.resize(numPar,-999.0);
    vFluct.resize(numPar,-999.0);
    wFluct.resize(numPar,-999.0);
    delta_uFluct.resize(numPar,-999.0);
    delta_vFluct.resize(numPar,-999.0);
    delta_wFluct.resize(numPar,-999.0);
    isRogue.resize(numPar,-999);
    isActive.resize(numPar,-999);

    // need to set the parId vars now
    for( int k = 0; k < numPar; k++)
    {
        parID.at(k) = k;
    }

    
    // --------------------------------------------------------
    // setup the netcdf output information storage
    // --------------------------------------------------------

    // setup desired output fields string
    // FM future work: can be added in fileOptions at some point
    output_fields = {   "t","parID","xPos_init","yPos_init","zPos_init","tStrt","sourceIdx", 
                        "xPos","yPos","zPos","uFluct","vFluct","wFluct","delta_uFluct","delta_vFluct","delta_wFluct",
                        "isRogue","isActive"};


    // set data dimensions, which in this case are cell-centered dimensions
    // time dimension
    NcDim NcDim_t = addDimension("t");
    // particles dimensions
    NcDim NcDim_par = addDimension("parID",numPar);

    // create attributes for time dimension
    std::vector<NcDim> dim_vect_t;
    dim_vect_t.push_back(NcDim_t);
    createAttScalar("t","time","s",dim_vect_t,&time);

    // create attributes space dimensions
    std::vector<NcDim> dim_vect_par;
    dim_vect_par.push_back(NcDim_par);
    createAttVector("parID","particle ID","--",dim_vect_par,&parID);


    // create 2D vector and put in the dimensions (time,par).
    // !!! make sure the order is specificall nt,nPar in this spot,
    //  the order doesn't seem to matter for other spots
    std::vector<NcDim> dim_vect_2d;
    dim_vect_2d.push_back(NcDim_t);
    dim_vect_2d.push_back(NcDim_par);


    // create attributes for adding all particle information
    createAttVector("xPos_init","initial-x-position","m",dim_vect_2d,&xPos_init);
    createAttVector("yPos_init","initial-y-position","m",dim_vect_2d,&yPos_init);
    createAttVector("zPos_init","initial-z-position","m",dim_vect_2d,&zPos_init);
    createAttVector("tStrt","particle-release-time","s",dim_vect_2d,&tStrt);
    createAttVector("sourceIdx","particle-sourceID","--",dim_vect_2d,&sourceIdx);
    
    createAttVector("xPos","x-position","m",dim_vect_2d,&xPos);
    createAttVector("yPos","y-position","m",dim_vect_2d,&yPos);
    createAttVector("zPos","z-position","m",dim_vect_2d,&zPos);
    createAttVector("uFluct","u-velocity-fluctuation","m s-1",dim_vect_2d,&uFluct);
    createAttVector("vFluct","v-velocity-fluctuation","m s-1",dim_vect_2d,&vFluct);
    createAttVector("wFluct","w-velocity-fluctuation","m s-1",dim_vect_2d,&wFluct);
    createAttVector("delta_uFluct","uFluct-difference","m s-1",dim_vect_2d,&delta_uFluct);
    createAttVector("delta_vFluct","vFluct-difference","m s-1",dim_vect_2d,&delta_vFluct);
    createAttVector("delta_wFluct","wFluct-difference","m s-1",dim_vect_2d,&delta_wFluct);
    createAttVector("isRogue","is-particle-rogue","bool",dim_vect_2d,&isRogue);
    createAttVector("isActive","is-particle-active","bool",dim_vect_2d,&isActive);


    // create output fields
    addOutputFields();

}

void PlumeOutputLagrangian::save(float currentTime)
{

    // return not doing anything if this file is not desired
    // in essence acting like this is an empty function
    if( doFileOutput == false )
    {
        return;
    }

    if( currentTime >= nextOutputTime && currentTime <= outputEndTime )
    {
        // copy particle info into the required output storage containers
        for( int par = 0; par < disp->pointList.size(); par++ )
        {
            xPos_init.at(par) = disp->pointList.at(par).xPos_init;
            yPos_init.at(par) = disp->pointList.at(par).yPos_init;
            zPos_init.at(par) = disp->pointList.at(par).zPos_init;
            tStrt.at(par) = disp->pointList.at(par).tStrt;
            sourceIdx.at(par) = disp->pointList.at(par).sourceIdx;

            xPos.at(par) = disp->pointList.at(par).xPos;
            yPos.at(par) = disp->pointList.at(par).yPos;
            zPos.at(par) = disp->pointList.at(par).zPos;
            uFluct.at(par) = disp->pointList.at(par).uFluct;
            vFluct.at(par) = disp->pointList.at(par).vFluct;
            wFluct.at(par) = disp->pointList.at(par).wFluct;
            delta_uFluct.at(par) = disp->pointList.at(par).delta_uFluct;
            delta_vFluct.at(par) = disp->pointList.at(par).delta_vFluct;
            delta_wFluct.at(par) = disp->pointList.at(par).delta_wFluct;

            // since no boolean output exists, going to have to convert the values to ints
            if( disp->pointList.at(par).isRogue == true )
            {
                isRogue.at(par) = 1;
            } else
            {
                isRogue.at(par) = 0;
            }
            if( disp->pointList.at(par).isActive == true )
            {
                isActive.at(par) = 1;
            } else
            {
                isActive.at(par) = 0;
            }
        }


        // set output time for correct netcdf output
        time = currentTime;

        // save the fields to NetCDF files
        saveOutputFields();


        // FM: only remove time dep variables from output array after first save
        // LA note: the output counter is an inherited variable
        if( output_counter == 0 )
        {
            rmTimeIndepFields();
        }

        // increment inherited output counter for next time insertion
        output_counter += 1;


        // update the next output time so output only happens at output frequency
        nextOutputTime = nextOutputTime + outputFrequency;

    }

};
