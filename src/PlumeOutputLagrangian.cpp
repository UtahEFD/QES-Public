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
PlumeOutputLagrangian::PlumeOutputLagrangian(PlumeInputData* PID,Dispersion* dis_ptr,std::string output_file)
  : NetCDFOutputGeneric(output_file)
{

    std::cout << "[PlumeOutputLagrangian] set up NetCDF file "<< output_file << std::endl;


    // this is the current simulation start time
    // LA note: this would need adjusted if we ever change the 
    //  input simulation parameters to include a simStartTime
    //  instead of just using simDur
    float simStartTime = 0.0;
    
    // setup output frequency control information
    // FM future work: need to create dedicated input variables
    //  LA: we want output from the simulation start time to the end so the only dedicated input variable still needed
    //   would be an output frequency. For now, use the updateFrequency_timeLoop variable for the outputFrequency.
    // LA note: because the time counter in this class uses time and not a timeIdx, also need to use the timestep
    outputStartTime = simStartTime;     // time to start output, adjusted if the output duration does not divide evenly by the output frequency
    outputEndTime =  PID->simParams->simDur;      // time to end output
    outputFrequency = PID->simParams->updateFrequency_timeLoop*PID->simParams->timeStep;        // output frequency
    
    
    // Determine whether outputStartTime needs adjusted to make the output duration divide evenly by the output frequency
    // This is essentially always keeping the outputEndTime at what it is (end of the simulation), and adjusting the outputStartTime
    // to avoid slicing off an output time unless we have to.
    float outputDur = outputEndTime - outputStartTime;
    // if doesn't divide evenly, need to adjust outputStartTime
    if( outputDur % outputFrequency != 0 )
    {
        // clever algorythm that always gets the exact number of outputs when output duration divides evenly by output frequency
        // and rounds the number of outputs down to what it would be if the start time were the next smallest evenly dividing number
        int nOutputs = std::floor(outputDur/outputFrequency) + 1;
    
        // clever algorythm to always calculate the desired output start time based off the number of outputs
        // the outputStartTime if not adjusting nOutputs
        float current_outputStartTime = outputEndTime - outputFrequency*(nOutputs-1);
        // the outputStartTime if adjusting nOutputs. Note nOutputs has one extra outputs
        float adjusted_outputStartTime = outputEndTime - outputFrequency*(nOutputs);
        if( adjusted_outputStartTime >= simStartTime )
        {
            // need to adjust the outputStartTime to be the adjusted_outputStartTime
            // warn the user that the outputStartTime is being adjusted before adjusting outputStartTime
            std::cout << "[PlumeOutputLagrangian]: adjusting outputStartTime because output duration did not divide evenly by outputFrequency" << std::endl;
            std::cout << "  original outputStartTime = \"" << outputStartTime << "\", outputEndTime = \"" << outputEndTime 
                    << "\", outputFrequency = \"" << outputFrequency << "\", new outputStartTime = \"" << adjusted_outputStartTime << "\"" << std::endl;
            outputStartTime = adjusted_outputStartTime;
        } else
        {
            // need to adjust the outputStartTime to be the current_outputStartTime
            // warn the user that the outputStartTime is being adjusted before adjusting outputStartTime
            std::cout << "[PlumeOutputLagrangian]: adjusting outputStartTime because output duration did not divide evenly by outputFrequency" << std::endl;
            std::cout << "  original outputStartTime = \"" << outputStartTime << "\", outputEndTime = \"" << outputEndTime 
                    << "\", outputFrequency = \"" << outputFrequency << "\", new outputStartTime = \"" << current_outputStartTime << "\"" << std::endl;
            outputStartTime = current_outputStartTime;
        }
    } // else does divide evenly, no need to adjust anything so no else
    

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

    
    for( int par = 0; par < disp->pointList.size(); par++ )
    {
        std::cout << "par[" << par << "]" << std::endl;
        std::cout << "xPos_init = \"" << disp->pointList.at(par).xPos_init << "\"" << std::endl;
        std::cout << "yPos_init = \"" << disp->pointList.at(par).yPos_init << "\"" << std::endl;
        std::cout << "zPos_init = \"" << disp->pointList.at(par).zPos_init << "\"" << std::endl;
    }


    // initialize all the output containers
    // normally this is done by doing a resize, then setting values later,
    // but the full output needs initial values that match the particle values from the get go
    // so I will add them by pushback
    for( int par = 0; par < numPar; par++)
    {
        parID.push_back(par);

        xPos_init.push_back(disp->pointList.at(par).xPos_init);
        yPos_init.push_back(disp->pointList.at(par).yPos_init);
        zPos_init.push_back(disp->pointList.at(par).zPos_init);
        tStrt.push_back(disp->pointList.at(par).tStrt);
        sourceIdx.push_back(disp->pointList.at(par).sourceIdx);

        xPos.push_back(disp->pointList.at(par).xPos);
        yPos.push_back(disp->pointList.at(par).yPos);
        zPos.push_back(disp->pointList.at(par).zPos);
        uFluct.push_back(disp->pointList.at(par).uFluct);
        vFluct.push_back(disp->pointList.at(par).vFluct);
        wFluct.push_back(disp->pointList.at(par).wFluct);
        delta_uFluct.push_back(disp->pointList.at(par).delta_uFluct);
        delta_vFluct.push_back(disp->pointList.at(par).delta_vFluct);
        delta_wFluct.push_back(disp->pointList.at(par).delta_wFluct);
        
        // since no boolean output exists, going to have to convert the values to ints
        if( disp->pointList.at(par).isRogue == true )
        {
            isRogue.push_back(1);
        } else
        {
            isRogue.push_back(0);
        }
        if( disp->pointList.at(par).isActive == true )
        {
            isActive.push_back(1);
        } else
        {
            isActive.push_back(0);
        }
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

    // only output if it is during the next output time but before the end time
    if( currentTime >= nextOutputTime && currentTime <= outputEndTime )
    {
        // copy particle info into the required output storage containers
        for( int par = 0; par < disp->nParsReleased; par++ )
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

        for( int par = 0; par < disp->pointList.size(); par++ )
        {
            std::cout << "par[" << par << "]" << std::endl;
            std::cout << "xPos_init = \"" << disp->pointList.at(par).xPos_init << "\"" << std::endl;
            std::cout << "yPos_init = \"" << disp->pointList.at(par).yPos_init << "\"" << std::endl;
            std::cout << "zPos_init = \"" << disp->pointList.at(par).zPos_init << "\"" << std::endl;
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
