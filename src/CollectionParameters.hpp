//
//  CollectionParamters.hpp
//  
//  This class handles xml collection box options
//  this is for collecting output from Lagrangian particle values to Eulerian values like concentration
//
//  Created by Jeremy Gibbs on 03/25/19.
//  Modified by Loren Atwood on 02/18/20.
//

#ifndef COLLECTIONPARAMETERS_HPP
#define COLLECTIONPARAMETERS_HPP

#include "util/ParseInterface.h"
#include <string>
#include <cmath>

class CollectionParameters : public ParseInterface {
    
    private:
    
    public:
    
        int nBoxesX, nBoxesY, nBoxesZ;
        float boxBoundsX1, boxBoundsY1, boxBoundsZ1;
        float boxBoundsX2, boxBoundsY2, boxBoundsZ2;
        float timeAvgStart;     // time to start concentration averaging, not the time to start output. Adjusted if the time averaging duration does not divide evenly by the averaging frequency
        float timeAvgFreq;      // time averaging frequency and output frequency
    	        
    	virtual void parseValues() {
        	parsePrimitive< float >(true, timeAvgStart,   "timeAvgStart");
    		parsePrimitive< float >(true, timeAvgFreq,    "timeAvgFreq");
        	parsePrimitive< float >(true, boxBoundsX1, "boxBoundsX1");
    		parsePrimitive< float >(true, boxBoundsY1, "boxBoundsY1");
    		parsePrimitive< float >(true, boxBoundsZ1, "boxBoundsZ1");
    		parsePrimitive< float >(true, boxBoundsX2, "boxBoundsX2");
    		parsePrimitive< float >(true, boxBoundsY2, "boxBoundsY2");
    		parsePrimitive< float >(true, boxBoundsZ2, "boxBoundsZ2");
    		parsePrimitive< int   >(true, nBoxesX,     "nBoxesX");
    		parsePrimitive< int   >(true, nBoxesY,     "nBoxesY");
    		parsePrimitive< int   >(true, nBoxesZ,     "nBoxesZ");

            // check some of the parsed values to see if they make sense
            checkParsedValues();
    	}

        void checkParsedValues()
        {
            // make sure that all variables are greater than 0 except where they need to be at least 0
            if( timeAvgStart < 0 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input timeAvgStart must be greater than or equal to zero!";
                std::cerr << " timeAvgStart = \"" << timeAvgStart << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( timeAvgFreq <= 0 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input timeAvgFreq must be greater than zero!";
                std::cerr << " timeAvgFreq = \"" << timeAvgFreq << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( boxBoundsX1 < 0 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsX1 must be zero or greater!";
                std::cerr << " boxBoundsX1 = \"" << boxBoundsX1 << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( boxBoundsY1 < 0 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsY1 must be zero or greater!";
                std::cerr << " boxBoundsY1 = \"" << boxBoundsY1 << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( boxBoundsZ1 < 0 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsZ1 must be zero or greater!";
                std::cerr << " boxBoundsZ1 = \"" << boxBoundsZ1 << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( boxBoundsX2 < 0 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsX2 must be zero or greater!";
                std::cerr << " boxBoundsX2 = \"" << boxBoundsX2 << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( boxBoundsY2 < 0 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsY2 must be zero or greater!";
                std::cerr << " boxBoundsY2 = \"" << boxBoundsY2 << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( boxBoundsZ2 < 0 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsZ2 must be zero or greater!";
                std::cerr << " boxBoundsZ2 = \"" << boxBoundsZ2 << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( nBoxesX < 1 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input nBoxesX must be one or greater!";
                std::cerr << " nBoxesX = \"" << nBoxesX << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( nBoxesY < 1 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input nBoxesY must be one or greater!";
                std::cerr << " nBoxesY = \"" << nBoxesY << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( nBoxesZ < 1 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input nBoxesZ must be one or greater!";
                std::cerr << " nBoxesZ = \"" << nBoxesZ << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }


            // make sure the input timeAvgStart is not greater than the timeEnd
            // LA note: since the timeAvgEnd is not an input anymore, but is the simulation duration
            //  (now always ending output and averaging at simulation end time), and since the simulation
            //  duration is not known at parse time, !!! This check needs done in the time averaging output at constructor time
            
            // make sure the boxBounds1 is not greater than the boxBounds2 for each dimension
            if( boxBoundsX1 > boxBoundsX2 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsX1 must be smaller than or equal to input boxBoundsX2!";
                std::cerr << " boxBoundsX1 = \"" << boxBoundsX1 << "\", boxBoundsX2 = \"" << boxBoundsX2 << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( boxBoundsY1 > boxBoundsY2 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsY1 must be smaller than or equal to input boxBoundsY2!";
                std::cerr << " boxBoundsY1 = \"" << boxBoundsY1 << "\", boxBoundsY2 = \"" << boxBoundsY2 << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( boxBoundsZ1 > boxBoundsZ2 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input boxBoundsZ1 must be smaller than or equal to input boxBoundsZ2!";
                std::cerr << " boxBoundsZ1 = \"" << boxBoundsZ1 << "\", boxBoundsZ2 = \"" << boxBoundsZ2 << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }

            
            // make sure timeAvgFreq is not bigger than the simulation duration
            // LA note: timeAvgFreq can be as big as the collection duration, or even smaller than the collection duration
            //  IF timeAvgFreq is at least the same size or smaller than the simulation duration
            //  UNFORTUNATELY, variables related to the simulation duration are not available here. 
            //  This means this should probably be checked in the time averaging output at constructor time
            
        }

};
#endif