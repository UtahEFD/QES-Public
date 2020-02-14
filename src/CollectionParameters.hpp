//
//  CollectionParamters.hpp
//  
//  This class handles xml collection box options
//  this is for collecting output from Lagrangian particle values to Eulerian values like concentration
//
//  Created by Jeremy Gibbs on 03/25/19.
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
        float timeStart, timeEnd, timeAvg ;		// Still confused why, but I think timeAvg usually needs set to be (timeEnd-timeStart) - timeStep. Something to look into in the future
    	        
    	virtual void parseValues() {
        	parsePrimitive< float >(true, timeStart,   "timeStart");
    		parsePrimitive< float >(true, timeEnd,     "timeEnd");
    		parsePrimitive< float >(true, timeAvg,     "timeAvg");
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
            if( timeStart < 0 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input timeStart must be greater than or equal to zero!";
                std::cerr << " timeStart = \"" << timeStart << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( timeEnd <= 0 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input timeEnd must be greater than zero!";
                std::cerr << " timeEnd = \"" << timeEnd << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }
            if( timeAvg <= 0 )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input timeAvg must be greater than zero!";
                std::cerr << " timeAvg = \"" << timeAvg << "\"" << std::endl;
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


            // make sure the input timeStart is not greater than the timeStart
            if( timeStart >= timeEnd )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input timeStart must be smaller than input timeEnd!";
                std::cerr << " timeStart = \"" << timeStart << "\", timeEnd = \"" << timeEnd << "\"" << std::endl;
                exit(EXIT_FAILURE);
            }

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

            
            // make sure timeAvg is not bigger than the collection duration
            // LA note: this is not the same way an nTimes variable is calculated in plume, I'm assuming that zero doesn't matter
            float collectionDur = timeEnd - timeStart;
            if( timeAvg > collectionDur )
            {
                std::cerr << "(CollectionParameters::checkParsedValues): input timeAvg must be smaller than or equal to calculated collectionDur!";
                std::cerr << " timeAvg = \"" << timeAvg << "\", collectionDur = \"" << collectionDur << "\"" << std::endl;
            }

        }

};
#endif