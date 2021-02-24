#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <limits>

#include "util/ParseInterface.h" 
#include "Vector3.h"

/*
  Author: Fabien Margairaz
  Date: Feb. 2020
*/

class TURBParams : public ParseInterface
{ 
private:
  
protected:
    
public:
    
    int methodLocalMixing;
    bool save2file;
    std::string filename,varname;

    int mlSamplesPerAirCell;

    Vector3<float>* sigConst; 

    bool flagNonLocalMixing;  
    float turbUpperBound;

    TURBParams()
    {}
    ~TURBParams()
    {}
  
    virtual void parseValues()
    {
        methodLocalMixing=0;
        parsePrimitive<int>(true, methodLocalMixing, "method");
        save2file = false;
        parsePrimitive<bool>(false, save2file, "save");
        filename = "";
        parsePrimitive<std::string>(false, filename, "LMfile");
        varname = "mixlength"; // default name
        parsePrimitive<std::string>(false, varname, "varname");

        // defaults for local mixing sample rates -- used with ray
        // tracing methods
        if (methodLocalMixing == 3) { // OptiX
            mlSamplesPerAirCell = 2000;
        }
        else {
            mlSamplesPerAirCell = 500;  // other ray-traced methods
        }
        parsePrimitive<int>(false, mlSamplesPerAirCell, "samples");
        if (methodLocalMixing == 3) {
            
            std::cout << "Setting samples per air cell for ray-traced mixing length to " << mlSamplesPerAirCell << std::endl;
        }
        
        
        if(methodLocalMixing < 0 || methodLocalMixing > 4) {
            std::cout << "[WARNING] unknown local mixing method -> set method to 0 (height above terrain)" << std::endl;
            methodLocalMixing = 0;
        }

        if( (methodLocalMixing == 4 || save2file == true ) && ( filename == "" ) ) {
            std::cout << "[WARNING] no local mixing file provided -> set method to 0 (height above terrain)" << std::endl;
            methodLocalMixing = 0;
        }
        if(methodLocalMixing == 0 || methodLocalMixing == 4) {
            save2file = "false";  
        }

        parseElement< Vector3<float> >(false, sigConst, "sigmaConst");

        flagNonLocalMixing=false;
        parsePrimitive<bool>(false, flagNonLocalMixing, "nonLocalMixing");

        turbUpperBound=100;
        parsePrimitive<float>(false, turbUpperBound, "turbUpperBound");
        
        

    }
  
};

