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

/*
  Author: Fabien Margairaz
  Date: Feb. 2020
*/

class LocalMixingParam : public ParseInterface
{ 
private:
  
protected:
    
public:

    int methodLocalMixing;
    bool save2file;
    std::string filename,varname;
  
    LocalMixingParam()
    {}
    ~LocalMixingParam()
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
        
    }
  
};

