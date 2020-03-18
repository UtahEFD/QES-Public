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
        save2file = "false";
        parsePrimitive<bool>(false, save2file, "save");
        filename = "";
        parsePrimitive<std::string>(false, filename, "LM");
        filename = "mixlength"; // default name
        parsePrimitive<std::string>(false, varname, "varname");
    }
  
};

