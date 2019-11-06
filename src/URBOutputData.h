#pragma once

#include <string>

#include "URBGeneralData.h"
#include "URBInputData.h"
#include "URBOutput_VizFields.h"
#include "URBOutput_TURBInputFile.h"

class URBOutputData
{
public:
URBOutputData() {}
URBOutputData(URBGeneralData*,URBInputData*,std::string);
~URBOutputData() {}
void save(URBGeneralData*);
private:
URBOutput_VizFields* output_viz;
URBOutput_TURBInputFile* output_turb;

};

