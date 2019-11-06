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

URBOutput_VizFields* output_viz = nullptr;
URBOutput_TURBInputFile* output_turb = nullptr;

};

