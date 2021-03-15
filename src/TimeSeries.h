#pragma once

#include "boost/date_time/posix_time/posix_time.hpp"
#include "util/ParseInterface.h"

class URBInputData;
class URBGeneralData;

namespace bt = boost::posix_time;

class TimeSeries : public ParseInterface
{
private:
    
public:
    
    int site_blayer_flag = 1;
    float site_z0;
    std::vector<float> site_wind_dir, site_z_ref, site_U_ref;
    float site_one_overL;
    float site_canopy_H, site_atten_coeff;
    std::string timeStamp="";
    
    time_t timeEpoch=-1;
    bt::ptime timePosix;

    virtual void parseValues()
    {
        parsePrimitive<std::string>(false, timeStamp, "timeStamp");
        parsePrimitive<time_t>(false, timeEpoch, "timeEpoch");
        parsePrimitive<int>(false, site_blayer_flag, "boundaryLayerFlag");
        parsePrimitive<float>(true, site_z0, "siteZ0");
        parsePrimitive<float>(true, site_one_overL, "reciprocal");
        parseMultiPrimitives<float>(true, site_z_ref, "height");
        parseMultiPrimitives<float>(true, site_U_ref, "speed");
        parseMultiPrimitives<float>(true, site_wind_dir, "direction");
        parsePrimitive<float>(false, site_canopy_H, "canopyHeight");
        parsePrimitive<float>(false, site_atten_coeff, "attenuationCoefficient");
        
        if(timeStamp == "" && timeEpoch == -1) {
            std::cout << "[WARNING] no timestamp provided" << std::endl;
            timeStamp="2020-01-01T00:00";
            timePosix=bt::from_iso_extended_string(timeStamp);
            timeEpoch=bt::to_time_t(timePosix);
        } else if (timeStamp != "" && timeEpoch == -1) {
            timePosix=bt::from_iso_extended_string(timeStamp);
            timeEpoch=bt::to_time_t(timePosix);
        } else if (timeEpoch != -1 && timeStamp == "") {
            timePosix=bt::from_time_t(timeEpoch);
            timeStamp=bt::to_iso_extended_string(timePosix);
        } else {
            timePosix=bt::from_iso_extended_string(timeStamp);
            bt::ptime testtime = bt::from_time_t(timeEpoch);
            if (testtime != timePosix) {
                std::cerr << "[ERROR] invalid timeStamp (timeEpoch != timeStamp)\n";
                exit(EXIT_FAILURE);
            }
        }

    }
    
};
