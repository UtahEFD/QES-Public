#pragma once

/*
 * This is a collection of variables containing information relevant to
 * sensors read from an xml.
 */

#include <algorithm>
#include "util/ParseInterface.h"
#include "TimeSeries.h"

class WINDSInputData;
class WINDSGeneralData;

class Sensor : public ParseInterface
{
private:

public:


    float site_xcoord, site_ycoord;

    int site_coord_flag = 1;
    int site_UTM_zone;
  	float site_UTM_x, site_UTM_y;
  	float site_lon, site_lat;

    std::vector<TimeSeries*> TS;


    virtual void parseValues()
    {
        parsePrimitive<int>(false, site_coord_flag, "site_coord_flag");
        parsePrimitive<float>(true, site_xcoord, "site_xcoord");
        parsePrimitive<float>(true, site_ycoord, "site_ycoord");
    		parsePrimitive<float>(false, site_UTM_x, "site_UTM_x");
    		parsePrimitive<float>(false, site_UTM_y, "site_UTM_y");
    		parsePrimitive<int>(false, site_UTM_zone, "site_UTM_zone");
    		parsePrimitive<float>(false, site_lon, "site_lon");
    		parsePrimitive<float>(false, site_lat, "site_lat");

        parseMultiElements<TimeSeries>(false, TS, "timeSeries");

    }

    void parseTree(pt::ptree t)
  	{
  			setTree(t);
  			setParents("root");
  			parseValues();
  	}


    /**
     * @brief Computes the wind velocity profile using Barn's scheme
     * at the site's sensor
     *
     * This function takes in information for each site's sensor (boundary layer flag, reciprocal coefficient, surface
     * roughness and measured wind velocity and direction), generates wind velocity profile for each sensor and finally
     * utilizes Barns scheme to interplote velocity to generate the initial velocity field for the domain.
     */
    void inputWindProfile(const WINDSInputData *WID, WINDSGeneralData *WGD, int index);


    /**
    * @brief Converts UTM to lat/lon and vice versa of the sensor coordiantes
    *
    */
    void UTMConverter (float rlon, float rlat, float rx, float ry, int UTM_PROJECTION_ZONE, int iway);

    /**
    * @brief Calculates the convergence value based on lat/lon input
    *
    */
    void getConvergence(float lon, float lat, int site_UTM_zone, float convergence);

};
