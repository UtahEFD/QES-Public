#pragma once

/*
 * This is a collection of variables containing information relevant to
 * sensors read from an xml.
 */

#include <algorithm>
#include "util/ParseInterface.h"

class URBInputData;
class URBGeneralData;

class Sensor : public ParseInterface
{
private:

public:

    int site_blayer_flag = 1;
    float site_one_overL, site_xcoord, site_ycoord;
    std::vector<float> site_wind_dir, site_z_ref, site_U_ref;

    float site_z0;

    float site_canopy_H, site_atten_coeff;

    int site_coord_flag = 1;
    int site_UTM_zone;
  	float site_UTM_x, site_UTM_y;
  	float site_lon, site_lat;

    virtual void parseValues()
    {

        parsePrimitive<float>(true, site_xcoord, "site_xcoord");
        parsePrimitive<float>(true, site_ycoord, "site_ycoord");
        parsePrimitive<int>(false, site_blayer_flag, "boundaryLayerFlag");
        parsePrimitive<float>(true, site_z0, "siteZ0");
        parsePrimitive<float>(true, site_one_overL, "reciprocal");
        parseMultiPrimitives<float>(true, site_z_ref, "height");
        parseMultiPrimitives<float>(true, site_U_ref, "speed");
        parseMultiPrimitives<float>(true, site_wind_dir, "direction");

        parsePrimitive<int>(false, site_coord_flag, "site_coord_flag");
    		parsePrimitive<float>(false, site_UTM_x, "site_UTM_x");
    		parsePrimitive<float>(false, site_UTM_y, "site_UTM_y");
    		parsePrimitive<int>(false, site_UTM_zone, "site_UTM_zone");
    		parsePrimitive<float>(false, site_lon, "site_lon");
    		parsePrimitive<float>(false, site_lat, "site_lat");

        parsePrimitive<float>(false, site_canopy_H, "canopyHeight");
        parsePrimitive<float>(false, site_atten_coeff, "attenuationCoefficient");

    }

    /**
     * @brief Computes the wind velocity profile using Barn's scheme
     * at the site's sensor
     *
     * This function takes in information for each site's sensor (boundary layer flag, reciprocal coefficient, surface
     * roughness and measured wind velocity and direction), generates wind velocity profile for each sensor and finally
     * utilizes Barns scheme to interplote velocity to generate the initial velocity field for the domain.
     */
    void inputWindProfile(const URBInputData *UID, URBGeneralData *ugd);


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
