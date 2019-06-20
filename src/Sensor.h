#pragma once

/*
 * This is a collection of variables containing information relevant to
 * sensors read from an xml.
 */

#include "util/ParseInterface.h"
#include "Canopy.h"
#include <algorithm>

class Sensor : public ParseInterface
{
private:

public:

    int site_blayer_flag;
    float site_one_overL, site_xcoord, site_ycoord;
    std::vector<float> site_wind_dir, site_z_ref, site_U_ref;

    float site_z0;

    float site_canopy_H, site_atten_coeff;

    int site_coord_flag, site_UTM_zone;
  	float site_UTM_x, site_UTM_y;
  	float site_lon, site_lat;

    virtual void parseValues()
    {

        parsePrimitive<float>(true, site_xcoord, "site_xcoord");
        parsePrimitive<float>(true, site_ycoord, "site_ycoord");
        parsePrimitive<int>(true, site_blayer_flag, "boundaryLayerFlag");
        parsePrimitive<float>(true, site_z0, "siteZ0");
        parsePrimitive<float>(true, site_one_overL, "reciprocal");
        parseMultiPrimitives<float>(true, site_z_ref, "height");
        parseMultiPrimitives<float>(true, site_U_ref, "speed");
        parseMultiPrimitives<float>(true, site_wind_dir, "direction");

        parsePrimitive<int>(true, site_coord_flag, "site_coord_flag");
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
    void inputWindProfile(float dx, float dy, float dz, int nx, int ny, int nz, std::vector<double> &u0,
    	 						std::vector<double> &v0, std::vector<double> &w0, std::vector<float> z, std::vector<Sensor*> sensors,
                  Canopy* canopy, float UTMx, float UTMy, float theta, float UTMZone, std::vector<float> z0_domain,
                  std::vector<int> terrain_id);

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
