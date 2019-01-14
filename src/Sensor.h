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
    float site_one_overL, site_xcoord, site_ycoord, site_wind_dir;

    float site_z0, site_z_ref, site_U_ref;

    float site_canopy_H, site_atten_coeff;

    virtual void parseValues()
    {

        parsePrimitive<float>(true, site_xcoord, "site_xcoord");
        parsePrimitive<float>(true, site_ycoord, "site_ycoord");
        parsePrimitive<int>(true, site_blayer_flag, "boundaryLayerFlag");
        parsePrimitive<float>(true, site_z0, "siteZ0");
        parsePrimitive<float>(true, site_one_overL, "reciprocal");
        parsePrimitive<float>(true, site_z_ref, "height");
        parsePrimitive<float>(true, site_U_ref, "speed");
        parsePrimitive<float>(true, site_wind_dir, "direction");

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
    void inputWindProfile(float dx, float dy, float dz, int nx, int ny, int nz, double *u0, double *v0, double *w0,
                          int num_sites, int *site_blayer_flag, float *site_one_overL, float *site_xcoord,
                          float *site_ycoord, float *site_wind_dir, float *site_z0, float *site_z_ref, float *site_U_ref,
                          float *x, float *y, float *z, Canopy* canopy, float *site_canopy_H, float *site_atten_coeff);

    /**
    * @brief Converts UTM to lat/lon and vice versa of the sensor coordiantes
    *
    */
    void UTMConverter (float rlon, float rlat, float rx, float ry, int UTM_PROJECTION_ZONE, int iway);

    /**
    * @brief Calculates the convergence value based on lat/lon input
    *
    */
    void getConvergence(float lon, float lat, int site_UTM_zone, float convergence, float pi);

};
