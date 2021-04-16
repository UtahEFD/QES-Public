#pragma once

#include <cmath>
#include <vector>
#include <map>

// forward declaration of WINDSInputData and WINDSGeneralData, which
// will be used by the derived classes and thus included there in the
// C++ files
class WINDSInputData;
class WINDSGeneralData;
class TURBGeneralData;

class Canopy
{
private:

public:
    
    Canopy()
    {}

    Canopy(const WINDSInputData* WID, WINDSGeneralData* WGD);

    virtual ~Canopy()
    {}
    
    /*
     * For all Canopy classes derived, this need to be defined
    virtual void parseValues()
    {
        parsePrimitive<int>(true, num_canopies, "num_canopies");
        // read the input data for canopies
        //parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyHomogeneous>("Homogeneous"));
        //parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyIsolatedTree>("IsolatedTree"));
        //parseMultiPolymorphs(false, canopies, Polymorph<Building, CanopyWindbreak>("Windbreak"));
        // add other type of canopy here
    }
    */
   
    virtual void canopyVegetation(WINDSGeneralData *wgd);
    virtual void canopyWake(WINDSGeneralData *wgd) 
    {
        return;
    }

    virtual int getCellFlagCanopy();
    virtual int getCellFlagWake();

    
protected: 
    
    int nx_canopy, ny_canopy, nz_canopy;
    int numcell_cent_2d, numcell_cent_3d;

    /*!
     * 
     */
    std::vector<float> canopy_atten;	  /**< Canopy attenuation coefficient */
    
    std::vector<float> canopy_bot;		  /**< Canopy bottom */
    std::vector<int> canopy_bot_index;	  /**< Canopy bottom index */
    std::vector<float> canopy_top;		  /**< Canopy top */
    std::vector<int> canopy_top_index;	  /**< Canopy top index */

    std::vector<float> canopy_base;	      /**< Canopy base */
    std::vector<float> canopy_height;	  /**< Canopy height */

    std::vector<float> canopy_z0;		  /**< Canopy surface roughness */
    std::vector<float> canopy_ustar;	  /**< Velocity gradient at the top of canopy */
    std::vector<float> canopy_d;		  /**< Canopy displacement length */
    
    /*!
     * This function takes in icellflag defined in the defineCanopy function along with variables initialized in
     * the readCanopy function and initial velocity field components (u0 and v0). This function applies the urban 
     * canopy parameterization and returns modified initial velocity field components.
     */
    void canopyCioncoParam(WINDSGeneralData * wgd);
    
    /*!
     * This function is being call from the plantInitial function and uses linear regression method to define 
     * ustar and surface roughness of the canopy.
     */
    void canopyRegression(WINDSGeneralData *wgd);
    
    /*!
     * This is a new function wrote by Lucas Ulmer and is being called from the plantInitial function. The purpose 
     * of this function is to use bisection method to find root of the specified equation. It calculates the 
     * displacement height when the bisection function is not finding it.
     */
    float canopySlopeMatch(float z0, float canopy_top, float canopy_atten);
    
    /*!
     * 
     */
    float canopyBisection(float ustar, float z0, float canopy_top, float canopy_atten, float vk, float psi_m);

private:
    

};

inline int Canopy::getCellFlagCanopy()
{
    return 18;
}

inline int Canopy::getCellFlagWake()
{
    return 19;
}
