#ifndef EULERIAN_H
#define EULERIAN_H

//#include "Util.h"
#include "Urb.hpp"
#include "Turb.hpp"
#include "PlumeInputData.hpp"
#include "TypeDefs.hpp"

#include <fstream>
#include <string>
#include <vector>


class Eulerian{
    
public:
    
    Eulerian(Urb*,Turb*,PlumeInputData*,const std::string& debugOutputFolder);   // copies the urb grid values for nx, ny, nz, nt, dx, dy, and dz to the Eulerian grid values,
                            // then calculates the tau gradients which are then used to calculate the flux_div grid values.
                            //    Eulerian grid accessing functions should be the main use of this class.
                            // since Urb and Turb stuff is not fully copied into this Eulerian dataset, some Eulerian grid accessing functions should probably be able to do calculations
                            // on datasets that are not stored in this class, but are values at this same Eulerian grid level. That being said, some of them could also be directly 
                            // put into Urb and Turb, depending on what is needed.     Since these functions work on Urb, Turb, and Eulerian data, I keep wondering if 
                            //     they should go somewhere else, but then again the only time data is accessed this way is if Eulerian is around, so maybe it's fine to have the 
                            // interp functions here.
    

    // Still need Tau variables.  They are defined here:
    std::vector<matrix6> taudx,taudy,taudz;     // these are the gradients of tau in each direction. Tau is still kept inside Turb
                                                // this is the only non symmetric parts of the gradient of the stress tensor 
                                                // (dtxxdx, dtxydx, dtxzdx, dtyydx, dtyzdx, dtzzdx)
                                                // so this is a vector of 6 values (e11, e12, e13, e22, e23, e33)
                                                // but linearized since it is at a 3D grid of points x,y,z that has been linearized
                                                // units are probably m/s^2

    std::vector<vec3> flux_div;     // taudx, taudy, and taudz are actually not used, except to calculate this value
                                    // this is just a value for each direction: flux_div_x, flux_div_y, flux_div_z
                                    // but now it is flux_div.e11, flux_div.e21, flux_div.e31


    void display(const matrix9&);       // you would think this should be a function put with the datatype. Seems useful, especially in debug mode
    void display(const matrix6&);       // you would think this should be a function put with the datatype. Seems useful, especially in debug mode
    void display(const vec3&);       // you would think this should be a function put with the datatype. Seems useful, especially in debug mode
    void display(const diagonal&);       // you would think this should be a function put with the datatype. Seems useful, especially in debug mode
    
    // just realized, what if urb and turb have different grids? For now assume they are the same grid
    int nx,ny,nz,nt;    // a copy of the urb grid information, the number of values in each dimension, including time
    double dx,dy,dz;    // a copy of the urb grid information, the difference between points in the grid
    double domainXstart;    // a copy of the urb domain starting x value
    double domainYstart;    // a copy of the urb domain starting y value
    double domainZstart;    // a copy of the urb domain starting z value

    double C_0;                 // used to separate out CoEps into its separate parts when doing debug output
    

    void setInterp3Dindexing(const vec3& xyz_particle);
    double interp3D(const std::vector<double>& EulerData,const std::string& dataName);
    vec3 interp3D(const std::vector<vec3>& EulerData);
    diagonal interp3D(const std::vector<diagonal>& EulerData,const std::string& dataName);
    matrix6 interp3D(const std::vector<matrix6>& EulerData);
    Wind interp3D(const std::vector<Wind>& EulerData);

    void outputVarInfo_text(Urb* urb, Turb* turb, const std::string& outputFolder);
    
 
private:

    // these are the current interp3D variables, as they are used for multiple interpolations for each particle
    int ii;     // this is the nearest cell index to the left in the x direction
    int jj;     // this is the nearest cell index to the left in the y direction
    int kk;     // this is the nearest cell index to the left in the z direction
    double iw;     // this is the normalized distance to the nearest cell index to the left in the x direction
    double jw;     // this is the normalized distance to the nearest cell index to the left in the y direction
    double kw;     // this is the normalized distance to the nearest cell index to the left in the z direction
    int ip;     // this is the counter to the next cell in the x direction, is set to zero to cause calculations to work but not reference outside of arrays if nx = 1
    int jp;     // this is the counter to the next cell in the y direction, is set to zero to cause calculations to work but not reference outside of arrays if ny = 1
    int kp;     // this is the counter to the next cell in the z direction, is set to zero to cause calculations to work but not reference outside of arrays if nz = 1

    // I keep wondering, since we never use the gradients again since they are just used to calculate flux_div, which is what is used instead
    // at some point in time should we get rid of storage of the gradient datasets? I guess they are useful for debugging.
    void setDX_1D(const Turb* turb, const int idx);
    void setDY_1D(const Turb* turb, const int idx);
    void setDZ_1D(const Turb* turb, const int idx);

    void setDX_Forward(const Turb* turb, const int idx);    // second order forward differencing for calc gradient in the x direction of tau
    void setDY_Forward(const Turb* turb, const int idx);    // second order forward differencing for calc gradient in the y direction of tau
    void setDZ_Forward(const Turb* turb, const int idx);    // second order forward differencing for calc gradient in the z direction of tau

    void setDX_Backward(const Turb* turb, const int idx);   // second order backward differencing for calc gradient in the x direction of tau
    void setDY_Backward(const Turb* turb, const int idx);   // second order backward differencing for calc gradient in the y direction of tau
    void setDZ_Backward(const Turb* turb, const int idx);   // second order backward differencing for calc gradient in the z direction of tau

    void createTauGrads(Urb*,Turb*);
    void createFluxDiv();       // this function takes the TauGrads and turns them into a bunch simpler values to use
    
};


inline void Eulerian::setDX_1D(const Turb* turb, const int idx)
{
    taudx.at(idx).e11 = 0.0;
    taudx.at(idx).e12 = 0.0;
    taudx.at(idx).e13 = 0.0;
    taudx.at(idx).e22 = 0.0;
    taudx.at(idx).e23 = 0.0;
    taudx.at(idx).e33 = 0.0;
}

// second order forward differencing for calc gradient in the x direction of tau
inline void Eulerian::setDX_Forward(const Turb* turb, const int idx)
{
    int idx_xp1 = idx+1;
    int idx_xp2 = idx+2;
    
    taudx.at(idx).e11 = ( -3.0*turb->tau.at(idx).e11 + 4.0*turb->tau.at(idx_xp1).e11 - turb->tau.at(idx_xp2).e11 ) * 0.5 / dx;
    taudx.at(idx).e12 = ( -3.0*turb->tau.at(idx).e12 + 4.0*turb->tau.at(idx_xp1).e12 - turb->tau.at(idx_xp2).e12 ) * 0.5 / dx;
    taudx.at(idx).e13 = ( -3.0*turb->tau.at(idx).e13 + 4.0*turb->tau.at(idx_xp1).e13 - turb->tau.at(idx_xp2).e13 ) * 0.5 / dx;
    taudx.at(idx).e22 = ( -3.0*turb->tau.at(idx).e22 + 4.0*turb->tau.at(idx_xp1).e22 - turb->tau.at(idx_xp2).e22 ) * 0.5 / dx;
    taudx.at(idx).e23 = ( -3.0*turb->tau.at(idx).e23 + 4.0*turb->tau.at(idx_xp1).e23 - turb->tau.at(idx_xp2).e23 ) * 0.5 / dx;
    taudx.at(idx).e33 = ( -3.0*turb->tau.at(idx).e33 + 4.0*turb->tau.at(idx_xp1).e33 - turb->tau.at(idx_xp2).e33 ) * 0.5 / dx;
}

// second order backward differencing for calc gradient in the x direction of tau
inline void Eulerian::setDX_Backward(const Turb* turb, const int idx)
{
    int idx_xm1 = idx-1;
    int idx_xm2 = idx-2;
                
    taudx.at(idx).e11 = ( 3.0*turb->tau.at(idx).e11 - 4.0*turb->tau.at(idx_xm1).e11 + turb->tau.at(idx_xm2).e11 ) * 0.5 / dx;
    taudx.at(idx).e12 = ( 3.0*turb->tau.at(idx).e12 - 4.0*turb->tau.at(idx_xm1).e12 + turb->tau.at(idx_xm2).e12 ) * 0.5 / dx;
    taudx.at(idx).e13 = ( 3.0*turb->tau.at(idx).e13 - 4.0*turb->tau.at(idx_xm1).e13 + turb->tau.at(idx_xm2).e13 ) * 0.5 / dx;
    taudx.at(idx).e22 = ( 3.0*turb->tau.at(idx).e22 - 4.0*turb->tau.at(idx_xm1).e22 + turb->tau.at(idx_xm2).e22 ) * 0.5 / dx;
    taudx.at(idx).e23 = ( 3.0*turb->tau.at(idx).e23 - 4.0*turb->tau.at(idx_xm1).e23 + turb->tau.at(idx_xm2).e23 ) * 0.5 / dx;
    taudx.at(idx).e33 = ( 3.0*turb->tau.at(idx).e33 - 4.0*turb->tau.at(idx_xm1).e33 + turb->tau.at(idx_xm2).e33 ) * 0.5 / dx;
}


inline void Eulerian::setDY_1D(const Turb* turb, const int idx)
{
    taudy.at(idx).e11 = 0.0;
    taudy.at(idx).e12 = 0.0;
    taudy.at(idx).e13 = 0.0;
    taudy.at(idx).e22 = 0.0;
    taudy.at(idx).e23 = 0.0;
    taudy.at(idx).e33 = 0.0;
}

// second order forward differencing for calc gradient in the y direction of tau
inline void Eulerian::setDY_Forward(const Turb* turb, const int idx)
{
    int idx_yp1 = idx + nx;
    int idx_yp2 = idx + (2.0*nx);
                    
    taudy.at(idx).e11 = ( -3.0*turb->tau.at(idx).e11 + 4.0*turb->tau.at(idx_yp1).e11 - turb->tau.at(idx_yp2).e11 ) * 0.5 / dy;
    taudy.at(idx).e12 = ( -3.0*turb->tau.at(idx).e12 + 4.0*turb->tau.at(idx_yp1).e12 - turb->tau.at(idx_yp2).e12 ) * 0.5 / dy;
    taudy.at(idx).e13 = ( -3.0*turb->tau.at(idx).e13 + 4.0*turb->tau.at(idx_yp1).e13 - turb->tau.at(idx_yp2).e13 ) * 0.5 / dy;
    taudy.at(idx).e22 = ( -3.0*turb->tau.at(idx).e22 + 4.0*turb->tau.at(idx_yp1).e22 - turb->tau.at(idx_yp2).e22 ) * 0.5 / dy;
    taudy.at(idx).e23 = ( -3.0*turb->tau.at(idx).e23 + 4.0*turb->tau.at(idx_yp1).e23 - turb->tau.at(idx_yp2).e23 ) * 0.5 / dy;
    taudy.at(idx).e33 = ( -3.0*turb->tau.at(idx).e33 + 4.0*turb->tau.at(idx_yp1).e33 - turb->tau.at(idx_yp2).e33 ) * 0.5 / dy;
}

// second order backward differencing for calc gradient in the y direction of tau
inline void Eulerian::setDY_Backward(const Turb* turb, const int idx)
{
    int idx_ym1 = idx - nx;
    int idx_ym2 = idx - (2.0*nx);
    
    taudy.at(idx).e11 = ( 3.0*turb->tau.at(idx).e11 - 4.0*turb->tau.at(idx_ym1).e11 + turb->tau.at(idx_ym2).e11 ) * 0.5 / dy;
    taudy.at(idx).e12 = ( 3.0*turb->tau.at(idx).e12 - 4.0*turb->tau.at(idx_ym1).e12 + turb->tau.at(idx_ym2).e12 ) * 0.5 / dy;
    taudy.at(idx).e13 = ( 3.0*turb->tau.at(idx).e13 - 4.0*turb->tau.at(idx_ym1).e13 + turb->tau.at(idx_ym2).e13 ) * 0.5 / dy;
    taudy.at(idx).e22 = ( 3.0*turb->tau.at(idx).e22 - 4.0*turb->tau.at(idx_ym1).e22 + turb->tau.at(idx_ym2).e22 ) * 0.5 / dy;
    taudy.at(idx).e23 = ( 3.0*turb->tau.at(idx).e23 - 4.0*turb->tau.at(idx_ym1).e23 + turb->tau.at(idx_ym2).e23 ) * 0.5 / dy;
    taudy.at(idx).e33 = ( 3.0*turb->tau.at(idx).e33 - 4.0*turb->tau.at(idx_ym1).e33 + turb->tau.at(idx_ym2).e33 ) * 0.5 / dy;
}


inline void Eulerian::setDZ_1D(const Turb* turb, const int idx)
{
    taudz.at(idx).e11 = 0.0;
    taudz.at(idx).e12 = 0.0;
    taudz.at(idx).e13 = 0.0;
    taudz.at(idx).e22 = 0.0;
    taudz.at(idx).e23 = 0.0;
    taudz.at(idx).e33 = 0.0;
}

// second order forward differencing for calc gradient in the z direction of tau
inline void Eulerian::setDZ_Forward(const Turb* turb, const int idx)
{
    int idx_zp1 = idx + (ny*nx);
    int idx_zp2 = idx + 2.0*(ny*nx);
    
    taudz.at(idx).e11 = ( -3.0*turb->tau.at(idx).e11 + 4.0*turb->tau.at(idx_zp1).e11 - turb->tau.at(idx_zp2).e11 ) * 0.5 / dz;
    taudz.at(idx).e12 = ( -3.0*turb->tau.at(idx).e12 + 4.0*turb->tau.at(idx_zp1).e12 - turb->tau.at(idx_zp2).e12 ) * 0.5 / dz;
    taudz.at(idx).e13 = ( -3.0*turb->tau.at(idx).e13 + 4.0*turb->tau.at(idx_zp1).e13 - turb->tau.at(idx_zp2).e13 ) * 0.5 / dz;
    taudz.at(idx).e22 = ( -3.0*turb->tau.at(idx).e22 + 4.0*turb->tau.at(idx_zp1).e22 - turb->tau.at(idx_zp2).e22 ) * 0.5 / dz;
    taudz.at(idx).e23 = ( -3.0*turb->tau.at(idx).e23 + 4.0*turb->tau.at(idx_zp1).e23 - turb->tau.at(idx_zp2).e23 ) * 0.5 / dz;
    taudz.at(idx).e33 = ( -3.0*turb->tau.at(idx).e33 + 4.0*turb->tau.at(idx_zp1).e33 - turb->tau.at(idx_zp2).e33 ) * 0.5 / dz;
}

// second order backward differencing for calc gradient in the z direction of tau
inline void Eulerian::setDZ_Backward(const Turb* turb, const int idx)
{
    int idx_zm1 = idx - (ny*nx);
    int idx_zm2 = idx - 2.0*(ny*nx);
                    
    taudz.at(idx).e11 = ( 3.0*turb->tau.at(idx).e11 - 4.0*turb->tau.at(idx_zm1).e11 + turb->tau.at(idx_zm2).e11 ) * 0.5 / dz;
    taudz.at(idx).e12 = ( 3.0*turb->tau.at(idx).e12 - 4.0*turb->tau.at(idx_zm1).e12 + turb->tau.at(idx_zm2).e12 ) * 0.5 / dz;
    taudz.at(idx).e13 = ( 3.0*turb->tau.at(idx).e13 - 4.0*turb->tau.at(idx_zm1).e13 + turb->tau.at(idx_zm2).e13 ) * 0.5 / dz;
    taudz.at(idx).e22 = ( 3.0*turb->tau.at(idx).e22 - 4.0*turb->tau.at(idx_zm1).e22 + turb->tau.at(idx_zm2).e22 ) * 0.5 / dz;
    taudz.at(idx).e23 = ( 3.0*turb->tau.at(idx).e23 - 4.0*turb->tau.at(idx_zm1).e23 + turb->tau.at(idx_zm2).e23 ) * 0.5 / dz;
    taudz.at(idx).e33 = ( 3.0*turb->tau.at(idx).e33 - 4.0*turb->tau.at(idx_zm1).e33 + turb->tau.at(idx_zm2).e33 ) * 0.5 / dz;
}

#endif
