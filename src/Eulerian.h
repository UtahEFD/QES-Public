#ifndef EULERIAN_H
#define EULERIAN_H

//#include "Util.h"
#include "Urb.hpp"
#include "Turb.hpp"
#include "TypeDefs.hpp"

#include <fstream>
#include <string>
#include <vector>

#include <helper_math.h> 

class Eulerian{
    
public:
    
    Eulerian(Urb*,Turb*);   // defines the values for vonKar and zo, then copies the urb grid values for nx, ny, nz, nt, dx, dy, and dz to the Eulerian grid values,
                            // then calculates the tau gradients and the A1 matrix.
                            // probably not going to need vonKar or zo anymore. Same with A1 matrix. Eulerian grid accessing functions should be the main use of this class.
                            // since Urb and Turb stuff is not fully copied into this Eulerian dataset, some Eulerian grid accessing functions should probably be able to do calculations
                            // on datasets that are not stored in this class, but are values at this same Eulerian grid level. That being said, some of them could also be directly 
                            // put into Urb and Turb, depending on what is needed. I bet some of these functions currently reside in Plume.
    

    double vonKar;  // the von karman coefficient, used in Boundary Layer meterology theory for calculations of near ground wind fields, the friction velocity ustar, and other Surface Layer variables
    
    typedef struct{ // not sure what this is yet
        int c;
    }cell;	
    std::vector<cell> CellType,CellBuild;   // this is not used in Eulerian.cpp, probably set/called from inside Plume or Dispersion?


    // 
    // New code variables here
    //
    



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



    std::vector<matrix9> eigVec,eigVecInv;      // I believe these are the eigenvector and inverse eigenvector of the A1 matrix, so probably won't be used
    

    
    std::vector<double> ustar,dudz;             // this is the friction velocity, and the gradient of the mean wind along the ground. Both are usually used in BL theory for value estimation along the ground
                                                // so probably won't be used, but maybe when we add in deposition, it comes back
                                                // actually, these are not used in Eulerian.cpp, dispersion, or plume!!! Must be leftovers from turb
    
    std::vector<diagonal> eigVal;               // I believe this is the eigenvalue of the A1 matrix, so probably won't be used
    
    std::vector<vec3> ka0,g2nd;                 // no clue what these are. ka0 comes from a vector multiply of the eigenvector inverse and some a0 variable, in the calc A1 matrix function
                                                // g2nd seems to be adding a bunch of the stress tensor components together, also done in the calc A1 matrix function
                                                // I suspect these are temporary variables for the calc A1 matrix function. That being said, they are both used in the old Plume.
    vec3 windP,windPRot;                // not sure what these are, but they are not used in Eulerian.cpp, just in Plume.cpp. They are used with something called uPrime, vPrime, wPrime.
                                        // these prime values are used in dispersion to set the initial values of new particles
                                        // this is just an e11, e21, e31 struct of values. Seems to be used like a temporary variable somehow
                                        // whatever the heck this is being used for makes no sense to me.
    
    void display(const matrix9&);       // you would think this should be a function put with the datatype. Seems useful, especially in debug mode
    void display(const matrix6&);       // you would think this should be a function put with the datatype. Seems useful, especially in debug mode
    void display(const vec3&);       // you would think this should be a function put with the datatype. Seems useful, especially in debug mode
    void display(const diagonal&);       // you would think this should be a function put with the datatype. Seems useful, especially in debug mode
    
    vec3 matrixVecMult(const matrix9&, const vec3&);    // since this involves two datatypes, not sure if you can put it into a single class. Is matrix multiplication of two datatypes
    std::vector<double> zInMeters;          // this isn't used anywhere
    matrix9 matrixInv(const matrix6&);      // since this involves two datatypes, not sure if you can put it into a single class. Is getting the inverse of the matrix6 datatype out as a matrix 9 value
    double matrixDet(const matrix6&);       // seems like this should be a function put with the datatype. Is getting the determinant of a matrix 6 type
    double matrixDet(const matrix9&);       // seems like this should be a function put with the datatype. Is getting the determinant of a matrix 9 type
    double matNormFro(const matrix9&);      // seems like this should be a function put with the datatype. I'm a bit confused by it. Looks like getting a transform of the matrix, performing matrix multiplication, then the norm, but it isn't a transform of the matrix. It ends up being the same values of the matrix. So is it matrix^2 then norm?
    double matNormFro(const matrix6&);      // seems like this should be a function put with the datatype. I'm a bit confused by it. Looks like getting a transform of the matrix, performing matrix multiplication, then the norm, but it isn't a transform of the matrix. It ends up being the same values of the matrix. So is it matrix^2 then norm?
    double matCondFro(const matrix6& mat);       // seems like this should be a function put with the datatype. I think this is calculating the condition of the matrix? It uses the matNormFro stuff in the calulation.
    double matCondFro(matrix9& mat,double);       // seems like this should be a function put with the datatype. I think this is calculating the condition of the matrix? It uses the matNormFro stuff in the calulation.
    
    int nx,ny,nz,nt;    // a copy of the urb grid information, the number of values in each dimension, including time
    double zo;      // a constant representing the roughness length
    double dx,dy,dz;    // a copy of the urb grid information, the difference between points in the grid


    void setInterp3Dindexing(const float3& xyz_particle);
    double interp3D(const std::vector<double>& EulerData);
    vec3 interp3D(const std::vector<vec3>& EulerData);
    diagonal interp3D(const std::vector<diagonal>& EulerData);
    matrix6 interp3D(const std::vector<matrix6>& EulerData);
    Wind interp3D(const std::vector<Wind>& EulerData);
    float3 interp3D(const std::vector<float3>& EulerData);  // this one is causing me troubles, this datatype doesn't appear to be defined and I'm having trouble with it
    
 
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

        
    void setDX_Forward(const Turb* turb, const int idx);    // second order forward differencing for calc gradient in the x direction of tau
    void setDY_Forward(const Turb* turb, const int idx);    // second order forward differencing for calc gradient in the y direction of tau
    void setDZ_Forward(const Turb* turb, const int idx);    // second order forward differencing for calc gradient in the z direction of tau

    void setDX_Backward(const Turb* turb, const int idx);   // second order backward differencing for calc gradient in the x direction of tau
    void setDY_Backward(const Turb* turb, const int idx);   // second order backward differencing for calc gradient in the y direction of tau
    void setDZ_Backward(const Turb* turb, const int idx);   // second order backward differencing for calc gradient in the z direction of tau

    void createUstar();         // this function doesn't exist
    void createTausAndLamdas(); // this function doesn't exist
    void createTauGrads(Urb*,Turb*);
    void createFluxDiv();       // this function takes the TauGrads and turns them into a bunch simpler values to use
    void writeSigmas();         // this function doesn't exist
    void createA1Matrix(Urb*,Turb*);        // this is ugly as heck. Looks like it starts out by setting a bunch of the A values. But it is confusing cause it seems to break it down into parts, then add stuff back together at the end.
                                            // takes 100 lines of code or so just to set the initial values of this A matrix. But these are all temporary values. It then stuffs the values into the real A matrix, and does some kind of imaginary value check
                                            // but at the same time, nothing is ever making it imaginary or no. instead it looks like a bunch of checks to make sure the matrix is realizable, and to swap values around with sort/swapping to make it realizable
                                            // okay so it is setting the values, then doing a HUGE list of checks to make sure it is doable or warning if it is not. Then calculates a bunch
                                            // of other values to be used later like the eigenvectors, eigenvalues, this strange ka0 and g2nd.
    void swap(double&,double&);     // moves two values around. Looks like this was used for swapping whole rows of the A matrix to make matrix math more stable. Not sure if we need this or not once we have the makeRealizable stuff going. Or I guess it can be part of makeRealizeable?
    
    // these look like useful functions for a matrix9 dataset, should probably put them in this matrix9 dataset. Probably going to use each of these functions in Brian's code implementation
    matrix9 matrixInv(matrix9&,double);
    matrix9 matrixMult(const matrix9&,const matrix9&);
    matrix9 matrixScalarMult(const matrix9&,const double&);
    matrix9 matrixSubs(const matrix9&,const matrix9&);  // only used in Eulerian.cpp. An interesting function, it is taking the difference between two matrix9 datasets. Shouldn't this be placed in the matrix9 function sets?
    
    // some vec3 functions that should probably be put with the datatype. I guess that depends on if we still need this stuff since we still compute an A1 matrix, but now in the moment instead of ahead of time.
    double vecNorm(const vec3&);                    // only used in Eulerian.cpp. Looks like taking a vec3 and computing the norm. Shouldn't this be added to the vec3 dataype list of functions?
    vec3 vecScalarDiv(const vec3&, const double&);      // only used in Eulerian.cpp. Looks like taking a vec3 and dividing each component by a scalar. Shouldn't this be added to the vec3 dataype list of functions?
    vec3 vecScalarMult(const vec3&, const double&);     // only used in Eulerian.cpp. Plume.cpp defines it's own version. Looks like taking a vec3 and multiplying each value by a scalar. Shouldn't this be placed in the vec3 datatype?
    double maxValAbs(const vec3&);          // only used in Eulerian.cpp when performing checks on the A1 matrix. Looks like a simple function just looking for which is the greatest absolute max of a three valued vector. Not sure if needs to be kept or no. Shouldn't this be placed in the vec3 datatype?
};

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
