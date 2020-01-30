#ifndef EULERIAN_H
#define EULERIAN_H


#include <iostream>
#include <ctime>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <chrono>

#include "Random.h"

#include "PlumeInputData.hpp"
#include "Urb.hpp"
#include "Turb.hpp"


class Eulerian{
    
    public:
        
        Eulerian(Urb*,Turb*,PlumeInputData*,const std::string& debugOutputFolder);   // copies the turb grid values for nx, ny, nz, nt, dx, dy, and dz to the Eulerian grid values,
                                // then calculates the tau gradients which are then used to calculate the flux_div grid values.
        

        // the Eulerian data held in this class is on the turb grid, so these are copies of the turb grid values
        int nx;     // a copy of the turb grid information. This is the number of points in the x dimension
        int ny;     // a copy of the turb grid information. This is the number of points in the y dimension
        int nz;     // a copy of the turb grid information. This is the number of points in the z dimension
        int nt;     // a copy of the turb grid information. This is the number of times for which the x,y, and z values are repeated
        double dx;      // a copy of the turb grid information. This is the difference between points in the x dimension, eventually could become an array
        double dy;      // a copy of the turb grid information. This is the difference between points in the y dimension, eventually could become an array
        double dz;      // a copy of the turb grid information. This is the difference between points in the z dimension, eventually could become an array
        double turbXstart;      // a copy of the turb grid information. The turb starting x value. Is not necessarily the turb domain starting x value because it could be cell centered.
        double turbYstart;      // a copy of the turb grid information. The turb starting y value. Is not necessarily the turb domain starting y value because it could be cell centered.
        double turbZstart;      // a copy of the turb grid information. The turb starting z value. Is not necessarily the turb domain starting z value because it could be cell centered.
        double turbXend;        // a copy of the turb grid information. The turb ending x value. Is not necessarily the turb domain ending x value because it could be cell centered.
        double turbYend;        // a copy of the turb grid information. The turb ending y value. Is not necessarily the turb domain ending y value because it could be cell centered.
        double turbZend;        // a copy of the turb grid information. The turb ending z value. Is not necessarily the turb domain ending z value because it could be cell centered.


        // other input variable
        double C_0;     // a copy of the turb grid information. This is used to separate out CoEps into its separate parts when doing debug output


        // my description of the flux_div might need to be corrected
        std::vector<double> flux_div_x;     // this is like the derivative of the forces acting on the x face
        std::vector<double> flux_div_y;     // this is like the derivative of the forces acting on the y face
        std::vector<double> flux_div_z;     // this is like the derivative of the forces acting on the z face


        void setInterp3Dindexing(const double& par_xPos, const double& par_yPos, const double& par_zPos);
        double interp3D(const std::vector<double>& EulerData,const std::string& dataName);

        
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


        // these are the gradients of many components of tau in many different direction. Tau is still kept inside Turb
        // this is only the derivatives that matter for calculating the flux_div
        // since this is just a temporary variable for calculating flux_div, it may be better to move this into the functions directly at some point
        // units are probably m/s^2
        // notice that tau is a symmetric tensor, so the flux div for each direction is taking a given x, y, or z face, 
        //  then calculating the derivative of each stress on that face in the given direction of that stress.
        //  because of symmetry, the storage names don't always show this, so the symmetrical names are given in comments to make it clearer.
        std::vector<double> dtxxdx; // dtxxdx
        std::vector<double> dtxydy; // dtxydy
        std::vector<double> dtxzdz; // dtxzdz

        std::vector<double> dtxydx; // dtyxdx
        std::vector<double> dtyydy; // dtyydy
        std::vector<double> dtyzdz; // dtyzdz

        std::vector<double> dtxzdx; // dtzxdx
        std::vector<double> dtyzdy; // dtzydy
        std::vector<double> dtzzdz; // dtzzyz

        
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
        

        void outputVarInfo_text(Urb* urb, Turb* turb, const std::string& outputFolder);
    
};


inline void Eulerian::setDX_1D(const Turb* turb, const int idx)
{
    dtxxdx.at(idx) = 0.0;
    dtxydx.at(idx) = 0.0;
    dtxzdx.at(idx) = 0.0;
}

// second order forward differencing for calc gradient in the x direction of tau
inline void Eulerian::setDX_Forward(const Turb* turb, const int idx)
{
    int idx_xp1 = idx+1;
    int idx_xp2 = idx+2;
    
    dtxxdx.at(idx) = ( -3.0*turb->txx.at(idx) + 4.0*turb->txx.at(idx_xp1) - turb->txx.at(idx_xp2) ) * 0.5 / dx;
    dtxydx.at(idx) = ( -3.0*turb->txy.at(idx) + 4.0*turb->txy.at(idx_xp1) - turb->txy.at(idx_xp2) ) * 0.5 / dx;
    dtxzdx.at(idx) = ( -3.0*turb->txz.at(idx) + 4.0*turb->txz.at(idx_xp1) - turb->txz.at(idx_xp2) ) * 0.5 / dx;
}

// second order backward differencing for calc gradient in the x direction of tau
inline void Eulerian::setDX_Backward(const Turb* turb, const int idx)
{
    int idx_xm1 = idx-1;
    int idx_xm2 = idx-2;
                
    dtxxdx.at(idx) = ( 3.0*turb->txx.at(idx) - 4.0*turb->txx.at(idx_xm1) + turb->txx.at(idx_xm2) ) * 0.5 / dx;
    dtxydx.at(idx) = ( 3.0*turb->txy.at(idx) - 4.0*turb->txy.at(idx_xm1) + turb->txy.at(idx_xm2) ) * 0.5 / dx;
    dtxzdx.at(idx) = ( 3.0*turb->txz.at(idx) - 4.0*turb->txz.at(idx_xm1) + turb->txz.at(idx_xm2) ) * 0.5 / dx;
}


inline void Eulerian::setDY_1D(const Turb* turb, const int idx)
{
    dtxydy.at(idx) = 0.0;
    dtyydy.at(idx) = 0.0;
    dtyzdy.at(idx) = 0.0;
}

// second order forward differencing for calc gradient in the y direction of tau
inline void Eulerian::setDY_Forward(const Turb* turb, const int idx)
{
    int idx_yp1 = idx + nx;
    int idx_yp2 = idx + (2.0*nx);
                    
    dtxydy.at(idx) = ( -3.0*turb->txy.at(idx) + 4.0*turb->txy.at(idx_yp1) - turb->txy.at(idx_yp2) ) * 0.5 / dy;
    dtyydy.at(idx) = ( -3.0*turb->tyy.at(idx) + 4.0*turb->tyy.at(idx_yp1) - turb->tyy.at(idx_yp2) ) * 0.5 / dy;
    dtyzdy.at(idx) = ( -3.0*turb->tyz.at(idx) + 4.0*turb->tyz.at(idx_yp1) - turb->tyz.at(idx_yp2) ) * 0.5 / dy;
}

// second order backward differencing for calc gradient in the y direction of tau
inline void Eulerian::setDY_Backward(const Turb* turb, const int idx)
{
    int idx_ym1 = idx - nx;
    int idx_ym2 = idx - (2.0*nx);
    
    dtxydy.at(idx) = ( 3.0*turb->txy.at(idx) - 4.0*turb->txy.at(idx_ym1) + turb->txy.at(idx_ym2) ) * 0.5 / dy;
    dtyydy.at(idx) = ( 3.0*turb->tyy.at(idx) - 4.0*turb->tyy.at(idx_ym1) + turb->tyy.at(idx_ym2) ) * 0.5 / dy;
    dtyzdy.at(idx) = ( 3.0*turb->tyz.at(idx) - 4.0*turb->tyz.at(idx_ym1) + turb->tyz.at(idx_ym2) ) * 0.5 / dy;
}


inline void Eulerian::setDZ_1D(const Turb* turb, const int idx)
{
    dtxzdz.at(idx) = 0.0;
    dtyzdz.at(idx) = 0.0;
    dtzzdz.at(idx) = 0.0;
}

// second order forward differencing for calc gradient in the z direction of tau
inline void Eulerian::setDZ_Forward(const Turb* turb, const int idx)
{
    int idx_zp1 = idx + (ny*nx);
    int idx_zp2 = idx + 2.0*(ny*nx);
    
    dtxzdz.at(idx) = ( -3.0*turb->txz.at(idx) + 4.0*turb->txz.at(idx_zp1) - turb->txz.at(idx_zp2) ) * 0.5 / dz;
    dtyzdz.at(idx) = ( -3.0*turb->tyz.at(idx) + 4.0*turb->tyz.at(idx_zp1) - turb->tyz.at(idx_zp2) ) * 0.5 / dz;
    dtzzdz.at(idx) = ( -3.0*turb->tzz.at(idx) + 4.0*turb->tzz.at(idx_zp1) - turb->tzz.at(idx_zp2) ) * 0.5 / dz;
}

// second order backward differencing for calc gradient in the z direction of tau
inline void Eulerian::setDZ_Backward(const Turb* turb, const int idx)
{
    int idx_zm1 = idx - (ny*nx);
    int idx_zm2 = idx - 2.0*(ny*nx);
                    
    dtxzdz.at(idx) = ( 3.0*turb->txz.at(idx) - 4.0*turb->txz.at(idx_zm1) + turb->txz.at(idx_zm2) ) * 0.5 / dz;
    dtyzdz.at(idx) = ( 3.0*turb->tyz.at(idx) - 4.0*turb->tyz.at(idx_zm1) + turb->tyz.at(idx_zm2) ) * 0.5 / dz;
    dtzzdz.at(idx) = ( 3.0*turb->tzz.at(idx) - 4.0*turb->tzz.at(idx_zm1) + turb->tzz.at(idx_zm2) ) * 0.5 / dz;
}

#endif
