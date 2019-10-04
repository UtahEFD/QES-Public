#ifndef EULERIAN_H
#define EULERIAN_H

//#include "Util.h"
#include "Urb.hpp"
#include "Turb.hpp"
#include "TypeDefs.hpp"

#include <fstream>
#include <string>
#include <vector>

class Eulerian{
    
public:
    
    Eulerian(Urb*,Turb*);
    
    double vonKar;
    typedef struct{
        int c;
    }cell;	
    std::vector<cell> CellType,CellBuild;


    // 
    // New code variables here
    //
    



    // Still need Tau variables.  They are defined here:
    std::vector<matrix6> taudx,taudy,taudz;



    std::vector<matrix9> eigVec,eigVecInv;
    

    
    std::vector<double> ustar,dudz;
    
    std::vector<diagonal> eigVal;
    
    std::vector<vec3> ka0,g2nd;
    vec3 windP,windPRot;
    
    void display(const matrix9&);
    void display(const matrix6&);
    void display(const vec3&);
    void display(const diagonal&);
    
    vec3 matrixVecMult(const matrix9&, const vec3&);
    std::vector<double> zInMeters; 
    matrix9 matrixInv(const matrix6&);
    double matrixDet(const matrix6&);
    double matrixDet(const matrix9&);
    double matNormFro(const matrix9&);
    double matNormFro(const matrix6&);
    double matCondFro(const matrix6& mat);
    double matCondFro(matrix9& mat,double);
    
    int nx,ny,nz,nt;
    double zo,dx,dy,dz;
 
private:   
        
    void setDX_Forward(const Turb* turb, const int idx);
    void setDY_Forward(const Turb* turb, const int idx);
    void setDZ_Forward(const Turb* turb, const int idx);

    void setDX_Backward(const Turb* turb, const int idx);
    void setDY_Backward(const Turb* turb, const int idx);
    void setDZ_Backward(const Turb* turb, const int idx);

    void createUstar();
    void createTausAndLamdas();
    void createTauGrads(Urb*,Turb*);
    void writeSigmas();
    void createA1Matrix(Urb*,Turb*);
    void swap(double&,double&);
    
    matrix9 matrixInv(matrix9&,double);
    matrix9 matrixMult(const matrix9&,const matrix9&);
    matrix9 matrixScalarMult(const matrix9&,const double&);
    matrix9 matrixSubs(const matrix9&,const matrix9&);
    
    double vecNorm(const vec3&);
    vec3 vecScalarDiv(const vec3&, const double&);
    vec3 vecScalarMult(const vec3&, const double&);
    double maxValAbs(const vec3&);
};

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

inline void Eulerian::setDX_Backward(const Turb* turb, const int idx)
{
    int idx_xm1 = idx-1;
    int idx_xm2 = idx-2;
                
    taudy.at(idx).e11 = ( 3.0*turb->tau.at(idx).e11 - 4.0*turb->tau.at(idx_xm1).e11 + turb->tau.at(idx_xm2).e11 ) * 0.5 / dx;
    taudy.at(idx).e12 = ( 3.0*turb->tau.at(idx).e12 - 4.0*turb->tau.at(idx_xm1).e12 + turb->tau.at(idx_xm2).e12 ) * 0.5 / dx;
    taudy.at(idx).e13 = ( 3.0*turb->tau.at(idx).e13 - 4.0*turb->tau.at(idx_xm1).e13 + turb->tau.at(idx_xm2).e13 ) * 0.5 / dx;
    taudy.at(idx).e22 = ( 3.0*turb->tau.at(idx).e22 - 4.0*turb->tau.at(idx_xm1).e22 + turb->tau.at(idx_xm2).e22 ) * 0.5 / dx;
    taudy.at(idx).e23 = ( 3.0*turb->tau.at(idx).e23 - 4.0*turb->tau.at(idx_xm1).e23 + turb->tau.at(idx_xm2).e23 ) * 0.5 / dx;
    taudy.at(idx).e33 = ( 3.0*turb->tau.at(idx).e33 - 4.0*turb->tau.at(idx_xm1).e33 + turb->tau.at(idx_xm2).e33 ) * 0.5 / dx;
}

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
