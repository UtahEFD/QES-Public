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

    std::vector<matrix9> eigVec,eigVecInv;
    
    std::vector<matrix6> taudx,taudy,taudz;
    
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
#endif
