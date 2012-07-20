#ifndef EULERIAN_H
#define EULERIAN_H

#include "Util.h"
#include <fstream>
#include <string>
#include <vector>
class eulerian{
    
public:
    eulerian();
    double vonKar;
    typedef struct{
        int c;
    }cell;	
    std::vector<cell> CellType,CellBuild;
    

    typedef struct{
        double u;
        double v;
        double w;
    }wind;	
    
    typedef struct{
        double e11;
        double e22;
        double e33;
    }diagonal;
    std::vector<wind> windVec;
    typedef struct{
        double e11;
        double e12;
        double e13;
        double e21;
        double e22;
        double e23;
        double e31;
        double e32;
        double e33;
    }matrix9;	
    std::vector<matrix9> eigVec,eigVecInv,lam;
    typedef struct{
        double e11;
        double e12;
        double e13;
        double e22;
        double e23;
        double e33;
    }matrix6;
    std::vector<matrix6> sig,tau,taudx,taudy,taudz;
    std::vector<double> CoEps,ustar,dudz;
    
    
    std::vector<diagonal> eigVal;
    
    typedef struct{
        double e11;
        double e21;
        double e31;
    }vec3;
    std::vector<vec3> ka0,g2nd;
    vec3 windP,windPRot;
    
    util utl;  
    void createEul(const util&);
    vec3 matrixVecMult(const matrix9&, const vec3&);
    void display(const matrix9&);
    void display(const matrix6&);
    void display(const vec3&);
    void display(const diagonal&);
    std::vector<double> zInMeters; 
    matrix9 matrixInv(const matrix6&);
    double matrixDet(const matrix6&);
    double matrixDet(const matrix9&);
    double matNormFro(const matrix9&);
    double matNormFro(const matrix6&);
    double matCondFro(const matrix6& mat);
    double matCondFro(const matrix9& mat);
    
    friend std::ostream &operator<<(std::ostream& os,eulerian &eul)
    {
      os <<std::endl << '[' << eul.tau[10].e11 << ' ' << eul.tau[10].e12 << ' ' << eul.tau[10].e13 << ']'<<std::endl
       << '[' <<0 << ' ' << eul.tau[10].e22<< ' ' << eul.tau[10].e23 << ']'<<std::endl 
       << '[' << 0  << ' ' <<0 << ' ' << eul.tau[10].e33<<  ']'<<std::endl ;
      return os;
      
    }
    
    
    
    
    int windField,nx,ny,nz;
 
private:   
    double zo,dx,dy,dz;
    
    void createWindField();
    void readCellType();
    void addBuildingsInWindField();
    void createSigmaAndEps();
    void createSigmaAndEpsQUIC();
    void createSigmaAndEpsQUICFull();
    void createUstar();
    void createTausAndLamdas();
    void createTauGrads();
    void writeSigmas();
    void createA1Matrix();
    void swap(double&,double&);
    void uniform();
    void shear();
    void windFromQUIC();
    
    matrix9 matrixInv(const matrix9&);
    matrix9 matrixMult(const matrix9&,const matrix9&);
    matrix9 matrixScalarMult(const matrix9&,const double&);
    matrix9 matrixSubs(const matrix9&,const matrix9&);
    
    double vecNorm(const vec3&);
    vec3 vecScalarDiv(const vec3&, const double&);
    vec3 vecScalarMult(const vec3&, const double&);
    double maxValAbs(const vec3&);
    void writeFile(const std::vector<matrix9>&,const char*);
    void writeFile(const std::vector<matrix6>&,const char*);
    void writeFile(const std::vector<diagonal>&,const char*);
    void writeFile(const std::vector<wind>&,const char*);
    
    
};
#endif
