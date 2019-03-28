//
//  Turb.hpp
//  
//  This class represents CUDA-TURB fields
//
//  Created by Jeremy Gibbs on 03/25/19.
//

#include <iostream>
#include "Turb.hpp"

using namespace netCDF;
using namespace netCDF::exceptions;

Turb :: Turb(Input* input) {
    std::cout<<"[Turb] \t\t Loading CUDA-TURB fields "<<std::endl;
    
    // get input dimensions
    input->getDimensionSize("x",grid.nx);
    input->getDimensionSize("y",grid.ny);
    input->getDimensionSize("z",grid.nz);
    input->getDimensionSize("t",grid.nt);
    
    int c = grid.nt*grid.nz*grid.ny*grid.nx;
    start = {0,0,0,0};
    count = {static_cast<unsigned long>(grid.nt),
             static_cast<unsigned long>(grid.nz),
             static_cast<unsigned long>(grid.ny),
             static_cast<unsigned long>(grid.nx)};
    
    // get input data and information
    grid.x.resize(grid.nx);
    grid.y.resize(grid.ny);
    grid.z.resize(grid.nz);
    grid.t.resize(grid.nt);
    tau.resize(grid.nt*grid.nz*grid.ny*grid.nx);
    sig.resize(grid.nt*grid.nz*grid.ny*grid.nx);
    lam.resize(grid.nt*grid.nz*grid.ny*grid.nx);
    CoEps.resize(grid.nt*grid.nz*grid.ny*grid.nx);
            
    input->getVariableData("x",grid.x);
    input->getVariableData("y",grid.y);
    input->getVariableData("z",grid.z);
    input->getVariableData("t",grid.t);
    input->getVariableData("CoEps",start,count,CoEps);
    
    // read in data
    std::vector<double> tau1(c),tau2(c),tau3(c),tau4(c),tau5(c),tau6(c);
    std::vector<double> sig1(c),sig2(c),sig3(c),sig4(c),sig5(c),sig6(c);
    std::vector<double> lam1(c),lam2(c),lam3(c),lam4(c),lam5(c),lam6(c),
                        lam7(c),lam8(c),lam9(c);
    
    input->getVariableData("tau_11",start,count,tau1);
    input->getVariableData("tau_12",start,count,tau2);
    input->getVariableData("tau_13",start,count,tau3);
    input->getVariableData("tau_22",start,count,tau4);
    input->getVariableData("tau_23",start,count,tau5);
    input->getVariableData("tau_33",start,count,tau6);
    input->getVariableData("sig_11",start,count,sig1);
    input->getVariableData("sig_12",start,count,sig2);
    input->getVariableData("sig_13",start,count,sig3);
    input->getVariableData("sig_22",start,count,sig4);
    input->getVariableData("sig_23",start,count,sig5);
    input->getVariableData("sig_33",start,count,sig6);
    input->getVariableData("lam_11",start,count,lam1);
    input->getVariableData("lam_12",start,count,lam2);
    input->getVariableData("lam_13",start,count,lam3);
    input->getVariableData("lam_21",start,count,lam4);
    input->getVariableData("lam_22",start,count,lam5);
    input->getVariableData("lam_23",start,count,lam6);
    input->getVariableData("lam_31",start,count,lam7);
    input->getVariableData("lam_32",start,count,lam8);
    input->getVariableData("lam_33",start,count,lam9);
    
    int id;
    for (int n=0;n<grid.nt;n++) {
        for(int k=0;k<grid.nz;k++) {
            for(int j=0; j<grid.ny;j++) { 
                for(int i=0;i<grid.nx;i++){  
                    
                    id = i + j*grid.nx + k*grid.nx*grid.ny + n*grid.nx*grid.ny*grid.nz;
                    
                    tau.at(id).e11 = tau1.at(id);
                    tau.at(id).e12 = tau2.at(id);
                    tau.at(id).e13 = tau3.at(id);
                    tau.at(id).e22 = tau4.at(id);
                    tau.at(id).e23 = tau5.at(id);
                    tau.at(id).e33 = tau6.at(id);
                    sig.at(id).e11 = sig1.at(id);
                    sig.at(id).e12 = sig2.at(id);
                    sig.at(id).e13 = sig3.at(id);
                    sig.at(id).e22 = sig4.at(id);
                    sig.at(id).e23 = sig5.at(id);
                    sig.at(id).e33 = sig6.at(id);
                    lam.at(id).e11 = lam1.at(id);
                    lam.at(id).e12 = lam2.at(id);
                    lam.at(id).e13 = lam3.at(id);
                    lam.at(id).e21 = lam4.at(id);
                    lam.at(id).e22 = lam5.at(id);
                    lam.at(id).e23 = lam6.at(id);
                    lam.at(id).e31 = lam7.at(id);
                    lam.at(id).e32 = lam8.at(id);
                    lam.at(id).e33 = lam9.at(id);
                }
            }
        }
    }
    
    // clean up temporary vectors
    tau1.clear();
    tau2.clear();
    tau3.clear();
    tau4.clear();
    tau5.clear();
    tau6.clear();
    sig1.clear();
    sig2.clear();
    sig3.clear();
    sig4.clear();
    sig5.clear();
    sig6.clear();
    lam1.clear();
    lam2.clear();
    lam3.clear();
    lam4.clear();
    lam5.clear();
    lam6.clear();
    lam7.clear();
    lam8.clear();
    lam9.clear();
}