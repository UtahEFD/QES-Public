#include <iostream>
#include <ctime>
#include <cmath>

#include <chrono>

#include "Eulerian.h"
#include "Random.h"
#include "Urb.hpp"
#include "Turb.hpp"

Eulerian::Eulerian(Urb* urb, Turb* turb, const std::string& debugOutputFolder) {
    
    std::cout<<"[Eulerian] \t Reading CUDA-URB & CUDA-TURB fields "<<std::endl;
    
    
    // grid information
    nt = urb->grid.nt;
    nz = urb->grid.nz;
    ny = urb->grid.ny;
    nx = urb->grid.nx;
    dz = urb->grid.dz;
    dy = urb->grid.dy;
    dx = urb->grid.dx;

    // get the urb domain start and end values, needed for wall boundary condition application
    domainXstart = urb->domainXstart;
    domainYstart = urb->domainYstart;
    domainZstart = urb->domainZstart;
    

    // compute stress gradients
    createTauGrads(urb,turb);

    // use the stress gradients to calculate the flux div
    createFluxDiv();

    // if the debug output folder is an empty string "", the debug output variables won't be written
    outputVarInfo_text(urb,turb,debugOutputFolder);

}

// This is here only so we can test the two versions a bit for
// timing... and then I will remove the older version.
#define USE_PREVIOUSCODE 1

#if USE_PREVIOUSCODE
void Eulerian::createTauGrads(Urb* urb, Turb* turb)
{
    std::cout<<"[Eulerian] \t Computing stress gradients "<<std::endl;
    
    auto timerStart = std::chrono::high_resolution_clock::now();

    taudx.resize(nx*ny*nz, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    taudy.resize(nx*ny*nz, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    taudz.resize(nx*ny*nz, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    
    // Loop over all cells in the domain up to 2 in from the edge
    for(int k=0; k<nz; ++k) {
        for(int j=0; j<ny; ++j) {
            for(int i=0; i<nx; ++i) {
                
                // Provides a linear index based on the 3D (i, j, k)
                // indices so we can access 
                int idx = k*ny*nx + j*nx + i;

                // DX components
                if (nx == 1) {
                    setDX_1D(turb, idx);
                }
                else if (i < (nx-2)) {
                    setDX_Forward(turb, idx);
                }
                else { 
                    setDX_Backward(turb, idx);
                }
                    
                // DY components
                if (ny == 1) {
                    setDY_1D(turb, idx);
                }
                else if (j < (ny-2)) {
                    setDY_Forward(turb, idx);
                }
                else { 
                    setDY_Backward(turb, idx);
                }

                // DZ components
                if (nz == 1) {
                    setDZ_1D(turb, idx);
                }
                else if (k < (nz-2)) {
                    setDZ_Forward(turb, idx);
                }
                else {
                    setDZ_Backward(turb, idx);
                }
            }
        }
    }

    auto timerEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = timerEnd - timerStart;
    std::cout << "\telapsed time: " << elapsed.count() << " s" << std::endl;   // Print out elapsed execution time
}

#else

void Eulerian::createTauGrads(Urb* urb, Turb* turb)
{
    std::cout<<"[Eulerian] \t Computing stress gradients "<<std::endl;
    
    auto timerStart = std::chrono::high_resolution_clock::now();

    taudx.resize(nx*ny*nz, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    taudy.resize(nx*ny*nz, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    taudz.resize(nx*ny*nz, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    
    // 2nd order Forward differencing up to 2 in from the edge in the direction of the gradient,
    // 2nd order Backward differencing for the last two in the direction of the gradient,
    // all this over all cells in the other two directions


    // DX forward differencing
    for(int k=0; k<nz; ++k) {
        for(int j=0; j<ny; ++j) {
            for(int i=0; i<nx-2; ++i) {
                
                // Provides a linear index based on the 3D (i, j, k)
                // indices so we can access 
                int idx = k*ny*nx + j*nx + i;

                setDX_Forward( turb, idx );

            }
        }
    }

    // DX backward differencing
    for(int k=0; k<nz; ++k) {
        for(int j=0; j<ny; ++j) {
            for(int i=nx-2; i<nx; ++i) {
                
                // Provides a linear index based on the 3D (i, j, k)
                // indices so we can access 
                int idx = k*ny*nx + j*nx + i;

                setDX_Backward( turb, idx );

            }
        }
    }


    // DY forward differencing
    for(int k=0; k<nz; ++k) {
        for(int j=0; j<ny-2; ++j) {
            for(int i=0; i<nx; ++i) {
                
                // Provides a linear index based on the 3D (i, j, k)
                // indices so we can access 
                int idx = k*ny*nx + j*nx + i;

                setDY_Forward( turb, idx );

            }
        }
    }

    // DY backward differencing
    for(int k=0; k<nz; ++k) {
        for(int j=ny-2; j<ny; ++j) {
            for(int i=0; i<nx; ++i) {
                
                // Provides a linear index based on the 3D (i, j, k)
                // indices so we can access 
                int idx = k*ny*nx + j*nx + i;

                setDY_Backward( turb, idx );

            }
        }
    }


    // DZ forward differencing
    for(int k=0; k<nz-2; ++k) {
        for(int j=0; j<ny; ++j) {
            for(int i=0; i<nx; ++i) {
                
                // Provides a linear index based on the 3D (i, j, k)
                // indices so we can access 
                int idx = k*ny*nx + j*nx + i;

                setDZ_Forward( turb, idx );

            }
        }
    }

    // DZ backward differencing
    for(int k=nz-2; k<nz; ++k) {
        for(int j=0; j<ny; ++j) {
            for(int i=0; i<nx; ++i) {
                
                // Provides a linear index based on the 3D (i, j, k)
                // indices so we can access 
                int idx = k*ny*nx + j*nx + i;

                setDZ_Backward( turb, idx );

            }
        }
    }


    auto timerEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = timerEnd - timerStart;
    std::cout << "\telapsed time: " << elapsed.count() << " s" << std::endl;   // Print out elapsed execution time
}

#endif

void Eulerian::createFluxDiv()
{

    std::cout << "[Eulerian] \t Computing flux_div values " << std::endl;

    // set the flux_div to the right size
    flux_div.resize(nx*ny*nz, {0.0, 0.0, 0.0});

    // loop through each cell and calculate the flux_div from the gradients of tao
    for(int idx = 0; idx < nx*ny*nz; idx++) {
        flux_div.at(idx).e11 = taudx.at(idx).e11 + taudy.at(idx).e12 + taudz.at(idx).e13;
        flux_div.at(idx).e21 = taudx.at(idx).e12 + taudy.at(idx).e22 + taudz.at(idx).e23;
        flux_div.at(idx).e31 = taudx.at(idx).e13 + taudy.at(idx).e23 + taudz.at(idx).e33;
    }
}

// this gets around the problem of repeated or not repeated information, just needs called once before each interpolation,
// then intepolation on all kinds of datatypes can be done
void Eulerian::setInterp3Dindexing(const vec3& xyz_particle)
{

    // the next steps are to figure out the right indices to grab the values for cube from the data, 
    // where indices are forced to be special if nx, ny, or nz are zero.
    // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

    // so this is called once before calling the interp3D function on many different datatypes
    // sets the current indices for grabbing the cube values and for interpolating with the cube,
    // but importantly sets ip,jp, and kp to zero if the number of cells in a dimension is 1
    // as this avoids referencing outside of array problems in an efficient manner
    // it also causes some stuff to be multiplied by zero so that interpolation works on any size of data without lots of if statements


    // set a particle position corrected by the start of the domain in each direction
    // the algorythm assumes the list starts at x = 0.
    vec3 par;
    par.e11 = xyz_particle.e11 - domainXstart;
    par.e21 = xyz_particle.e21 - domainYstart;
    par.e31 = xyz_particle.e31 - domainZstart;

    // index of nearest node in negative direction
    // by adding a really small number to dx, it stops it from putting
    // the stuff on the right wall of the cell into the next cell, and
    // puts everything from the left wall to the right wall of a cell
    // into the left cell. Makes it simpler for interpolation, as without this,
    // the interpolation would reference outside the array if the input position was exactly on
    // nx, ny, or nz.
    // basically adding a small number to dx shifts the indices so that instead of going
    // from 0 to nx - 1, they go from 0 to nx - 2. This means that ii + ip can at most be nx - 1
    // and only if a particle lands directly on the far boundary condition edge
    ii = floor(par.e11/(dx+1e-9));
    jj = floor(par.e21/(dy+1e-9));
    kk = floor(par.e31/(dz+1e-9));

    // fractional distance between nearest nodes
    iw = (par.e11/(dx+1e-9) - floor(par.e11/(dx+1e-9)));
    jw = (par.e21/(dy+1e-9) - floor(par.e21/(dy+1e-9)));
    kw = (par.e31/(dz+1e-9) - floor(par.e31/(dz+1e-9)));

    // initialize the counters from the indices
    ip = 1;
    jp = 1;
    kp = 1;

    // now set the indices and the counters from the indices
    if( nx == 1 )
    {
        ii = 0;
        iw = 0.0;
        ip = 0;
    }
    if( ny == 1 )
    {
        jj = 0;
        jw = 0.0;
        jp = 0;
    }
    if( nz == 1 )
    {
        kk = 0;
        kw = 0.0;
        kp = 0;
    }

    // now check to make sure that the indices are within the Eulerian grid domain
    // Notice that this no longer includes throwing an error if particles touch the far walls
    // because adding a small number to dx in the index calculation forces the index to be completely left side biased
    if( ii < 0 || ii+ip > nx-1 )
    {
        std::cerr << "ERROR (Eulerian::setInterp3Dindexing): particle x position is out of range! x = \"" << xyz_particle.e11 
            << "\" ii+ip = \"" << ii << "\"+\"" << ip << "\",   nx-1 = \"" << nx-1 << "\"" << std::endl;
        exit(1);
    }
    if( jj < 0 || jj+jp > ny-1 )
    {
        std::cerr << "ERROR (Eulerian::setInterp3Dindexing): particle y position is out of range! y = \"" << xyz_particle.e21 
            << "\" jj+jp = \"" << jj << "\"+\"" << jp << "\",   ny-1 = \"" << ny-1 << "\"" << std::endl;
        exit(1);
    }
    if( kk < 0 || kk+kp > nz-1 )
    {
        std::cerr << "ERROR (Eulerian::setInterp3Dindexing): particle z position is out of range! z = \"" << xyz_particle.e31 
            << "\" kk+kp = \"" << kk << "\"+\"" << kp << "\",   nz-1 = \"" << nz-1 << "\"" << std::endl;
        exit(1);
    }

}

// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
// the extra dataName is so that if the input is eps, the value is not allowed to be zero, but is set to two orders of magnitude bigger than sigma2
// since this would actually be CoEps, the value actually needs to be one order of magnitude bigger than sigma2?
double Eulerian::interp3D(const std::vector<double>& EulerData,const std::string& dataName)
{

    // initialize the output value
    double outputVal = 0.0;
    

    // first set a cube of size two to zero.
    // This is important because if nx, ny, or nz are only size 1, referencing two spots in cube won't reference outside the array.
    // the next steps are to figure out the right indices to grab the values for cube from the data, 
    // where indices are forced to be special if nx, ny, or nz are zero.
    // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

    
    // now set the cube to zero, then fill it using the indices and the counters from the indices
    double cube[2][2][2] = {0.0};

    // now set the cube values
    for(int kkk = 0; kkk <= kp; kkk++)
    {
        for(int jjj = 0; jjj <= jp; jjj++)
        {
            for(int iii = 0; iii <= ip; iii++)
            {
                // set the actual indices to use for the linearized Euler data
                int idx = (kk+kkk)*ny*nx + (jj+jjj)*nx + (ii+iii);
                cube[iii][jjj][kkk] = EulerData.at(idx);
            }
        }
    }

    // now do the interpolation, with the cube, the counters from the indices,
    // and the normalized width between the point locations and the closest cell left walls
    double u_low  = (1-iw)*(1-jw)*cube[0][0][0] + iw*(1-jw)*cube[1][0][0] + iw*jw*cube[1][1][0] + (1-iw)*jw*cube[0][1][0];
    double u_high = (1-iw)*(1-jw)*cube[0][0][1] + iw*(1-jw)*cube[1][0][1] + iw*jw*cube[1][1][1] + (1-iw)*jw*cube[0][1][1];
    outputVal = (u_high-u_low)*kw + u_low;

    // make sure CoEps is always bigger than zero, and eps is two orders of magnitude bigger than sigma2
    // I guess since this is actually CoEps, the value needs to be one order of magnitutde bigger than sigma2
    if( dataName == "Eps" )
    {
        if( outputVal <= 1e-6 )
        {
            outputVal = 1e-6;
        }
    }
    if( dataName == "CoEps" )
    {
        double C_0 = 4.0;
        if( outputVal <= 1e-6*C_0 )
        {
            outputVal = 1e-6*C_0;
        }
    }

    return outputVal;

}

// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
vec3 Eulerian::interp3D(const std::vector<vec3>& EulerData)
{

    // initialize the output value
    vec3 outputVal;
    outputVal.e11 = 0.0;
    outputVal.e21 = 0.0;
    outputVal.e31 = 0.0;
    

    // first set a cube of size two to zero.
    // This is important because if nx, ny, or nz are only size 1, referencing two spots in cube won't reference outside the array.
    // the next steps are to figure out the right indices to grab the values for cube from the data, 
    // where indices are forced to be special if nx, ny, or nz are zero.
    // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

    
    // now set the cube to zero, then fill it using the indices and the counters from the indices
    double cube_e11[2][2][2] = {0.0};
    double cube_e21[2][2][2] = {0.0};
    double cube_e31[2][2][2] = {0.0};

    // now set the cube values
    for(int kkk = 0; kkk <= kp; kkk++)
    {
        for(int jjj = 0; jjj <= jp; jjj++)
        {
            for(int iii = 0; iii <= ip; iii++)
            {
                // set the actual indices to use for the linearized Euler data
                int idx = (kk+kkk)*ny*nx + (jj+jjj)*nx + (ii+iii);
                cube_e11[iii][jjj][kkk] = EulerData.at(idx).e11;
                cube_e21[iii][jjj][kkk] = EulerData.at(idx).e21;
                cube_e31[iii][jjj][kkk] = EulerData.at(idx).e31;
            }
        }
    }

    // now do the interpolation, with the cube, the counters from the indices,
    // and the normalized width between the point locations and the closest cell left walls
    double u_low  = (1-iw)*(1-jw)*cube_e11[0][0][0] + iw*(1-jw)*cube_e11[1][0][0] + iw*jw*cube_e11[1][1][0] + (1-iw)*jw*cube_e11[0][1][0];
    double u_high = (1-iw)*(1-jw)*cube_e11[0][0][1] + iw*(1-jw)*cube_e11[1][0][1] + iw*jw*cube_e11[1][1][1] + (1-iw)*jw*cube_e11[0][1][1];
    outputVal.e11 = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_e21[0][0][0] + iw*(1-jw)*cube_e21[1][0][0] + iw*jw*cube_e21[1][1][0] + (1-iw)*jw*cube_e21[0][1][0];
    u_high        = (1-iw)*(1-jw)*cube_e21[0][0][1] + iw*(1-jw)*cube_e21[1][0][1] + iw*jw*cube_e21[1][1][1] + (1-iw)*jw*cube_e21[0][1][1];
    outputVal.e21 = (u_high-u_low)*kw + u_low;
    
    u_low         = (1-iw)*(1-jw)*cube_e31[0][0][0] + iw*(1-jw)*cube_e31[1][0][0] + iw*jw*cube_e31[1][1][0] + (1-iw)*jw*cube_e31[0][1][0];
    u_high        = (1-iw)*(1-jw)*cube_e31[0][0][1] + iw*(1-jw)*cube_e31[1][0][1] + iw*jw*cube_e31[1][1][1] + (1-iw)*jw*cube_e31[0][1][1];
    outputVal.e31 = (u_high-u_low)*kw + u_low;

    return outputVal;

}

// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
// the extra dataName is so that if the input is sigma2, the value is not allowed to be zero, but is set to two orders of magnitude smaller than eps
diagonal Eulerian::interp3D(const std::vector<diagonal>& EulerData,const std::string& dataName)
{
    
    // initialize the output value
    diagonal outputVal;
    outputVal.e11 = 0.0;
    outputVal.e22 = 0.0;
    outputVal.e33 = 0.0;
    

    // first set a cube of size two to zero.
    // This is important because if nx, ny, or nz are only size 1, referencing two spots in cube won't reference outside the array.
    // the next steps are to figure out the right indices to grab the values for cube from the data, 
    // where indices are forced to be special if nx, ny, or nz are zero.
    // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

    
    // now set the cube to zero, then fill it using the indices and the counters from the indices
    double cube_e11[2][2][2] = {0.0};
    double cube_e22[2][2][2] = {0.0};
    double cube_e33[2][2][2] = {0.0};

    // now set the cube values
    for(int kkk = 0; kkk <= kp; kkk++)
    {
        for(int jjj = 0; jjj <= jp; jjj++)
        {
            for(int iii = 0; iii <= ip; iii++)
            {
                // set the actual indices to use for the linearized Euler data
                int idx = (kk+kkk)*ny*nx + (jj+jjj)*nx + (ii+iii);
                cube_e11[iii][jjj][kkk] = EulerData.at(idx).e11;
                cube_e22[iii][jjj][kkk] = EulerData.at(idx).e22;
                cube_e33[iii][jjj][kkk] = EulerData.at(idx).e33;
            }
        }
    }

    // now do the interpolation, with the cube, the counters from the indices,
    // and the normalized width between the point locations and the closest cell left walls
    double u_low  = (1-iw)*(1-jw)*cube_e11[0][0][0] + iw*(1-jw)*cube_e11[1][0][0] + iw*jw*cube_e11[1][1][0] + (1-iw)*jw*cube_e11[0][1][0];
    double u_high = (1-iw)*(1-jw)*cube_e11[0][0][1] + iw*(1-jw)*cube_e11[1][0][1] + iw*jw*cube_e11[1][1][1] + (1-iw)*jw*cube_e11[0][1][1];
    outputVal.e11 = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_e22[0][0][0] + iw*(1-jw)*cube_e22[1][0][0] + iw*jw*cube_e22[1][1][0] + (1-iw)*jw*cube_e22[0][1][0];
    u_high        = (1-iw)*(1-jw)*cube_e22[0][0][1] + iw*(1-jw)*cube_e22[1][0][1] + iw*jw*cube_e22[1][1][1] + (1-iw)*jw*cube_e22[0][1][1];
    outputVal.e22 = (u_high-u_low)*kw + u_low;
    
    u_low         = (1-iw)*(1-jw)*cube_e33[0][0][0] + iw*(1-jw)*cube_e33[1][0][0] + iw*jw*cube_e33[1][1][0] + (1-iw)*jw*cube_e33[0][1][0];
    u_high        = (1-iw)*(1-jw)*cube_e33[0][0][1] + iw*(1-jw)*cube_e33[1][0][1] + iw*jw*cube_e33[1][1][1] + (1-iw)*jw*cube_e33[0][1][1];
    outputVal.e33 = (u_high-u_low)*kw + u_low;

    // make sure sigma is always bigger than zero, and two orders of magnitude smaller than Eps
    if( dataName == "sigma2" )
    {
        if( outputVal.e11 == 0.0 )
        {
            outputVal.e11 = 1e-8;
        }
        if( outputVal.e22 == 0.0 )
        {
            outputVal.e22 = 1e-8;
        }
        if( outputVal.e33 == 0.0 )
        {
            outputVal.e33 = 1e-8;
        }
    }

    return outputVal;
    
}

// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
matrix6 Eulerian::interp3D(const std::vector<matrix6>& EulerData)
{

    // initialize the output value
    matrix6 outputVal;
    outputVal.e11 = 0.0;
    outputVal.e12 = 0.0;
    outputVal.e13 = 0.0;
    outputVal.e22 = 0.0;
    outputVal.e23 = 0.0;
    outputVal.e33 = 0.0;
    

    // first set a cube of size two to zero.
    // This is important because if nx, ny, or nz are only size 1, referencing two spots in cube won't reference outside the array.
    // the next steps are to figure out the right indices to grab the values for cube from the data, 
    // where indices are forced to be special if nx, ny, or nz are zero.
    // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

    
    // now set the cube to zero, then fill it using the indices and the counters from the indices
    double cube_e11[2][2][2] = {0.0};
    double cube_e12[2][2][2] = {0.0};
    double cube_e13[2][2][2] = {0.0};
    double cube_e22[2][2][2] = {0.0};
    double cube_e23[2][2][2] = {0.0};
    double cube_e33[2][2][2] = {0.0};

    // now set the cube values
    for(int kkk = 0; kkk <= kp; kkk++)
    {
        for(int jjj = 0; jjj <= jp; jjj++)
        {
            for(int iii = 0; iii <= ip; iii++)
            {
                // set the actual indices to use for the linearized Euler data
                int idx = (kk+kkk)*ny*nx + (jj+jjj)*nx + (ii+iii);
                cube_e11[iii][jjj][kkk] = EulerData.at(idx).e11;
                cube_e12[iii][jjj][kkk] = EulerData.at(idx).e12;
                cube_e13[iii][jjj][kkk] = EulerData.at(idx).e13;
                cube_e22[iii][jjj][kkk] = EulerData.at(idx).e22;
                cube_e23[iii][jjj][kkk] = EulerData.at(idx).e23;
                cube_e33[iii][jjj][kkk] = EulerData.at(idx).e33;
            }
        }
    }

    // now do the interpolation, with the cube, the counters from the indices,
    // and the normalized width between the point locations and the closest cell left walls
    double u_low  = (1-iw)*(1-jw)*cube_e11[0][0][0] + iw*(1-jw)*cube_e11[1][0][0] + iw*jw*cube_e11[1][1][0] + (1-iw)*jw*cube_e11[0][1][0];
    double u_high = (1-iw)*(1-jw)*cube_e11[0][0][1] + iw*(1-jw)*cube_e11[1][0][1] + iw*jw*cube_e11[1][1][1] + (1-iw)*jw*cube_e11[0][1][1];
    outputVal.e11 = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_e12[0][0][0] + iw*(1-jw)*cube_e12[1][0][0] + iw*jw*cube_e12[1][1][0] + (1-iw)*jw*cube_e12[0][1][0];
    u_high        = (1-iw)*(1-jw)*cube_e12[0][0][1] + iw*(1-jw)*cube_e12[1][0][1] + iw*jw*cube_e12[1][1][1] + (1-iw)*jw*cube_e12[0][1][1];
    outputVal.e12 = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_e13[0][0][0] + iw*(1-jw)*cube_e13[1][0][0] + iw*jw*cube_e13[1][1][0] + (1-iw)*jw*cube_e13[0][1][0];
    u_high        = (1-iw)*(1-jw)*cube_e13[0][0][1] + iw*(1-jw)*cube_e13[1][0][1] + iw*jw*cube_e13[1][1][1] + (1-iw)*jw*cube_e13[0][1][1];
    outputVal.e13 = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_e22[0][0][0] + iw*(1-jw)*cube_e22[1][0][0] + iw*jw*cube_e22[1][1][0] + (1-iw)*jw*cube_e22[0][1][0];
    u_high        = (1-iw)*(1-jw)*cube_e22[0][0][1] + iw*(1-jw)*cube_e22[1][0][1] + iw*jw*cube_e22[1][1][1] + (1-iw)*jw*cube_e22[0][1][1];
    outputVal.e22 = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_e23[0][0][0] + iw*(1-jw)*cube_e23[1][0][0] + iw*jw*cube_e23[1][1][0] + (1-iw)*jw*cube_e23[0][1][0];
    u_high        = (1-iw)*(1-jw)*cube_e23[0][0][1] + iw*(1-jw)*cube_e23[1][0][1] + iw*jw*cube_e23[1][1][1] + (1-iw)*jw*cube_e23[0][1][1];
    outputVal.e23 = (u_high-u_low)*kw + u_low;
    
    u_low         = (1-iw)*(1-jw)*cube_e33[0][0][0] + iw*(1-jw)*cube_e33[1][0][0] + iw*jw*cube_e33[1][1][0] + (1-iw)*jw*cube_e33[0][1][0];
    u_high        = (1-iw)*(1-jw)*cube_e33[0][0][1] + iw*(1-jw)*cube_e33[1][0][1] + iw*jw*cube_e33[1][1][1] + (1-iw)*jw*cube_e33[0][1][1];
    outputVal.e33 = (u_high-u_low)*kw + u_low;

    return outputVal;

}

// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
Wind Eulerian::interp3D(const std::vector<Wind>& EulerData)
{

    // initialize the output value
    Wind outputVal;
    outputVal.u = 0.0;
    outputVal.v = 0.0;
    outputVal.w = 0.0;
    

    // first set a cube of size two to zero.
    // This is important because if nx, ny, or nz are only size 1, referencing two spots in cube won't reference outside the array.
    // the next steps are to figure out the right indices to grab the values for cube from the data, 
    // where indices are forced to be special if nx, ny, or nz are zero.
    // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

    
    // now set the cube to zero, then fill it using the indices and the counters from the indices
    double cube_u[2][2][2] = {0.0};
    double cube_v[2][2][2] = {0.0};
    double cube_w[2][2][2] = {0.0};

    // now set the cube values
    for(int kkk = 0; kkk <= kp; kkk++)
    {
        for(int jjj = 0; jjj <= jp; jjj++)
        {
            for(int iii = 0; iii <= ip; iii++)
            {
                // set the actual indices to use for the linearized Euler data
                int idx = (kk+kkk)*ny*nx + (jj+jjj)*nx + (ii+iii);
                cube_u[iii][jjj][kkk] = EulerData.at(idx).u;
                cube_v[iii][jjj][kkk] = EulerData.at(idx).v;
                cube_w[iii][jjj][kkk] = EulerData.at(idx).w;
            }
        }
    }

    // now do the interpolation, with the cube, the counters from the indices,
    // and the normalized width between the point locations and the closest cell left walls
    double u_low  = (1-iw)*(1-jw)*cube_u[0][0][0] + iw*(1-jw)*cube_u[1][0][0] + iw*jw*cube_u[1][1][0] + (1-iw)*jw*cube_u[0][1][0];
    double u_high = (1-iw)*(1-jw)*cube_u[0][0][1] + iw*(1-jw)*cube_u[1][0][1] + iw*jw*cube_u[1][1][1] + (1-iw)*jw*cube_u[0][1][1];
    outputVal.u = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_v[0][0][0] + iw*(1-jw)*cube_v[1][0][0] + iw*jw*cube_v[1][1][0] + (1-iw)*jw*cube_v[0][1][0];
    u_high        = (1-iw)*(1-jw)*cube_v[0][0][1] + iw*(1-jw)*cube_v[1][0][1] + iw*jw*cube_v[1][1][1] + (1-iw)*jw*cube_v[0][1][1];
    outputVal.v = (u_high-u_low)*kw + u_low;
    
    u_low         = (1-iw)*(1-jw)*cube_w[0][0][0] + iw*(1-jw)*cube_w[1][0][0] + iw*jw*cube_w[1][1][0] + (1-iw)*jw*cube_w[0][1][0];
    u_high        = (1-iw)*(1-jw)*cube_w[0][0][1] + iw*(1-jw)*cube_w[1][0][1] + iw*jw*cube_w[1][1][1] + (1-iw)*jw*cube_w[0][1][1];
    outputVal.w = (u_high-u_low)*kw + u_low;

    return outputVal;

}

void Eulerian::outputVarInfo_text(Urb* urb, Turb* turb, const std::string& outputFolder)
{
    // if the debug output folder is an empty string "", the debug output variables won't be written
    if( outputFolder == "" )
    {
        return;
    }

    std::cout << "writing Eulerian debug variables" << std::endl;


    // set some variables for use in the function
    FILE *fzout;    // changing file to which information will be written
    std::string currentFile = "";


    // now write out the Eulerian grid information to the debug folder
    // at some time this could be wrapped up into a bunch of functions, for now just type it all out without functions



    currentFile = outputFolder + "/urb_xCellGrid.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nx; idx++)
    {
        fprintf(fzout,"%lf\n",urb->grid.x.at(idx));
    }
    fclose(fzout);

    currentFile = outputFolder + "/urb_yCellGrid.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < ny; idx++)
    {
        fprintf(fzout,"%lf\n",urb->grid.y.at(idx));
    }
    fclose(fzout);

    currentFile = outputFolder + "/urb_zCellGrid.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int idx = 0; idx < nz; idx++)
    {
        fprintf(fzout,"%lf\n",urb->grid.z.at(idx));
    }
    fclose(fzout);



    currentFile = outputFolder + "/eulerian_uMean.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",urb->wind.at(kk*nx*ny + jj*nx + ii).u);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_vMean.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",urb->wind.at(kk*nx*ny + jj*nx + ii).v);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_wMean.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",urb->wind.at(kk*nx*ny + jj*nx + ii).w);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_sigma2.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",turb->sig.at(kk*nx*ny + jj*nx + ii).e33);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_epps.txt";
    double C_0 = 4.0;
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",turb->CoEps.at(kk*nx*ny + jj*nx + ii)/C_0);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_txx.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",turb->tau.at(kk*nx*ny + jj*nx + ii).e11);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_txy.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",turb->tau.at(kk*nx*ny + jj*nx + ii).e12);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_txz.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",turb->tau.at(kk*nx*ny + jj*nx + ii).e13);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_tyy.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",turb->tau.at(kk*nx*ny + jj*nx + ii).e22);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_tyz.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",turb->tau.at(kk*nx*ny + jj*nx + ii).e23);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_tzz.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",turb->tau.at(kk*nx*ny + jj*nx + ii).e33);
            }
        }
    }
    fclose(fzout);



    currentFile = outputFolder + "/eulerian_dtxxdx.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",taudx.at(kk*nx*ny + jj*nx + ii).e11);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_dtxydx.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",taudx.at(kk*nx*ny + jj*nx + ii).e12);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_dtxzdx.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",taudx.at(kk*nx*ny + jj*nx + ii).e13);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_dtxydy.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",taudy.at(kk*nx*ny + jj*nx + ii).e12);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_dtyydy.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",taudy.at(kk*nx*ny + jj*nx + ii).e22);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_dtyzdy.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",taudy.at(kk*nx*ny + jj*nx + ii).e23);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_dtxzdz.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",taudz.at(kk*nx*ny + jj*nx + ii).e13);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_dtyzdz.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",taudz.at(kk*nx*ny + jj*nx + ii).e23);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_dtzzdz.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",taudz.at(kk*nx*ny + jj*nx + ii).e33);
            }
        }
    }
    fclose(fzout);



    currentFile = outputFolder + "/eulerian_flux_div_x.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",flux_div.at(kk*nx*ny + jj*nx + ii).e11);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_flux_div_y.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",flux_div.at(kk*nx*ny + jj*nx + ii).e21);
            }
        }
    }
    fclose(fzout);

    currentFile = outputFolder + "/eulerian_flux_div_z.txt";
    fzout = fopen(currentFile.c_str(), "w");
    for(int kk = 0; kk < nz; kk++)
    {
        for(int jj = 0; jj < ny; jj++)
        {
            for(int ii = 0; ii < nx; ii++)
            {
                fprintf(fzout,"%lf\n",flux_div.at(kk*nx*ny + jj*nx + ii).e31);
            }
        }
    }
    fclose(fzout);

    // now that all is finished, clean up the file pointer
    fzout = NULL;

}




void Eulerian::display(const matrix9& mat){
    std::cout<<std::endl;
    std::cout<<mat.e11<<"  "<<mat.e12<<"  "<<mat.e13<<"  "<<std::endl;
    std::cout<<mat.e21<<"  "<<mat.e22<<"  "<<mat.e23<<"  "<<std::endl;
    std::cout<<mat.e31<<"  "<<mat.e32<<"  "<<mat.e33<<"  "<<std::endl;
}

void Eulerian::display(const matrix6& mat){
    std::cout<<std::endl;
    std::cout<<mat.e11<<"  "<<mat.e12<<"  "<<mat.e13<<"  "<<std::endl;
    std::cout<<mat.e12<<"  "<<mat.e22<<"  "<<mat.e23<<"  "<<std::endl;
    std::cout<<mat.e13<<"  "<<mat.e23<<"  "<<mat.e33<<"  "<<std::endl;
}

void Eulerian::display(const vec3& vec){
    std::cout<<std::endl;
    std::cout<<vec.e11<<std::endl;
    std::cout<<vec.e21<<std::endl;
    std::cout<<vec.e31<<std::endl;
}

void Eulerian::display(const diagonal& mat){
    std::cout<<std::endl;
    std::cout<<mat.e11<<"  "<<"0"<<"  "<<"0"<<"  "<<std::endl;
    std::cout<<"0"<<"  "<<mat.e22<<"  "<<"0"<<"  "<<std::endl;
    std::cout<<"0"<<"  "<<"0"<<"  "<<mat.e33<<"  "<<std::endl;
}
