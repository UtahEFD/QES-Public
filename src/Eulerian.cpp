#include <iostream>
#include <ctime>
#include <cmath>

#include <chrono>

#include "Eulerian.h"
#include "Random.h"
#include "Urb.hpp"
#include "Turb.hpp"

Eulerian::Eulerian(Urb* urb, Turb* turb) {
    
    std::cout<<"[Eulerian] \t Reading CUDA-URB & CUDA-TURB fields "<<std::endl;
    
    // define local variables
    zo     = 0.10;
    vonKar = 0.41;
    
    // grid information
    nt = urb->grid.nt;
    nz = urb->grid.nz;
    ny = urb->grid.ny;
    nx = urb->grid.nx;
    dz = urb->grid.dz;
    dy = urb->grid.dy;
    dx = urb->grid.dx;
    
    // compute stress gradients
    createTauGrads(urb,turb);

    // use the stress gradients to calculate the flux div
    createFluxDiv();
    
    // construct matrices
    createA1Matrix(urb,turb);
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
                if (i < (nx-2)) {
                    setDX_Forward(turb, idx);
                }
                else { 
                    setDX_Backward(turb, idx);
                }
                    
                // DY components
                if (j < (ny-2)) {
                    setDY_Forward(turb, idx);
                }
                else { 
                    setDY_Backward(turb, idx);
                }

                // DZ components
                if (k < (nz-2)) {
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
    std::cout << "\telapsed time: " << elapsed.count() << " s\n";   // Print out elapsed execution time
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
    std::cout << "\telapsed time: " << elapsed.count() << " s\n";   // Print out elapsed execution time
}

#endif

void Eulerian::createFluxDiv()
{
    // loop through each cell and calculate the flux_div_vel from the gradients of tao
    for(int idx = 1; idx < nx*ny*nz; idx++) {
        flux_div.at(idx).e11 = taudx.at(idx).e11 + taudy.at(idx).e12 + taudz.at(idx).e13;
        flux_div.at(idx).e12 = taudx.at(idx).e12 + taudy.at(idx).e22 + taudz.at(idx).e23;
        flux_div.at(idx).e13 = taudx.at(idx).e13 + taudy.at(idx).e23 + taudz.at(idx).e33;
    }
}

// this gets around the problem of repeated or not repeated information, just needs called once before each interpolation,
// then intepolation on all kinds of datatypes can be done
void Eulerian::setInterp3Dindexing(const float3& xyz_particle)
{
    // the next steps are to figure out the right indices to grab the values for cube from the data, 
    // where indices are forced to be special if nx, ny, or nz are zero.
    // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

    // so this is called once before calling the interp3D function on many different datatypes
    // sets the current indices for grabbing the cube values and for interpolating with the cube,
    // but importantly sets ip,jp, and kp to zero if the number of cells in a dimension is 1
    // as this avoids referencing outside of array problems in an efficient manner
    // it also causes some stuff to be multiplied by zero so that interpolation works on any size of data without lots of if statements

    // index of nearest node in negative direction
    ii = floor(xyz_particle(1)/dx)+1;
    jj = floor(xyz_particle(2)/dy)+1;
    kk = floor(xyz_particle(3)/dz)+1;

    // fractional distance between nearest nodes
    iw = (xyz_particle(1)/dx-floor(xyz_particle(1)/dx));
    jw = (xyz_particle(2)/dy-floor(xyz_particle(2)/dy));
    kw = (xyz_particle(3)/dz-floor(xyz_particle(3)/dz));

    // initialize the counters from the indices
    ip = 1;
    jp = 1;
    kp = 1;

    // now set the indices and the counters from the indices
    if( nx == 1 )
    {
        ii = 1;
        iw = 0.0;
        ip = 0;
    }
    if( ny == 1 )
    {
        jj = 1;
        jw = 0.0;
        jp = 0;
    }
    if( nz == 1 )
    {
        kk = 1;
        kw = 0.0;
        kp = 0;
    }

}

// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
double Eulerian::interp3D(const std::vector<double>& EulerData)
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
    for(int kkk = 0; kk <= kk+kp; kkk++)
    {
        for(int jjj = 0; jj <= jj+jp; jjj++)
        {
            for(int iii = 0; ii <= ii+ip; iii++)
            {
                // set the actual indices to use for the linearized Euler data
                int idx = kkk*ny*nx + jjj*nx + iii;
                cube(iii,jjj,kkk) = EulerData(idx);
            }
        }
    }

    // now do the interpolation, with the cube, the counters from the indices,
    // and the normalized width between the point locations and the closest cell left walls
    double u_low  = (1-iw)*(1-jw)*cube(1,1,1) + iw*(1-jw)*cube(2,1,1) + iw*jw*cube(2,2,1) + (1-iw)*jw*cube(1,2,1);
    double u_high = (1-iw)*(1-jw)*cube(1,1,2) + iw*(1-jw)*cube(2,1,2) + iw*jw*cube(2,2,2) + (1-iw)*jw*cube(1,2,2);
    outputVal = (u_high-u_low)*kw + u_low;

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
    for(int kkk = 0; kk <= kk+kp; kkk++)
    {
        for(int jjj = 0; jj <= jj+jp; jjj++)
        {
            for(int iii = 0; ii <= ii+ip; iii++)
            {
                // set the actual indices to use for the linearized Euler data
                int idx = kkk*ny*nx + jjj*nx + iii;
                cube_e11(iii,jjj,kkk) = EulerData.at(idx).e11;
                cube_e21(iii,jjj,kkk) = EulerData.at(idx).e21;
                cube_e31(iii,jjj,kkk) = EulerData.at(idx).e31;
            }
        }
    }

    // now do the interpolation, with the cube, the counters from the indices,
    // and the normalized width between the point locations and the closest cell left walls
    double u_low  = (1-iw)*(1-jw)*cube_e11(1,1,1) + iw*(1-jw)*cube_e11(2,1,1) + iw*jw*cube_e11(2,2,1) + (1-iw)*jw*cube_e11(1,2,1);
    double u_high = (1-iw)*(1-jw)*cube_e11(1,1,2) + iw*(1-jw)*cube_e11(2,1,2) + iw*jw*cube_e11(2,2,2) + (1-iw)*jw*cube_e11(1,2,2);
    outputVal.e11 = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_e21(1,1,1) + iw*(1-jw)*cube_e21(2,1,1) + iw*jw*cube_e21(2,2,1) + (1-iw)*jw*cube_e21(1,2,1);
    u_high        = (1-iw)*(1-jw)*cube_e21(1,1,2) + iw*(1-jw)*cube_e21(2,1,2) + iw*jw*cube_e21(2,2,2) + (1-iw)*jw*cube_e21(1,2,2);
    outputVal.e21 = (u_high-u_low)*kw + u_low;
    
    u_low         = (1-iw)*(1-jw)*cube_e31(1,1,1) + iw*(1-jw)*cube_e31(2,1,1) + iw*jw*cube_e31(2,2,1) + (1-iw)*jw*cube_e31(1,2,1);
    u_high        = (1-iw)*(1-jw)*cube_e31(1,1,2) + iw*(1-jw)*cube_e31(2,1,2) + iw*jw*cube_e31(2,2,2) + (1-iw)*jw*cube_e31(1,2,2);
    outputVal.e31 = (u_high-u_low)*kw + u_low;

    return outputVal;

}

// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
diagonal Eulerian::interp3D(const std::vector<diagonal>& EulerData)
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
    for(int kkk = 0; kk <= kk+kp; kkk++)
    {
        for(int jjj = 0; jj <= jj+jp; jjj++)
        {
            for(int iii = 0; ii <= ii+ip; iii++)
            {
                // set the actual indices to use for the linearized Euler data
                int idx = kkk*ny*nx + jjj*nx + iii;
                cube_e11(iii,jjj,kkk) = EulerData.at(idx).e11;
                cube_e22(iii,jjj,kkk) = EulerData.at(idx).e22;
                cube_e33(iii,jjj,kkk) = EulerData.at(idx).e33;
            }
        }
    }

    // now do the interpolation, with the cube, the counters from the indices,
    // and the normalized width between the point locations and the closest cell left walls
    double u_low  = (1-iw)*(1-jw)*cube_e11(1,1,1) + iw*(1-jw)*cube_e11(2,1,1) + iw*jw*cube_e11(2,2,1) + (1-iw)*jw*cube_e11(1,2,1);
    double u_high = (1-iw)*(1-jw)*cube_e11(1,1,2) + iw*(1-jw)*cube_e11(2,1,2) + iw*jw*cube_e11(2,2,2) + (1-iw)*jw*cube_e11(1,2,2);
    outputVal.e11 = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_e22(1,1,1) + iw*(1-jw)*cube_e22(2,1,1) + iw*jw*cube_e22(2,2,1) + (1-iw)*jw*cube_e22(1,2,1);
    u_high        = (1-iw)*(1-jw)*cube_e22(1,1,2) + iw*(1-jw)*cube_e22(2,1,2) + iw*jw*cube_e22(2,2,2) + (1-iw)*jw*cube_e22(1,2,2);
    outputVal.e22 = (u_high-u_low)*kw + u_low;
    
    u_low         = (1-iw)*(1-jw)*cube_e33(1,1,1) + iw*(1-jw)*cube_e33(2,1,1) + iw*jw*cube_e33(2,2,1) + (1-iw)*jw*cube_e33(1,2,1);
    u_high        = (1-iw)*(1-jw)*cube_e33(1,1,2) + iw*(1-jw)*cube_e33(2,1,2) + iw*jw*cube_e33(2,2,2) + (1-iw)*jw*cube_e33(1,2,2);
    outputVal.e33 = (u_high-u_low)*kw + u_low;

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
    for(int kkk = 0; kk <= kk+kp; kkk++)
    {
        for(int jjj = 0; jj <= jj+jp; jjj++)
        {
            for(int iii = 0; ii <= ii+ip; iii++)
            {
                // set the actual indices to use for the linearized Euler data
                int idx = kkk*ny*nx + jjj*nx + iii;
                cube_e11(iii,jjj,kkk) = EulerData.at(idx).e11;
                cube_e12(iii,jjj,kkk) = EulerData.at(idx).e12;
                cube_e13(iii,jjj,kkk) = EulerData.at(idx).e13;
                cube_e22(iii,jjj,kkk) = EulerData.at(idx).e22;
                cube_e23(iii,jjj,kkk) = EulerData.at(idx).e23;
                cube_e33(iii,jjj,kkk) = EulerData.at(idx).e33;
            }
        }
    }

    // now do the interpolation, with the cube, the counters from the indices,
    // and the normalized width between the point locations and the closest cell left walls
    double u_low  = (1-iw)*(1-jw)*cube_e11(1,1,1) + iw*(1-jw)*cube_e11(2,1,1) + iw*jw*cube_e11(2,2,1) + (1-iw)*jw*cube_e11(1,2,1);
    double u_high = (1-iw)*(1-jw)*cube_e11(1,1,2) + iw*(1-jw)*cube_e11(2,1,2) + iw*jw*cube_e11(2,2,2) + (1-iw)*jw*cube_e11(1,2,2);
    outputVal.e11 = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_e12(1,1,1) + iw*(1-jw)*cube_e12(2,1,1) + iw*jw*cube_e12(2,2,1) + (1-iw)*jw*cube_e12(1,2,1);
    u_high        = (1-iw)*(1-jw)*cube_e12(1,1,2) + iw*(1-jw)*cube_e12(2,1,2) + iw*jw*cube_e12(2,2,2) + (1-iw)*jw*cube_e12(1,2,2);
    outputVal.e12 = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_e13(1,1,1) + iw*(1-jw)*cube_e13(2,1,1) + iw*jw*cube_e13(2,2,1) + (1-iw)*jw*cube_e13(1,2,1);
    u_high        = (1-iw)*(1-jw)*cube_e13(1,1,2) + iw*(1-jw)*cube_e13(2,1,2) + iw*jw*cube_e13(2,2,2) + (1-iw)*jw*cube_e13(1,2,2);
    outputVal.e13 = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_e22(1,1,1) + iw*(1-jw)*cube_e22(2,1,1) + iw*jw*cube_e22(2,2,1) + (1-iw)*jw*cube_e22(1,2,1);
    u_high        = (1-iw)*(1-jw)*cube_e22(1,1,2) + iw*(1-jw)*cube_e22(2,1,2) + iw*jw*cube_e22(2,2,2) + (1-iw)*jw*cube_e22(1,2,2);
    outputVal.e22 = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_e23(1,1,1) + iw*(1-jw)*cube_e23(2,1,1) + iw*jw*cube_e23(2,2,1) + (1-iw)*jw*cube_e23(1,2,1);
    u_high        = (1-iw)*(1-jw)*cube_e23(1,1,2) + iw*(1-jw)*cube_e23(2,1,2) + iw*jw*cube_e23(2,2,2) + (1-iw)*jw*cube_e23(1,2,2);
    outputVal.e23 = (u_high-u_low)*kw + u_low;
    
    u_low         = (1-iw)*(1-jw)*cube_e33(1,1,1) + iw*(1-jw)*cube_e33(2,1,1) + iw*jw*cube_e33(2,2,1) + (1-iw)*jw*cube_e33(1,2,1);
    u_high        = (1-iw)*(1-jw)*cube_e33(1,1,2) + iw*(1-jw)*cube_e33(2,1,2) + iw*jw*cube_e33(2,2,2) + (1-iw)*jw*cube_e33(1,2,2);
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
    for(int kkk = 0; kk <= kk+kp; kkk++)
    {
        for(int jjj = 0; jj <= jj+jp; jjj++)
        {
            for(int iii = 0; ii <= ii+ip; iii++)
            {
                // set the actual indices to use for the linearized Euler data
                int idx = kkk*ny*nx + jjj*nx + iii;
                cube_u(iii,jjj,kkk) = EulerData.at(idx).u;
                cube_v(iii,jjj,kkk) = EulerData.at(idx).v;
                cube_w(iii,jjj,kkk) = EulerData.at(idx).w;
            }
        }
    }

    // now do the interpolation, with the cube, the counters from the indices,
    // and the normalized width between the point locations and the closest cell left walls
    double u_low  = (1-iw)*(1-jw)*cube_u(1,1,1) + iw*(1-jw)*cube_u(2,1,1) + iw*jw*cube_u(2,2,1) + (1-iw)*jw*cube_u(1,2,1);
    double u_high = (1-iw)*(1-jw)*cube_u(1,1,2) + iw*(1-jw)*cube_u(2,1,2) + iw*jw*cube_u(2,2,2) + (1-iw)*jw*cube_u(1,2,2);
    outputVal.u = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_v(1,1,1) + iw*(1-jw)*cube_v(2,1,1) + iw*jw*cube_v(2,2,1) + (1-iw)*jw*cube_v(1,2,1);
    u_high        = (1-iw)*(1-jw)*cube_v(1,1,2) + iw*(1-jw)*cube_v(2,1,2) + iw*jw*cube_v(2,2,2) + (1-iw)*jw*cube_v(1,2,2);
    outputVal.v = (u_high-u_low)*kw + u_low;
    
    u_low         = (1-iw)*(1-jw)*cube_w(1,1,1) + iw*(1-jw)*cube_w(2,1,1) + iw*jw*cube_w(2,2,1) + (1-iw)*jw*cube_w(1,2,1);
    u_high        = (1-iw)*(1-jw)*cube_w(1,1,2) + iw*(1-jw)*cube_w(2,1,2) + iw*jw*cube_w(2,2,2) + (1-iw)*jw*cube_w(1,2,2);
    outputVal.w = (u_high-u_low)*kw + u_low;

    return outputVal;

}

// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
float3 Eulerian::interp3D(const std::vector<float3>& EulerData)
{

    // initialize the output value
    float3 outputVal;
    outputVal.x = 0.0;
    outputVal.y = 0.0;
    outputVal.z = 0.0;
    

    // first set a cube of size two to zero.
    // This is important because if nx, ny, or nz are only size 1, referencing two spots in cube won't reference outside the array.
    // the next steps are to figure out the right indices to grab the values for cube from the data, 
    // where indices are forced to be special if nx, ny, or nz are zero.
    // This allows the interpolation to multiply by zero any 2nd values that are set to zero in cube.

    
    // now set the cube to zero, then fill it using the indices and the counters from the indices
    double cube_x[2][2][2] = {0.0};
    double cube_y[2][2][2] = {0.0};
    double cube_z[2][2][2] = {0.0};

    // now set the cube values
    for(int kkk = 0; kk <= kk+kp; kkk++)
    {
        for(int jjj = 0; jj <= jj+jp; jjj++)
        {
            for(int iii = 0; ii <= ii+ip; iii++)
            {
                // set the actual indices to use for the linearized Euler data
                int idx = kkk*ny*nx + jjj*nx + iii;
                cube_x(iii,jjj,kkk) = EulerData.at(idx).x;
                cube_y(iii,jjj,kkk) = EulerData.at(idx).y;
                cube_z(iii,jjj,kkk) = EulerData.at(idx).z;
            }
        }
    }

    // now do the interpolation, with the cube, the counters from the indices,
    // and the normalized width between the point locations and the closest cell left walls
    double u_low  = (1-iw)*(1-jw)*cube_x(1,1,1) + iw*(1-jw)*cube_x(2,1,1) + iw*jw*cube_x(2,2,1) + (1-iw)*jw*cube_x(1,2,1);
    double u_high = (1-iw)*(1-jw)*cube_x(1,1,2) + iw*(1-jw)*cube_x(2,1,2) + iw*jw*cube_x(2,2,2) + (1-iw)*jw*cube_x(1,2,2);
    outputVal.x = (u_high-u_low)*kw + u_low;

    u_low         = (1-iw)*(1-jw)*cube_y(1,1,1) + iw*(1-jw)*cube_y(2,1,1) + iw*jw*cube_y(2,2,1) + (1-iw)*jw*cube_y(1,2,1);
    u_high        = (1-iw)*(1-jw)*cube_y(1,1,2) + iw*(1-jw)*cube_y(2,1,2) + iw*jw*cube_y(2,2,2) + (1-iw)*jw*cube_y(1,2,2);
    outputVal.y = (u_high-u_low)*kw + u_low;
    
    u_low         = (1-iw)*(1-jw)*cube_z(1,1,1) + iw*(1-jw)*cube_z(2,1,1) + iw*jw*cube_z(2,2,1) + (1-iw)*jw*cube_z(1,2,1);
    u_high        = (1-iw)*(1-jw)*cube_z(1,1,2) + iw*(1-jw)*cube_z(2,1,2) + iw*jw*cube_z(2,2,2) + (1-iw)*jw*cube_z(1,2,2);
    outputVal.z = (u_high-u_low)*kw + u_low;

    return outputVal;

}



void Eulerian::createA1Matrix(Urb* urb, Turb* turb) {
    
    std::cout<<"[Eulerian] \t Creating matrices"<<std::endl;
    
    eigVal.resize(nx*ny*nz);
    eigVec.resize(nx*ny*nz);
    eigVecInv.resize(nx*ny*nz);
    ka0.resize(nx*ny*nz);
    g2nd.resize(nx*ny*nz);
  
    double cond_A1=0.0;
    double det_A1=0.0;
    int number=0;
    int flagnum=0;
    
    int id;
    for(int k=1; k<nz-1; k++) { 
        for(int j=1; j<ny-1; j++) {
            for(int i=1; i<nx-1; i++) {
                
                id = k*ny*nx+j*nx+i;
                
                int idxp1=id+1;
                int idxm1=id-1;
                int idxp2=id+2;
                int idxm2=id-2;
                int idyp1=id+nx;
                int idym1=id-nx;
                int idzp1=id+(ny*nx);
                int idzm1=id-(ny*nx);
    
                cond_A1=0.0;
                det_A1=0.0;
                
                if(urb->grid.icell[id]!=0) {
                    
                    double A1_1e11 = -0.5*turb->CoEps.at(id)*turb->lam.at(id).e11;
                    double A1_1e12 = -0.5*turb->CoEps.at(id)*turb->lam.at(id).e12;
                    double A1_1e13 = -0.5*turb->CoEps.at(id)*turb->lam.at(id).e13;
                    double A1_1e21 = -0.5*turb->CoEps.at(id)*turb->lam.at(id).e21;
                    double A1_1e22 = -0.5*turb->CoEps.at(id)*turb->lam.at(id).e22;
                    double A1_1e23 = -0.5*turb->CoEps.at(id)*turb->lam.at(id).e23;
                    double A1_1e31 = -0.5*turb->CoEps.at(id)*turb->lam.at(id).e31;
                    double A1_1e32 = -0.5*turb->CoEps.at(id)*turb->lam.at(id).e32;
                    double A1_1e33 = -0.5*turb->CoEps.at(id)*turb->lam.at(id).e33;
                    
                    double A1_2e11 = 0.5*turb->lam.at(id).e11*taudx.at(id).e11*urb->wind.at(id).u;
                    double A1_2e12 = 0.5*turb->lam.at(id).e11*taudx.at(id).e12*urb->wind.at(id).u;
                    double A1_2e13 = 0.5*turb->lam.at(id).e11*taudx.at(id).e13*urb->wind.at(id).u;
                    double A1_2e21 = 0.5*turb->lam.at(id).e12*taudx.at(id).e11*urb->wind.at(id).u;
                    double A1_2e22 = 0.5*turb->lam.at(id).e12*taudx.at(id).e12*urb->wind.at(id).u;
                    double A1_2e23 = 0.5*turb->lam.at(id).e12*taudx.at(id).e13*urb->wind.at(id).u;
                    double A1_2e31 = 0.5*turb->lam.at(id).e13*taudx.at(id).e11*urb->wind.at(id).u;
                    double A1_2e32 = 0.5*turb->lam.at(id).e13*taudx.at(id).e12*urb->wind.at(id).u;
                    double A1_2e33 = 0.5*turb->lam.at(id).e13*taudx.at(id).e13*urb->wind.at(id).u;
                    
                    double A1_3e11 = 0.5*turb->lam.at(id).e21*taudx.at(id).e12*urb->wind.at(id).u;
                    double A1_3e12 = 0.5*turb->lam.at(id).e21*taudx.at(id).e22*urb->wind.at(id).u;
                    double A1_3e13 = 0.5*turb->lam.at(id).e21*taudx.at(id).e23*urb->wind.at(id).u;
                    double A1_3e21 = 0.5*turb->lam.at(id).e22*taudx.at(id).e12*urb->wind.at(id).u;
                    double A1_3e22 = 0.5*turb->lam.at(id).e22*taudx.at(id).e22*urb->wind.at(id).u;
                    double A1_3e23 = 0.5*turb->lam.at(id).e22*taudx.at(id).e23*urb->wind.at(id).u;
                    double A1_3e31 = 0.5*turb->lam.at(id).e23*taudx.at(id).e12*urb->wind.at(id).u;
                    double A1_3e32 = 0.5*turb->lam.at(id).e23*taudx.at(id).e22*urb->wind.at(id).u;
                    double A1_3e33 = 0.5*turb->lam.at(id).e23*taudx.at(id).e23*urb->wind.at(id).u;
                    
                    double A1_4e11 = 0.5*turb->lam.at(id).e31*taudx.at(id).e13*urb->wind.at(id).u;
                    double A1_4e12 = 0.5*turb->lam.at(id).e31*taudx.at(id).e23*urb->wind.at(id).u;
                    double A1_4e13 = 0.5*turb->lam.at(id).e31*taudx.at(id).e33*urb->wind.at(id).u;
                    double A1_4e21 = 0.5*turb->lam.at(id).e32*taudx.at(id).e13*urb->wind.at(id).u;
                    double A1_4e22 = 0.5*turb->lam.at(id).e32*taudx.at(id).e23*urb->wind.at(id).u;
                    double A1_4e23 = 0.5*turb->lam.at(id).e32*taudx.at(id).e33*urb->wind.at(id).u;
                    double A1_4e31 = 0.5*turb->lam.at(id).e33*taudx.at(id).e13*urb->wind.at(id).u;
                    double A1_4e32 = 0.5*turb->lam.at(id).e33*taudx.at(id).e23*urb->wind.at(id).u;
                    double A1_4e33 = 0.5*turb->lam.at(id).e33*taudx.at(id).e33*urb->wind.at(id).u;
                    
                    double A1_5e11 = 0.5*turb->lam.at(id).e11*taudy.at(id).e11*urb->wind.at(id).v;
                    double A1_5e12 = 0.5*turb->lam.at(id).e11*taudy.at(id).e12*urb->wind.at(id).v;
                    double A1_5e13 = 0.5*turb->lam.at(id).e11*taudy.at(id).e13*urb->wind.at(id).v;
                    double A1_5e21 = 0.5*turb->lam.at(id).e12*taudy.at(id).e11*urb->wind.at(id).v;
                    double A1_5e22 = 0.5*turb->lam.at(id).e12*taudy.at(id).e12*urb->wind.at(id).v;
                    double A1_5e23 = 0.5*turb->lam.at(id).e12*taudy.at(id).e13*urb->wind.at(id).v;
                    double A1_5e31 = 0.5*turb->lam.at(id).e13*taudy.at(id).e11*urb->wind.at(id).v;
                    double A1_5e32 = 0.5*turb->lam.at(id).e13*taudy.at(id).e12*urb->wind.at(id).v;
                    double A1_5e33 = 0.5*turb->lam.at(id).e13*taudy.at(id).e13*urb->wind.at(id).v;
                    
                    double A1_6e11 = 0.5*turb->lam.at(id).e21*taudy.at(id).e12*urb->wind.at(id).v;
                    double A1_6e12 = 0.5*turb->lam.at(id).e21*taudy.at(id).e22*urb->wind.at(id).v;
                    double A1_6e13 = 0.5*turb->lam.at(id).e21*taudy.at(id).e23*urb->wind.at(id).v;
                    double A1_6e21 = 0.5*turb->lam.at(id).e22*taudy.at(id).e12*urb->wind.at(id).v;
                    double A1_6e22 = 0.5*turb->lam.at(id).e22*taudy.at(id).e22*urb->wind.at(id).v;
                    double A1_6e23 = 0.5*turb->lam.at(id).e22*taudy.at(id).e23*urb->wind.at(id).v;
                    double A1_6e31 = 0.5*turb->lam.at(id).e23*taudy.at(id).e12*urb->wind.at(id).v;
                    double A1_6e32 = 0.5*turb->lam.at(id).e23*taudy.at(id).e22*urb->wind.at(id).v;
                    double A1_6e33 = 0.5*turb->lam.at(id).e23*taudy.at(id).e23*urb->wind.at(id).v;
                    
                    double A1_7e11 = 0.5*turb->lam.at(id).e31*taudy.at(id).e13*urb->wind.at(id).v;
                    double A1_7e12 = 0.5*turb->lam.at(id).e31*taudy.at(id).e23*urb->wind.at(id).v;
                    double A1_7e13 = 0.5*turb->lam.at(id).e31*taudy.at(id).e33*urb->wind.at(id).v;
                    double A1_7e21 = 0.5*turb->lam.at(id).e32*taudy.at(id).e13*urb->wind.at(id).v;
                    double A1_7e22 = 0.5*turb->lam.at(id).e32*taudy.at(id).e23*urb->wind.at(id).v;
                    double A1_7e23 = 0.5*turb->lam.at(id).e32*taudy.at(id).e33*urb->wind.at(id).v;
                    double A1_7e31 = 0.5*turb->lam.at(id).e33*taudy.at(id).e13*urb->wind.at(id).v;
                    double A1_7e32 = 0.5*turb->lam.at(id).e33*taudy.at(id).e23*urb->wind.at(id).v;
                    double A1_7e33 = 0.5*turb->lam.at(id).e33*taudy.at(id).e33*urb->wind.at(id).v;
                    
                    double A1_8e11 = 0.5*turb->lam.at(id).e11*taudz.at(id).e11*urb->wind.at(id).w;
                    double A1_8e12 = 0.5*turb->lam.at(id).e11*taudz.at(id).e12*urb->wind.at(id).w;
                    double A1_8e13 = 0.5*turb->lam.at(id).e11*taudz.at(id).e13*urb->wind.at(id).w;
                    double A1_8e21 = 0.5*turb->lam.at(id).e12*taudz.at(id).e11*urb->wind.at(id).w;
                    double A1_8e22 = 0.5*turb->lam.at(id).e12*taudz.at(id).e12*urb->wind.at(id).w;
                    double A1_8e23 = 0.5*turb->lam.at(id).e12*taudz.at(id).e13*urb->wind.at(id).w;
                    double A1_8e31 = 0.5*turb->lam.at(id).e13*taudz.at(id).e11*urb->wind.at(id).w;
                    double A1_8e32 = 0.5*turb->lam.at(id).e13*taudz.at(id).e12*urb->wind.at(id).w;
                    double A1_8e33 = 0.5*turb->lam.at(id).e13*taudz.at(id).e13*urb->wind.at(id).w;
                    
                    double A1_9e11 = 0.5*turb->lam.at(id).e21*taudz.at(id).e12*urb->wind.at(id).w;
                    double A1_9e12 = 0.5*turb->lam.at(id).e21*taudz.at(id).e22*urb->wind.at(id).w;
                    double A1_9e13 = 0.5*turb->lam.at(id).e21*taudz.at(id).e23*urb->wind.at(id).w;
                    double A1_9e21 = 0.5*turb->lam.at(id).e22*taudz.at(id).e12*urb->wind.at(id).w;
                    double A1_9e22 = 0.5*turb->lam.at(id).e22*taudz.at(id).e22*urb->wind.at(id).w;
                    double A1_9e23 = 0.5*turb->lam.at(id).e22*taudz.at(id).e23*urb->wind.at(id).w;
                    double A1_9e31 = 0.5*turb->lam.at(id).e23*taudz.at(id).e12*urb->wind.at(id).w;
                    double A1_9e32 = 0.5*turb->lam.at(id).e23*taudz.at(id).e22*urb->wind.at(id).w;
                    double A1_9e33 = 0.5*turb->lam.at(id).e23*taudz.at(id).e23*urb->wind.at(id).w;
                    
                    double A1_10e11= 0.5*turb->lam.at(id).e31*taudz.at(id).e13*urb->wind.at(id).w;
                    double A1_10e12= 0.5*turb->lam.at(id).e31*taudz.at(id).e23*urb->wind.at(id).w;
                    double A1_10e13= 0.5*turb->lam.at(id).e31*taudz.at(id).e33*urb->wind.at(id).w;
                    double A1_10e21= 0.5*turb->lam.at(id).e32*taudz.at(id).e13*urb->wind.at(id).w;
                    double A1_10e22= 0.5*turb->lam.at(id).e32*taudz.at(id).e23*urb->wind.at(id).w;
                    double A1_10e23= 0.5*turb->lam.at(id).e32*taudz.at(id).e33*urb->wind.at(id).w;
                    double A1_10e31= 0.5*turb->lam.at(id).e33*taudz.at(id).e13*urb->wind.at(id).w;
                    double A1_10e32= 0.5*turb->lam.at(id).e33*taudz.at(id).e23*urb->wind.at(id).w;
                    double A1_10e33= 0.5*turb->lam.at(id).e33*taudz.at(id).e33*urb->wind.at(id).w;
                    
                    double A1e11 = A1_1e11 + A1_2e11 + A1_3e11 + A1_4e11 +
                        A1_5e11 + A1_6e11 + A1_7e11 + A1_8e11 + A1_9e11 + A1_10e11;
                    
                    double A1e12= A1_1e12 + A1_2e12 + A1_3e12 + A1_4e12 +
                        A1_5e12 + A1_6e12 + A1_7e12 + A1_8e12 + A1_9e12 + A1_10e12;
                    
                    double A1e13= A1_1e13 + A1_2e13 + A1_3e13 + A1_4e13 +
                        A1_5e13 + A1_6e13 + A1_7e13 + A1_8e13 + A1_9e13 + A1_10e13;
                    
                    double A1e21= A1_1e21 + A1_2e21 + A1_3e21 + A1_4e21 +
                        A1_5e21 + A1_6e21 + A1_7e21 + A1_8e21 + A1_9e21 + A1_10e21;
                    
                    double A1e22= A1_1e22 + A1_2e22 + A1_3e22 + A1_4e22 +
                        A1_5e22 + A1_6e22 + A1_7e22 + A1_8e22 + A1_9e22 + A1_10e22;
                    
                    double A1e23= A1_1e23 + A1_2e23 + A1_3e23 + A1_4e23 +
                        A1_5e23 + A1_6e23 + A1_7e23 + A1_8e23 + A1_9e23 + A1_10e23;
                    
                    double A1e31= A1_1e31 + A1_2e31 + A1_3e31 + A1_4e31 +
                        A1_5e31 + A1_6e31 + A1_7e31 + A1_8e31 + A1_9e31 + A1_10e31;
                    
                    double A1e32= A1_1e32 + A1_2e32 + A1_3e32 + A1_4e32 +
                        A1_5e32 + A1_6e32 + A1_7e32 + A1_8e32 + A1_9e32 + A1_10e32;
                    
                    double A1e33= A1_1e33 + A1_2e33 + A1_3e33 + A1_4e33 +
                        A1_5e33 + A1_6e33 + A1_7e33 + A1_8e33 + A1_9e33 + A1_10e33;

      
                    bool imaginary=true;
                    matrix9 mat9;
                    while(imaginary) {
                        imaginary=false;
                        
                        mat9.e11=A1e11;
                        mat9.e12=A1e12;
                        mat9.e13=A1e13;
                        mat9.e21=A1e21;
                        mat9.e22=A1e22;
                        mat9.e23=A1e23;
                        mat9.e31=A1e31;
                        mat9.e32=A1e32;
                        mat9.e33=A1e33;
                        
                        cond_A1=matCondFro(mat9,turb->tke[id]);
                        det_A1=matrixDet(mat9);
                                                
                        double a=-1; //ax^3+bx^2+cx+d=0
                        double b=A1e11+A1e22+A1e33;
                        double c=A1e12*A1e21 + A1e13*A1e31 + A1e23*A1e32
                                -A1e11*A1e22 - A1e11*A1e33 - A1e22*A1e33;
                        double d=A1e11*A1e22*A1e33 - A1e11*A1e23*A1e32
                                -A1e21*A1e12*A1e33 + A1e21*A1e13*A1e32
                                +A1e31*A1e12*A1e23 - A1e31*A1e13*A1e22;
                        
                        //checking if the roots are real of imaginary
                        double f=( (3.0*c/a)-((b*b)/(a*a)) ) / 3.0;
                        double g=( ((2.0*b*b*b)/(a*a*a)) - ((9.0*b*c)/(a*a)) + (27.0*d/a) ) / 27.0;
                        double h= (g*g/4.0) + (f*f*f/27.0);
                        
                        double tolP=1e-100;//tolerance on positive side (as h is double not an int, we cannot use equality logical operator)
                        double tolN=-1e-100;//tolerance on negative side
              
                        //Three cases
                        if(h>1e-3){ //1 real root, 2 imaginary roots
                            imaginary=true;
                            A1e12=0;
                            A1e13=0;
                            A1e21=0;
                            A1e23=0;
                            A1e31=0;
                            A1e32=0;
                            int iV=id%nx;
                            int jV=(id/nx)%ny;
                            int kV=(id/(nx*ny))%nz;
                            //std::cerr<<"Imaginary roots ....exiting as h ="<<h<<std::endl;
                            //std::cout<< "For equatio ax^3 + bx^2 + cx + d=0"<<std::endl;
                            //std::cout<<"a :"<<a<<std::endl;
                            //std::cout<<"b :"<<b<<std::endl;
                            //std::cout<<"c :"<<c<<std::endl;
                            //std::cout<<"d :"<<d<<std::endl;
                            //std::cout<<"The original matrix is..."<<std::endl;
                            //display(mat9);
                            //std::cout<<"The d(tau)/dx  matrix is..."<<std::endl;
                            //display(taudx.at(id));
                            //std::cout<<"The d(tau)/dy  matrix is..."<<std::endl;
                            //display(taudy.at(id));
                            //std::cout<<"The d(tau)/dz  matrix is..."<<std::endl;
                            //display(taudz.at(id));
                            //std::cout<<"The tau  matrix is..."<<std::endl;
                            //display(turb->tau.at(id));
                            //std::cout<<"The lamda  matrix is..."<<std::endl;
                            //display(turb->lam.at(id));
                            //std::cout<<"The CoEps is..."<<std::endl;
                            //std::cout<<turb->CoEps.at(id)<<std::endl;
                            //std::cout<<"indicies at which this happend are (i,j,k) :"<<iV<<"   "<<jV<<"   "<<kV<<std::endl;
                        }
                        else if(h<=tolP && h>=tolN && g<=tolP && g>=tolN && f<=tolP && f>=tolN) {// All roots are real and equal
                            eigVal.at(id).e11=pow(d/a,1.0/3.0);
                            eigVal.at(id).e22=eigVal.at(id).e11;
                            eigVal.at(id).e33=eigVal.at(id).e11;
                        }
                        else{ //real roots
                            double ii=sqrt( (g*g/4.0)-h );
                            double jj=pow(ii,1.0/3.0);
                            double kk=acos( -(g/(2.0*ii)) );
                            double L=-1.0*jj;
                            double M=cos(kk/3.0);
                            double N=sqrt(3.0)*sin(kk/3.0);
                            double P=-1.0*(b/(3.0*a));
                            
                            double largest=2*jj*cos(kk/3.0)-(b/(3.0*a));
                            double middle=L*(M+N)+P;
                            double smallest=L*(M-N)+P;
                            
                            if(largest<middle) //bubble sort; sorting for largest to smallest for eigen values
                                swap(largest,middle);
                            if(middle<smallest)
                                swap(smallest,middle);
                            if(largest<middle)
                                swap(largest,middle);
                            
                            eigVal.at(id).e11=largest;
                            eigVal.at(id).e22=middle;
                            eigVal.at(id).e33=smallest;//eigen values
                            
                            //checking if eigenvalues are nan or not                  
                            //if(isnan(largest) || isnan(middle) || isnan(smallest)) {
                            //    std::cout<<"Nan: "<<largest<<"  "<<middle<<"  "<<smallest<<std::endl;
                            //    std::cout<<M<<"  "<<N<<"  "<<kk<<"  "<<g<<"  "<<ii<<std::endl;
                            //    std::cout<<i<<"  "<<j<<"  "<<k<<std::endl;
                            //    std::cout<<"The tau  matrix is..."<<std::endl;
                            //    display(turb->tau.at(id));
                            //    std::cout<<"The original matrix is..."<<std::endl;
                            //    display(mat9);
                            //}
                        }
                    }// while imaginary
      
                    //eigen Values has to be negative!!!
                    double snumm=0.;
                    double lnumm=100.;
                    
                    //      if(eigVal.at(id).e11>numm ||eigVal.at(id).e22>numm ||eigVal.at(id).e33>numm || (i==inn && j==jnn && k==knn)){
                    //if(eigVal.at(id).e11>=snumm &&eigVal.at(id).e11<lnumm) {
                    //    if(eigVal.at(id).e11<-1.0)std::cout<<"Eigen: "<<eigVal.at(id).e11<<std::endl;
                    //    number++;
                    //    flagnum=1;
                    //}
                    
                    double larMidFac=(eigVal.at(id).e11-eigVal.at(id).e22)/50.0;
                    double firstVal=eigVal.at(id).e11+larMidFac;
                    
                    double smallMidFac=(eigVal.at(id).e22-eigVal.at(id).e33)/50.0;
                    double thirdVal=eigVal.at(id).e33-smallMidFac;
                    
                    double secondVal=0.0;
                    if(larMidFac>smallMidFac)
                      secondVal=eigVal.at(id).e22+larMidFac;
                    else
                      secondVal=eigVal.at(id).e22-smallMidFac;

                    double eigValData[]={firstVal, secondVal, thirdVal};
                    
                    matrix9 eye;//identity matrix
                    
                    eye.e11=1;
                    eye.e12=0;
                    eye.e13=0;
                    eye.e21=0;
                    eye.e22=1;
                    eye.e23=0;
                    eye.e31=0;
                    eye.e32=0;
                    eye.e33=1;
                    
                    for(int ieigen=0;ieigen<3;ieigen++) {
                        
                        vec3 vecX;
                        vecX.e11=1.0;
                        vecX.e21=1.0;
                        vecX.e31=1.0;
                        
                        double err=1000;//initial error
                        double s=eigValData[ieigen];
                        
                        while(err>1.0e-5) {
                            double maxVec1=maxValAbs(vecX);
                            matrix9 idenEigVal=matrixScalarMult(eye,s);
                            matrix9 matSubs=matrixSubs(mat9,idenEigVal);
                            matrix9 matSubsInv=matrixInv(matSubs,turb->tke[id]);
                            vec3 vecY=matrixVecMult(matSubsInv,vecX);
                            vecX=vecScalarDiv(vecY,vecNorm(vecY));
                            double maxVec2=maxValAbs(vecX);
                            if(maxVec1==0.0) {
                                std::cerr<<"Divide by Zero!!! (Eulerian.cpp -3)"<<std::endl;
                                exit(1);
                            }
                            err=fabs((maxVec1-maxVec2)/maxVec1);
                        }
                        if(ieigen==0) {
                            eigVec.at(id).e11=vecX.e11;
                            eigVec.at(id).e21=vecX.e21;
                            eigVec.at(id).e31=vecX.e31;    
                            vec3 temp;
                            temp.e11=eigVec.at(id).e11;
                            temp.e21=eigVec.at(id).e21;
                            temp.e31=eigVec.at(id).e31;
                            if(vecNorm(temp)<0.9 && vecNorm(temp)>1.1) {
                                std::cerr<<"Vector is not normalized......exiting....."<<std::endl;
                                std::cout<<"Norm is :"<<vecNorm(temp)<<std::endl;
                                std::cout<<"indicies are: (i,j,k) :"<<i<<"   "<<j<<"   "<<k<<std::endl; 
                                exit(1);
                            }
                        }
                        if(ieigen==1){
                            eigVec.at(id).e12=vecX.e11;
                            eigVec.at(id).e22=vecX.e21;
                            eigVec.at(id).e32=vecX.e31;
                            vec3 temp;
                            temp.e11=eigVec.at(id).e12;
                            temp.e21=eigVec.at(id).e22;
                            temp.e31=eigVec.at(id).e32;
                            if(vecNorm(temp)<0.9 && vecNorm(temp)>1.1) {
                                std::cerr<<"Vector is not normalized......exiting....."<<std::endl;
                                std::cout<<"Norm is :"<<vecNorm(temp)<<std::endl;
                                std::cout<<"indicies are: (i,j,k) :"<<i<<"   "<<j<<"   "<<k<<std::endl; 
                                exit(1);
                            }
                        }
                        if(ieigen==2){
                            eigVec.at(id).e13=vecX.e11;
                            eigVec.at(id).e23=vecX.e21;
                            eigVec.at(id).e33=vecX.e31;
                            vec3 temp;
                            temp.e11=eigVec.at(id).e13;
                            temp.e21=eigVec.at(id).e23;
                            temp.e31=eigVec.at(id).e33;
                            if(vecNorm(temp)<0.9 && vecNorm(temp)>1.1) {
                                std::cerr<<"Vector is not vecNormalized......exiting....."<<std::endl;
                                std::cout<<"Norm is :"<<vecNorm(temp)<<std::endl;
                                std::cout<<"indicies are: (i,j,k) :"<<i<<"   "<<j<<"   "<<k<<std::endl; 
                                exit(1);
                            }
                        }
                    }
                    eigVecInv.at(id)=matrixInv(eigVec.at(id),turb->tke[id]);
                    
                    vec3 a0;
                       
                    a0.e11=0.5*(taudx.at(id).e11+taudy.at(id).e12+taudz.at(id).e13);
                    a0.e21=0.5*(taudx.at(id).e12+taudy.at(id).e22+taudz.at(id).e23);
                    a0.e31=0.5*(taudx.at(id).e13+taudy.at(id).e23+taudz.at(id).e33);
                    
                    ka0.at(id)=matrixVecMult(eigVecInv.at(id),a0);
                    
                    g2nd.at(id).e11=0.5*(turb->lam.at(id).e11*taudx.at(id).e11+turb->lam.at(id).e21*taudx.at(id).e12
                                 +turb->lam.at(id).e31*taudx.at(id).e13);
                    g2nd.at(id).e21=0.5*(turb->lam.at(id).e12*taudy.at(id).e12+turb->lam.at(id).e22*taudy.at(id).e22
                                 +turb->lam.at(id).e32*taudy.at(id).e23);
                    g2nd.at(id).e31=0.5*(turb->lam.at(id).e13*taudz.at(id).e13+turb->lam.at(id).e23*taudz.at(id).e23
                                 +turb->lam.at(id).e33*taudz.at(id).e33);
                }
            }
            
            //if(flagnum==1) std::cout<<"Total:  "<<number<<std::endl;
        }
    }
}

double Eulerian::maxValAbs(const vec3& vec) {
    double maxAbs;
    vec3 vecAbs;
    vecAbs.e11=fabs(vec.e11);
    vecAbs.e21=fabs(vec.e21);
    vecAbs.e31=fabs(vec.e31);
    
    if(vecAbs.e11>vecAbs.e21) {
        if(vecAbs.e31>vecAbs.e11) {
            maxAbs=vecAbs.e31;
        } else { 
            maxAbs=vecAbs.e11;
        }
    } else {
        if(vecAbs.e31>vecAbs.e21) {
            maxAbs=vecAbs.e31;
        } else {
            maxAbs=vecAbs.e21;
        }
    }
    return maxAbs;
}

vec3 Eulerian::vecScalarMult(const vec3& vec, const double& s){
    vec3 vecRet;
    vecRet.e11=vec.e11*s;
    vecRet.e21=vec.e21*s;
    vecRet.e31=vec.e31*s;
    
    return vecRet;
}

vec3 Eulerian::vecScalarDiv(const vec3& vec, const double& s){
    vec3 vecRet;
    if(s!=0.0){
    vecRet.e11=vec.e11/s;
    vecRet.e21=vec.e21/s;
    vecRet.e31=vec.e31/s;
    }
    else{
      std::cerr<<"Divide by ZERO!!! (Eulerian.cpp - 2)"<<std::endl;
      exit(1);
    }
    return vecRet;
}

double Eulerian::vecNorm(const vec3& vec){
    return (sqrt(vec.e11*vec.e11 + vec.e21*vec.e21 + vec.e31*vec.e31));
}

double Eulerian::matCondFro(matrix9& mat,double tke) {
    matrix9 matInv=matrixInv(mat,tke);
    double normMat=matNormFro(mat);
    double normMatInv=matNormFro(matInv);
    return  normMat*normMatInv ;
}

double Eulerian::matCondFro(const matrix6& mat6) {
    matrix9 matInv=matrixInv(mat6);
    double normMat=matNormFro(mat6);
    double normMatInv=matNormFro(matInv);
    return  normMat*normMatInv ;
}

double Eulerian::matNormFro(const matrix9& mat) {
    
    matrix9 matTrans,matMult;
    matTrans.e11=mat.e11;
    matTrans.e12=mat.e21;
    matTrans.e13=mat.e31;
    matTrans.e21=mat.e12;
    matTrans.e22=mat.e22;
    matTrans.e23=mat.e32;
    matTrans.e31=mat.e13;
    matTrans.e32=mat.e23;
    matTrans.e33=mat.e33;
    
    matMult=matrixMult(mat,matTrans);
    double sumDiag=matMult.e11+matMult.e22+matMult.e33;
    return sqrt(sumDiag);
}

double Eulerian::matNormFro(const matrix6& mat6) {
    
    matrix9 mat,matTrans,matMult;
    
    mat.e11=mat6.e11;
    mat.e12=mat6.e12;
    mat.e13=mat6.e13;
    mat.e21=mat6.e12;
    mat.e22=mat6.e22;
    mat.e23=mat6.e23;
    mat.e31=mat6.e13;
    mat.e32=mat6.e23;
    mat.e33=mat6.e33;
    
    matTrans.e11=mat.e11;
    matTrans.e12=mat.e21;
    matTrans.e13=mat.e31;
    matTrans.e21=mat.e12;
    matTrans.e22=mat.e22;
    matTrans.e23=mat.e32;
    matTrans.e31=mat.e13;
    matTrans.e32=mat.e23;
    matTrans.e33=mat.e33;
    
    matMult=matrixMult(mat,matTrans);
    double sumDiag=matMult.e11+matMult.e22+matMult.e33;
    return sqrt(sumDiag);
}

void Eulerian::swap(double &a,double &b) {
    double temp=a;
    a=b;
    b=temp;
}

double  Eulerian::matrixDet(const matrix9& mat) {
    double detMat=(mat.e11*mat.e22*mat.e33)-
    (mat.e11*mat.e23*mat.e32)-
    (mat.e12*mat.e21*mat.e33)+
    (mat.e12*mat.e23*mat.e31)+
    (mat.e13*mat.e21*mat.e32)-
    (mat.e13*mat.e22*mat.e31);
    return detMat;
}

double Eulerian::matrixDet(const matrix6& matIni) {
  
  matrix9 mat;
  mat.e11=matIni.e11;
  mat.e12=matIni.e12;
  mat.e13=matIni.e13;
  mat.e21=matIni.e12;
  mat.e22=matIni.e22;
  mat.e23=matIni.e23;
  mat.e31=matIni.e13;
  mat.e32=matIni.e23;
  mat.e33=matIni.e33;
  
  double detMat=(mat.e11*mat.e22*mat.e33)-
    (mat.e11*mat.e23*mat.e32)-
    (mat.e12*mat.e21*mat.e33)+
    (mat.e12*mat.e23*mat.e31)+
    (mat.e13*mat.e21*mat.e32)-
    (mat.e13*mat.e22*mat.e31);
  
  return detMat;
}

matrix9 Eulerian::matrixInv(matrix9& mat, double tke) {
    
    matrix9 matInv;
    
    double detMat=(mat.e11*mat.e22*mat.e33)-
                  (mat.e11*mat.e23*mat.e32)-
                  (mat.e12*mat.e21*mat.e33)+
                  (mat.e12*mat.e23*mat.e31)+
                  (mat.e13*mat.e21*mat.e32)-
                  (mat.e13*mat.e22*mat.e31);
    
    // Fix non-invertable matrix
    if (detMat==0.0) {
        mat.e12=0.0;
        mat.e13=0.0;
        mat.e21=0.0;
        mat.e23=0.0;
        mat.e31=0.0;
        mat.e32=0.0;
        mat.e11=0.67*tke;
        mat.e22=0.67*tke;
        mat.e33=0.67*tke;
    }
    
    // confirm that determinant is non-zero
    detMat=(mat.e11*mat.e22*mat.e33)-
           (mat.e11*mat.e23*mat.e32)-
           (mat.e12*mat.e21*mat.e33)+
           (mat.e12*mat.e23*mat.e31)+
           (mat.e13*mat.e21*mat.e32)-
           (mat.e13*mat.e22*mat.e31);
    
    if(detMat!=0.0) {
      matInv.e11=( (mat.e22*mat.e33)-(mat.e23*mat.e32) )/detMat;
      matInv.e12=( (mat.e13*mat.e32)-(mat.e12*mat.e33) )/detMat;
      matInv.e13=( (mat.e12*mat.e23)-(mat.e13*mat.e22) )/detMat;
      matInv.e21=( (mat.e23*mat.e31)-(mat.e21*mat.e33) )/detMat;
      matInv.e22=( (mat.e11*mat.e33)-(mat.e13*mat.e31) )/detMat;
      matInv.e23=( (mat.e13*mat.e21)-(mat.e11*mat.e23) )/detMat;
      matInv.e31=( (mat.e21*mat.e32)-(mat.e22*mat.e31) )/detMat;
      matInv.e32=( (mat.e12*mat.e31)-(mat.e11*mat.e32) )/detMat;
      matInv.e33=( (mat.e11*mat.e22)-(mat.e12*mat.e21) )/detMat;
    }
    else{
        display(mat);
        std::cerr<<"Divide by Zero!!! (Eulerian.cpp - 1,mat9)"<<std::endl;
        exit(1);
    }
    return matInv;
}


matrix9 Eulerian::matrixInv(const matrix6& matIni){
    
    matrix9 matInv,mat;
    mat.e11=matIni.e11;
    mat.e12=matIni.e12;
    mat.e13=matIni.e13;
    mat.e21=matIni.e12;
    mat.e22=matIni.e22;
    mat.e23=matIni.e23;
    mat.e31=matIni.e13;
    mat.e32=matIni.e23;
    mat.e33=matIni.e33;
    
    // looks the same as how Brian calculates the determinant for his matrix inversion
    double detMat=(mat.e11*mat.e22*mat.e33)-
      (mat.e11*mat.e23*mat.e32)-
      (mat.e12*mat.e21*mat.e33)+
      (mat.e12*mat.e23*mat.e31)+
      (mat.e13*mat.e21*mat.e32)-
      (mat.e13*mat.e22*mat.e31);
    
    // Brian has a different handle for this, if abs(detMat) < 1e-10, then set it to 10^10, with a warning which I sometimes had to turn off
    // then he just does the matrix inversion and it works out
    if(detMat!=0.0){
      
      matInv.e11=( (mat.e22*mat.e33)-(mat.e23*mat.e32) )/detMat;
      matInv.e12=( (mat.e13*mat.e32)-(mat.e12*mat.e33) )/detMat;    // brian has an extra - on the front term
      matInv.e13=( (mat.e12*mat.e23)-(mat.e13*mat.e22) )/detMat;
      matInv.e21=( (mat.e23*mat.e31)-(mat.e21*mat.e33) )/detMat;
      matInv.e22=( (mat.e11*mat.e33)-(mat.e13*mat.e31) )/detMat;
      matInv.e23=( (mat.e13*mat.e21)-(mat.e11*mat.e23) )/detMat;
      matInv.e31=( (mat.e21*mat.e32)-(mat.e22*mat.e31) )/detMat;
      matInv.e32=( (mat.e12*mat.e31)-(mat.e11*mat.e32) )/detMat;
      matInv.e33=( (mat.e11*mat.e22)-(mat.e12*mat.e21) )/detMat;
    }
    else{
        std::cout<<matIni.e23<<"  "<<mat.e23<<std::endl;
        std::cerr<<"Divide by Zero!!! (Eulerian.cpp - 1,mat6)"<<std::endl;
        exit(1);
    }
    
    return matInv;
}

matrix9 Eulerian::matrixMult(const matrix9& mat1,const matrix9& mat2) {
    
    matrix9 matMult;
    matMult.e11= mat1.e11*mat2.e11 + mat1.e12*mat2.e21 + mat1.e13*mat2.e31;
    matMult.e12= mat1.e11*mat2.e12 + mat1.e12*mat2.e22 + mat1.e13*mat2.e32;
    matMult.e13= mat1.e11*mat2.e13 + mat1.e12*mat2.e23 + mat1.e13*mat2.e33;
    matMult.e21= mat1.e21*mat2.e11 + mat1.e22*mat2.e21 + mat1.e23*mat2.e31;
    matMult.e22= mat1.e21*mat2.e12 + mat1.e22*mat2.e22 + mat1.e23*mat2.e32;
    matMult.e23= mat1.e21*mat2.e13 + mat1.e22*mat2.e23 + mat1.e23*mat2.e33;
    matMult.e31= mat1.e31*mat2.e11 + mat1.e32*mat2.e21 + mat1.e33*mat2.e31;
    matMult.e32= mat1.e31*mat2.e12 + mat1.e32*mat2.e22 + mat1.e33*mat2.e32;
    matMult.e33= mat1.e31*mat2.e13 + mat1.e32*mat2.e23 + mat1.e33*mat2.e33;
    
    return matMult;
}

matrix9 Eulerian::matrixScalarMult(const matrix9& mat ,const double& s) {
    matrix9 matRet;
    matRet.e11= mat.e11*s;
    matRet.e12= mat.e12*s;
    matRet.e13= mat.e13*s;
    matRet.e21= mat.e21*s;
    matRet.e22= mat.e22*s;
    matRet.e23= mat.e23*s;
    matRet.e31= mat.e31*s;
    matRet.e32= mat.e32*s;
    matRet.e33= mat.e33*s;
    
    return matRet;
}

matrix9 Eulerian::matrixSubs(const matrix9& mat1,const matrix9& mat2) {
    
    matrix9 matSubs;
    matSubs.e11= mat1.e11-mat2.e11;
    matSubs.e12= mat1.e12-mat2.e12;
    matSubs.e13= mat1.e13-mat2.e13;
    matSubs.e21= mat1.e21-mat2.e21;
    matSubs.e22= mat1.e22-mat2.e22;
    matSubs.e23= mat1.e23-mat2.e23;
    matSubs.e31= mat1.e31-mat2.e31;
    matSubs.e32= mat1.e32-mat2.e32;
    matSubs.e33= mat1.e33-mat2.e33;
    
    return matSubs;
}

vec3 Eulerian::matrixVecMult(const matrix9& mat,const vec3& vec) {
    
    vec3 matVecMult;
    matVecMult.e11= mat.e11*vec.e11 + mat.e12*vec.e21 + mat.e13*vec.e31;
    matVecMult.e21= mat.e21*vec.e11 + mat.e22*vec.e21 + mat.e23*vec.e31;
    matVecMult.e31= mat.e31*vec.e11 + mat.e32*vec.e21 + mat.e33*vec.e31;
    return matVecMult;
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
