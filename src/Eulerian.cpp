
#include "Eulerian.h"


Eulerian::Eulerian( PlumeInputData* PID,URBGeneralData* UGD,TURBGeneralData* TGD, const bool& debug_val)
{
    
    std::cout<<"[Eulerian] \t Reading CUDA-URB & CUDA-TGD fields "<<std::endl;
    
    // copy debug information
    debug = debug_val;
    
    // copy UGD grid information
    nz = UGD->nz;
    ny = UGD->ny;
    nx = UGD->nx;
    dz = UGD->dz;
    dy = UGD->dy;
    dx = UGD->dx;

    iStart=1;iEnd=nx-2;
    jStart=1;jEnd=ny-2;
    kStart=1;kEnd=nz-2;
        
    // get the TGD domain start and end values, other TGD grid information
    TurbXstart = TGD->x_cc[iStart];
    TurbXend = TGD->x_cc[iEnd];
    
    TurbYstart = TGD->y_cc[jStart];
    TurbYend = TGD->y_cc[jEnd];
    
    TurbZstart = TGD->y_fc[kStart];
    TurbZend = TGD->z_cc[kEnd];

    // LA future work: probably need to do something right here to determine what is the 
    // true domain size not just for TGD, but for urb as well
    
    // set additional values from the input
    C_0 = PID->simParams->C_0;

    // set the tau gradient sizes
    dtxxdx.resize(UGD->numcell_cent,0.0);
    dtxydy.resize(UGD->numcell_cent,0.0);
    dtxzdz.resize(UGD->numcell_cent,0.0);

    dtxydx.resize(UGD->numcell_cent,0.0);
    dtyydy.resize(UGD->numcell_cent,0.0);
    dtyzdz.resize(UGD->numcell_cent,0.0);

    dtxzdx.resize(UGD->numcell_cent,0.0);
    dtyzdy.resize(UGD->numcell_cent,0.0);
    dtzzdz.resize(UGD->numcell_cent,0.0);
    
    // set the flux_div to the right size
    flux_div_x.resize(UGD->numcell_cent,0.0);
    flux_div_y.resize(UGD->numcell_cent,0.0);
    flux_div_z.resize(UGD->numcell_cent,0.0);

    // temp storage of sigma's
    sig_x.resize(UGD->numcell_cent,0.0);
    sig_y.resize(UGD->numcell_cent,0.0);
    sig_z.resize(UGD->numcell_cent,0.0);

    for(int idx = 0; idx < (nx-1)*(ny-1)*(nz-1); idx++) {
        sig_x.at(idx) = std::sqrt(TGD->txx.at(idx));
        sig_y.at(idx) = std::sqrt(TGD->tyy.at(idx));
        sig_z.at(idx) = std::sqrt(TGD->tzz.at(idx));
    }

    // compute stress gradients
    createTauGrads(TGD);

    // use the stress gradients to calculate the flux div
    createFluxDiv();

}

void Eulerian::createTauGrads(TURBGeneralData* TGD)
{
    std::cout<<"[Eulerian] \t Computing stress gradients "<<std::endl;
    
    // start recording execution time
    if( debug == true ) {
        timers.startNewTimer("calcGradient");
    }
    
    // 2nd order Forward differencing up to 2 in from the edge in the direction of the gradient,
    // 2nd order Backward differencing for the last two in the direction of the gradient,
    // all this over all cells in the other two directions

    // DX forward differencing
    for(int k=kStart; k<kEnd; ++k) {
        for(int j=jStart; j<jEnd; ++j) {
            for(int i=iStart; i<iEnd-2; ++i) {
                // Provides a linear index based on the 3D (i, j, k)
                int idx = k*(ny-1)*(nx-1) + j*(nx-1) + i;
                setDX_Forward( TGD, idx );
            }
        }
    }

    // DX backward differencing
    for(int k=kStart; k<kEnd; ++k) {
        for(int j=jStart; j<jEnd; ++j) {
            for(int i=iEnd-2; i<iEnd; ++i) {
                // Provides a linear index based on the 3D (i, j, k)
                int idx = k*(ny-1)*(nx-1) + j*(nx-1) + i;
                setDX_Backward( TGD, idx );
            }
        }
    }


    // DY forward differencing
    for(int k=kStart; k<kEnd; ++k) {
        for(int j=jStart; j<jEnd-2; ++j) {
            for(int i=iStart; i<iEnd; ++i) {
                // Provides a linear index based on the 3D (i, j, k)
                int idx = k*(ny-1)*(nx-1) + j*(nx-1) + i;
                setDY_Forward( TGD, idx );
            }
        }
    }

    // DY backward differencing
    for(int k=kStart; k<kEnd; ++k) {
        for(int j=jEnd-2; j<jEnd; ++j) {
            for(int i=iStart; i<iEnd; ++i) {
                // Provides a linear index based on the 3D (i, j, k)
                int idx = k*(ny-1)*(nx-1) + j*(nx-1) + i;
                setDY_Backward( TGD, idx );
            }
        }
    }


    // DZ forward differencing
    for(int k=kStart; k<kEnd-2; ++k) {
        for(int j=jStart; j<jEnd; ++j) {
            for(int i=iStart; i<iEnd; ++i) {
                // Provides a linear index based on the 3D (i, j, k)
                int idx = k*(ny-1)*(nx-1) + j*(nx-1) + i;
                setDZ_Forward( TGD, idx );
            }
        }
    }

    // DZ backward differencing
    for(int k=kEnd-2; k<kEnd; ++k) {
        for(int j=jStart; j<jEnd; ++j) {
            for(int i=iEnd; i<iEnd; ++i) {
                // Provides a  linear index based on the 3D (i, j, k)
                int idx = k*(ny-1)*(nx-1) + j*(nx-1) + i;
                setDZ_Backward( TGD, idx );
            }
        }
    }
    
    // print out elapsed execution time
    if( debug == true ) {
        timers.printStoredTime("calcGradient");
    }
}

void Eulerian::createFluxDiv() 
{
    std::cout << "[Eulerian] \t Computing flux_div values " << std::endl;
    
    // loop through each cell and calculate the flux_div from the gradients of tao
    for(int idx = 0; idx < (nx-1)*(ny-1)*(nz-1); idx++) {
        flux_div_x.at(idx) = dtxxdx.at(idx) + dtxydy.at(idx) + dtxzdz.at(idx);
        flux_div_y.at(idx) = dtxydx.at(idx) + dtyydy.at(idx) + dtyzdz.at(idx);
        flux_div_z.at(idx) = dtxzdx.at(idx) + dtyzdy.at(idx) + dtzzdz.at(idx);
    }
}


// this gets around the problem of repeated or not repeated information, just needs called once before each interpolation,
// then intepolation on all kinds of datatypes can be done
void Eulerian::setInterp3Dindexing(const double& par_xPos, const double& par_yPos, const double& par_zPos)
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
    double par_x = par_xPos - TurbXstart;
    double par_y = par_yPos - TurbYstart;
    double par_z = par_zPos - TurbZstart;

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
    ii = floor(par_x/(dx+1e-9));
    jj = floor(par_y/(dy+1e-9));
    kk = floor(par_z/(dz+1e-9));

    // fractional distance between nearest nodes
    iw = (par_x/(dx+1e-9) - floor(par_x/(dx+1e-9)));
    jw = (par_y/(dy+1e-9) - floor(par_y/(dy+1e-9)));
    kw = (par_z/(dz+1e-9) - floor(par_z/(dz+1e-9)));

    // initialize the counters from the indices
    ip = 1;
    jp = 1;
    kp = 1;

    // now set the indices and the counters from the indices
    if( nx == 1 ) {
        ii = 0;
        iw = 0.0;
        ip = 0;
    }
    if( ny == 1 ) {
        jj = 0;
        jw = 0.0;
        jp = 0;
    }
    if( nz == 1 ) {
        kk = 0;
        kw = 0.0;
        kp = 0;
    }

    // now check to make sure that the indices are within the Eulerian grid domain
    // Notice that this no longer includes throwing an error if particles touch the far walls
    // because adding a small number to dx in the index calculation forces the index to be completely left side biased
    if( ii < 0 || ii+ip > nx-1 ) {
        std::cerr << "ERROR (Eulerian::setInterp3Dindexing): particle x position is out of range! x = \"" << par_xPos 
            << "\" ii+ip = \"" << ii << "\"+\"" << ip << "\",   nx-1 = \"" << nx-1 << "\"" << std::endl;
        exit(1);
    }
    if( jj < 0 || jj+jp > ny-1 ) {
        std::cerr << "ERROR (Eulerian::setInterp3Dindexing): particle y position is out of range! y = \"" << par_yPos 
            << "\" jj+jp = \"" << jj << "\"+\"" << jp << "\",   ny-1 = \"" << ny-1 << "\"" << std::endl;
        exit(1);
    }
    if( kk < 0 || kk+kp > nz-1 ) {
        std::cerr << "ERROR (Eulerian::setInterp3Dindexing): particle z position is out of range! z = \"" << par_zPos 
            << "\" kk+kp = \"" << kk << "\"+\"" << kp << "\",   nz-1 = \"" << nz-1 << "\"" << std::endl;
        exit(1);
    }

}

// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
double Eulerian::interp3D(const std::vector<float>& EulerData)
{

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

    return (u_high-u_low)*kw + u_low;

}

// always call this after setting the interpolation indices with the setInterp3Dindexing() function!
double Eulerian::interp3D(const std::vector<double>& EulerData)
{

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

    return (u_high-u_low)*kw + u_low;

}
