#include "NetCDFData.h"


void NetCDFData::getData(float* newX, float* newY, float* newZ, double* newU, double* newV, double* newW, int newDX, int newDY, int newDZ)
{
	dimX = newDX;
	dimY = newDY;
	dimZ = newDZ;

    x = new float [dimX-1];
    y = new float [dimY-1];
    z = new float [dimZ-1];

    for (int i = 0; i < dimX - 1; i++)
    	x[i] = newX[i];

    for (int i = 0; i < dimY - 1; i++)
    	y[i] = newY[i];

    for (int i = 0; i < dimZ - 1; i++)
    	z[i] = newZ[i];


    u = new double [dimX * dimY * dimZ];
    v = new double [dimX * dimY * dimZ];
    w = new double [dimX * dimY * dimZ];
	
	for (int i = 0; i < dimX * dimY * dimZ; i++){
		u[i] = newU[i];
		v[i] = newV[i];
		w[i] = newW[i];
	}
}

bool NetCDFData::outputCellFaceResults(std::string fileName)
{


    NcError err(NcError::silent_nonfatal);
    NcFile dataFile(fileName.c_str(), NcFile::Replace);
    if (!dataFile.is_valid())
    	return false;

    std::cout << "here1\n";

    NcDim *Nx, *Ny, *Nz;
    //if (!(latDim = dataFile.add_dim("latitude", NLAT)))
    if (!(Nx = dataFile.add_dim("x", dimX - 1 )))
    	return false;
    if (!(Ny = dataFile.add_dim("y", dimY - 1 )))
    	return false;
	if (!(Nz = dataFile.add_dim("z", dimZ - 1 )))
    	return false;

std::cout << "here2\n";

    NcVar *xVar, *yVar, *zVar;
    if (!(xVar = dataFile.add_var("x", ncFloat, Nx )))
    	return false;
    if (!(yVar = dataFile.add_var("y", ncFloat, Ny )))
    	return false;
	if (!(zVar = dataFile.add_var("z", ncFloat, Nz )))
    	return false;

std::cout << "here3\n";

    if (!(xVar->add_att("units", "meters")))
    	return false;
    if (!(yVar->add_att("units", "meters")))
    	return false;
    if (!(zVar->add_att("units", "meters")))
    	return false;

std::cout << "here4\n";

    NcVar *velX, *velY, *velZ;
    if (!(velX = dataFile.add_var("velocityX", ncFloat, Nx, Ny, Nz)))
    	return false;
    if (!(velY = dataFile.add_var("velocityY", ncFloat, Nx, Ny, Nz)))
    	return false;
    if (!(velZ = dataFile.add_var("velocityZ", ncFloat, Nx, Ny, Nz)))
    	return false;

std::cout << "here5\n";

    if (!(velX->add_att("units", "xVel")))
    	return false;
    if (!(velY->add_att("units", "yVel")))
    	return false;
    if (!(velZ->add_att("units", "zVel")))
    	return false;

std::cout << "here6\n";

    if (!xVar->put(x, dimX - 1))
    	return false;
    if (!yVar->put(y, dimY - 1))
    	return false;
    if (!zVar->put(z, dimZ - 1))
    	return false;

    float ***u_out, ***v_out, ***w_out;
    u_out = new float** [dimX-1];
    v_out = new float** [dimX-1];
    w_out = new float** [dimX-1];
	
	for (int i = 0; i < dimX - 1; i++){
		u_out[i] = new float* [dimY-1];
		v_out[i] = new float* [dimY-1];
		w_out[i] = new float* [dimY-1];
		for (int j = 0; j < dimY - 1; j++){
			u_out[i][j] = new float [dimZ-1];
			v_out[i][j] = new float [dimZ-1];
			w_out[i][j] = new float [dimZ-1];
			for (int k = 0; k < dimZ - 1; k++)
			{
				int icell_face = i + j*dimX + k*dimY*dimZ;
				u_out[i][j][k] = (float)u[icell_face];
				v_out[i][j][k] = (float)v[icell_face];
				w_out[i][j][k] = (float)w[icell_face];
			}
		}
	}
std::cout << "here7\n";

    if (!velX->put(&u_out[0][0][0], dimX - 1, dimY - 1, dimZ - 1))
    	return false;
    if (!velY->put(&v_out[0][0][0], dimX - 1, dimY - 1, dimZ - 1))
    	return false;
    if (!velZ->put(&w_out[0][0][0], dimX - 1, dimY - 1, dimZ - 1))
    	return false;

    return true;
}