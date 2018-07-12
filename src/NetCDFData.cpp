#include "NetCDFData.h"


void NetCDFData::getData(float* newX, float* newY, float* newZ, double* newU, double* newV, double* newW, int newDX, int newDY, int newDZ)
{
	dimX = newDX-1;
	dimY = newDY-1;
	dimZ = newDZ-2;

    x = new float [dimX];
    y = new float [dimY];
    z = new float [dimZ];

    for (int i = 0; i < dimX; i++)
    	x[i] = newX[i];

    for (int i = 0; i < dimY; i++)
    	y[i] = newY[i];

    for (int i = 0; i < dimZ; i++)
    	z[i] = newZ[i];


    u = new double [(dimX) * (dimY) * (dimZ)];
    v = new double [(dimX) * (dimY) * (dimZ)];
    w = new double [(dimX) * (dimY) * (dimZ)];
	
    for (int k = 0; k < dimZ; k++) 
        for (int j = 0; j < dimY; j++)
            for (int i = 0; i < dimX; i++)
            {
                int iAll = i + j*(newDX) + k*(newDX)*(newDY);
                int iReduced = i + j*(dimX) + k*(dimX)*(dimY);
                u[iReduced] = newU[iAll];
                v[iReduced] = newV[iAll];
                w[iReduced] = newW[iAll];    
            }
}

void NetCDFData::getDataICell(int* newICellFlags, float* newX, float* newY, float* newZ, int nDX, int nDY, int nDZ, long newSize)
{
    size = newSize;
    dimXF = nDX;
    dimYF = nDY;
    dimZF = nDZ - 1;

    xF = new float[dimXF];
    for (int i = 0; i < dimXF; i++)
        xF[i] = newX[i];

    yF = new float[dimYF];
    for (int i = 0; i < dimYF; i++)
        yF[i] = newY[i];

    zF = new float[dimZF];
    for (int i = 0; i < dimZF; i++)
        zF[i] = newZ[i];

    iCellFlags = new int[size];
    for (int i = 0; i < size; i++)
        iCellFlags[i] = newICellFlags[i];
}

bool NetCDFData::outputCellFaceResults(std::string fileName)
{    

    int lenX = dimX, lenY = dimY, lenZ = dimZ;


    try 
    {
        NcFile dataFile(fileName.c_str(), NcFile::replace);

        NcDim Nx, Ny, Nz;
        Nx = dataFile.addDim("x", lenX);
        Ny = dataFile.addDim("y", lenY);
    	Nz = dataFile.addDim("z", lenZ);

        NcVar xVar, yVar, zVar;
        xVar = dataFile.addVar("x", ncFloat, Nx );
        yVar = dataFile.addVar("y", ncFloat, Ny );
    	zVar = dataFile.addVar("z", ncFloat, Nz );


        xVar.putVar(x);
        yVar.putVar(y);
        zVar.putVar(z);


        xVar.putAtt("units", "meters");
        yVar.putAtt("units", "meters");
        zVar.putAtt("units", "meters");

        std::vector<NcDim> dimVector;
        dimVector.push_back(Nz);
        dimVector.push_back(Ny);
        dimVector.push_back(Nx);

        NcVar velX, velY, velZ;
        velX = dataFile.addVar("velocityX", ncDouble, dimVector);
        velY = dataFile.addVar("velocityY", ncDouble, dimVector);
        velZ = dataFile.addVar("velocityZ", ncDouble, dimVector);

        velX.putAtt("units", "xVel");
        velY.putAtt("units", "yVel");
        velZ.putAtt("units", "zVel");

  /*  double *u_out, *v_out, *w_out;
    u_out = new double[lenX * lenY * lenZ];
    v_out = new double[lenX * lenY * lenZ];
    w_out = new double[lenX * lenY * lenZ];
    for (int i = 0; i < lenX; i++)
        for (int j = 0; j < lenY; j++)
            for (int k = 0; k < lenZ; k++)
            {
                int idx = (lenY * lenX) * k +    lenX * j  +   i;
                u_out[idx] = k;
                v_out[idx] = 1.0;
                w_out[idx] = 1.0;
            }*/
       std::vector<size_t> start, count;
       start.push_back(0);
       start.push_back(0);
       start.push_back(0);
       count.push_back(lenZ);
       count.push_back(lenY);
       count.push_back(lenX);

       velX.putVar(start,count, u);
       velY.putVar(start,count, v);
       velZ.putVar(start,count, w);


        dataFile.close();
    }
    catch(NcException& e)
    {
      e.what(); 
      return false;
    }
    return true;
}

bool NetCDFData::outputICellFlags(std::string fileName)
{    

    int lenX = dimXF, lenY = dimYF, lenZ = dimZF, lenS = size;


    try 
    {
        NcFile dataFile(fileName.c_str(), NcFile::replace);


        NcDim Nx, Ny, Nz;
        Nx = dataFile.addDim("x", lenX);
        Ny = dataFile.addDim("y", lenY);
        Nz = dataFile.addDim("z", lenZ);


        NcVar xVar, yVar, zVar;
        xVar = dataFile.addVar("x", ncFloat, Nx );
        yVar = dataFile.addVar("y", ncFloat, Ny );
        zVar = dataFile.addVar("z", ncFloat, Nz );


        xVar.putVar(xF);
        yVar.putVar(yF);
        zVar.putVar(zF);


        xVar.putAtt("units", "meters");
        yVar.putAtt("units", "meters");
        zVar.putAtt("units", "meters");

        std::vector<NcDim> dimVector;
        dimVector.push_back(Nz);
        dimVector.push_back(Ny);
        dimVector.push_back(Nx);

        NcVar cellVals;
        cellVals = dataFile.addVar("iCellFlag Values", ncInt, dimVector);


        cellVals.putAtt("value", "flags");


       std::vector<size_t> start, count;
       start.push_back(0);
       start.push_back(0);
       start.push_back(0);
       count.push_back(lenZ);
       count.push_back(lenY);
       count.push_back(lenX);

       cellVals.putVar(start,count, iCellFlags);

        dataFile.close();
    }
    catch(NcException& e)
    {
      e.what(); 
      return false;
    }
    return true;
}