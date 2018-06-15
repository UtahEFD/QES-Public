#include "NetCDFData.h"


void NetCDFData::getData(float* newX, float* newY, float* newZ, double* newU, double* newV, double* newW, int newDX, int newDY, int newDZ)
{
	dimX = newDX;
	dimY = newDY;
	dimZ = newDZ;

    x = new float [dimX-1];
    y = new float [dimY-1];
    z = new float [dimZ-2];

    for (int i = 0; i < dimX - 1; i++)
    	x[i] = newX[i];

    for (int i = 0; i < dimY - 1; i++)
    	y[i] = newY[i];

    for (int i = 0; i < dimZ - 1; i++)
    	z[i] = newZ[i];


    u = new double [(dimX - 1) * (dimY - 1) * (dimZ- 2)];
    v = new double [(dimX - 1) * (dimY - 1) * (dimZ- 2)];
    w = new double [(dimX - 1) * (dimY - 1) * (dimZ- 2)];
	
	for (int i = 0; i < (dimX - 1) * (dimY - 1) * (dimZ- 2); i++){
		u[i] = newU[i];
		v[i] = newV[i];
		w[i] = newW[i];
	}
}

bool NetCDFData::outputCellFaceResults(std::string fileName)
{    

    int lenX = dimX - 1, lenY = dimY - 1, lenZ = dimZ - 2;


    try 
    {
        NcFile dataFile(fileName.c_str(), NcFile::replace);

        std::cout << "here1\n";

        NcDim Nx, Ny, Nz;
        Nx = dataFile.addDim("x", lenX);
        Ny = dataFile.addDim("y", lenY);
    	Nz = dataFile.addDim("z", lenZ);

    std::cout << "here2\n";

        NcVar xVar, yVar, zVar;
        xVar = dataFile.addVar("x", ncFloat, Nx );
        yVar = dataFile.addVar("y", ncFloat, Ny );
    	zVar = dataFile.addVar("z", ncFloat, Nz );

    std::cout << "here3\n";

        xVar.putVar(x);
        yVar.putVar(y);
        zVar.putVar(z);


        xVar.putAtt("units", "meters");
        yVar.putAtt("units", "meters");
        zVar.putAtt("units", "meters");

        std::vector<NcDim> dimVector;
        dimVector.push_back(Nx);
        dimVector.push_back(Ny);
        dimVector.push_back(Nz);

    std::cout << "here4\n";

        NcVar velX, velY, velZ;
        velX = dataFile.addVar("velocityX", ncDouble, dimVector);
        velY = dataFile.addVar("velocityY", ncDouble, dimVector);
        velZ = dataFile.addVar("velocityZ", ncDouble, dimVector);

    std::cout << "here5\n";

        velX.putAtt("units", "xVel");
        velY.putAtt("units", "yVel");
        velZ.putAtt("units", "zVel");

    std::cout << "here6\n";

    double *u_out, *v_out, *w_out;
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
            }


       std::vector<size_t> start, count;
       start.push_back(0);
       start.push_back(0);
       start.push_back(0);
       count.push_back(lenX);
       count.push_back(lenY);
       count.push_back(lenZ);

    std::cout << "here7\n";

       velX.putVar(start,count, u);
       velY.putVar(start,count, v);
       velZ.putVar(start,count, w);


    std::cout << "here8\n";

        dataFile.close();
    }
    catch(NcException& e)
    {
      e.what(); 
      return false;
    }
    return true;
}