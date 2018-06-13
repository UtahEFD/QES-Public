 
#include <netcdfcpp.h>
#include <stdlib.h>
#include <stdio.h>

 
 /* This is the name of the data file we will create. */
 #define FILE_NAME "simple_xy.nc"
 
 /* Handle errors by printing an error message and exiting with a
  * non-zero status. */
 #define ERRCODE 2
 #define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}
 
using namespace std;

// We are writing 2D data, a 6 x 12 grid. 
static const int NDIMS = 3;
static const int NX = 6;
static const int NY = 12;
static const int NZ = 4;

// Return this in event of a problem.
static const int NC_ERR = 2;

int 
main(void)
{
   // This is the data array we will write. It will just be filled
   // with a progression of numbers for this example.
   int*** dataOut;
   dataOut = new int**[NX];
   for (int i = 0; i < NX; i++)
   {
	dataOut[i] = new int*[NY];
	for (int j = 0; j < NY; j++)
		dataOut[i][j] = new int[NZ];
   }
  
   // Create some pretend data. If this wasn't an example program, we
   // would have some real data to write, for example, model output.
   for(int i = 0; i < NX; i++) 
	  for(int j = 0; j < NY; j++)
		for (int z = 0; z < NZ; z++)
			dataOut[i][j][z] = (i * NY + j) * (z + 1);

   // Create the file. The Replace parameter tells netCDF to overwritea
   // this file, if it already exists.
   NcFile dataFile("simple_xy.nc", NcFile::Replace);

   // You should always check whether a netCDF file creation or open
   // constructor succeeded.
   if (!dataFile.is_valid())
   {
	  cout << "Couldn't open file!\n";
	  return NC_ERR;
   }
  
   // For other method calls, the default behavior of the C++ API is
   // to exit with a message if there is an error.  If that behavior
   // is OK, there is no need to check return values in simple cases
   // like the following.

   // When we create netCDF dimensions, we get back a pointer to an
   // NcDim for each one.
   NcDim* xDim = dataFile.add_dim("x", NX);
   NcDim* yDim = dataFile.add_dim("y", NY);
   NcDim* zDim = dataFile.add_dim("z", NZ);
  
   // Define a netCDF variable. The type of the variable in this case
   // is ncInt (32-bit integer).
   NcVar *data = dataFile.add_var("data", ncInt, xDim, yDim, zDim);
	 
   // Write the pretend data to the file. Although netCDF supports
   // reading and writing subsets of data, in this case we write all
   // the data in one operation.
   data->put(&dataOut[0][0][0], NX, NY, NZ);

   // The file will be automatically close when the NcFile object goes
   // out of scope. This frees up any internal netCDF resources
   // associated with the file, and flushes any buffers.
   cout << "*** SUCCESS writing example file simple_xy.nc!" << endl;

   return 0;
}