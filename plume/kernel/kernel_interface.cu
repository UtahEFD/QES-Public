/*
* kernel_interface.cu
* This file is part of CUDAPLUME
*
* Copyright (C) 2012 - Alex Geng
*
* CUDAPLUME is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* CUDAPLUME is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with CUDAPLUME. If not, see <http://www.gnu.org/licenses/>.
*/

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"
 // includes cuda.h and cuda_runtime_api.h 
#include <cutil_inline.h>   
#include <cstdlib>
#include <cstdio>
#include <string.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cuda_gl_interop.h>
#include <iomanip> 
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include "device/advect_kernel.cu" 
#include "device/cell_concentration.cu"   
#include <cstdlib>
#include <sys/time.h>

extern "C"
{  
uint numThreads, numBlocks; 
  
 extern
// void bindTexture(cudaArray *cellArray, cudaChannelFormatDesc channelDesc, CellTextureType textureType)
void bindTexture(const cudaArray *&cellArray, const cudaChannelFormatDesc &channelDesc, const char* texname)
{   
  const textureReference* texPt=NULL; 
  cudaGetTextureReference(&texPt, texname);  
  ((textureReference *)texPt)->normalized = false;                      // access with nonnormalized texture coordinates 
  ((textureReference *)texPt)->filterMode = cudaFilterModePoint;      // 
  ((textureReference *)texPt)->addressMode[0] = cudaAddressModeClamp;   // Clamp texture coordinates
  ((textureReference *)texPt)->addressMode[1] = cudaAddressModeClamp;
  ((textureReference *)texPt)->addressMode[2] = cudaAddressModeClamp;
  cutilSafeCall(cudaBindTextureToArray(texPt, cellArray, &channelDesc));  
}  

void cudaInit(int argc, char **argv)
{   
  int devID;
  // use command-line specified CUDA device, otherwise use device with highest Gflops/s
//   if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
//       devID = cutilDeviceInit(argc, argv);
//       if (devID < 0) {
// 	  printf("No CUDA Capable devices found, exiting...\n"); 
//       }
//   } else {
      devID = cutGetMaxGflopsDeviceId();
      cudaSetDevice( devID );
//   }
}

void cudaGLInit(int argc, char **argv)
{   
  
//   computeGridSize(numParticles, 256, numBlocks, numThreads);
//   random_file.open ("random_gen_by_global.txt", std::ios::out);
  // use command-line specified CUDA device, otherwise use device with highest Gflops/s
//   if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
//       cutilDeviceInit(argc, argv);
//       printf("123123\n");
//   } else {
      cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
//   }
}

void allocateArray(void **devPtr, size_t size)
{
  cutilSafeCall(cudaMalloc(devPtr, size));
}

void freeArray(void *devPtr)
{
  cutilSafeCall(cudaFree(devPtr));
}

void threadSync()
{
  cutilSafeCall(cutilDeviceSynchronize());
}

void copyArrayToDevice(void* device, const void* host, int offset, int size)
{
  cutilSafeCall(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
}

void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
{
  cutilSafeCall(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, 
					      cudaGraphicsMapFlagsNone));
}

void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
  cutilSafeCall(cudaGraphicsUnregisterResource(cuda_vbo_resource));	
}

void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
{
  void *ptr;
  cutilSafeCall(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
  size_t num_bytes; 
  cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes, *cuda_vbo_resource));
  return ptr;
}

void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
  cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

void copyArrayFromDevice(void* host, const void* device, 
			struct cudaGraphicsResource **cuda_vbo_resource, int size)
{   
  if (cuda_vbo_resource)
      device = mapGLBufferObject(cuda_vbo_resource);

  cutilSafeCall(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
  
  if (cuda_vbo_resource)
      unmapGLBufferObject(*cuda_vbo_resource);
}

void setParameters(ConstParams *hostParams)
{
  // copy parameters to constant memory
  cutilSafeCall( cudaMemcpyToSymbol(g_params, hostParams, sizeof(ConstParams)) );
}
 
// compute grid and thread block size for a given number of elements
//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b){
  return (a % b != 0) ? (a / b + 1) : (a / b);
}
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
  numThreads = min(blockSize, n);
  numBlocks = iDivUp(n, numThreads);
}
  
 
void advectPar_with_textureMemory(float *pos, float *winP, uint* concens, 
				  float deltaTime,  uint numParticles)
// void advectPar_with_textureMemory(float *pos, float *winP, bool* seeds_flag,
// 		     uint* concens, float deltaTime,  uint numParticles)
// 		     float deltaTime,  uint numParticles)
{   
//   thrust::device_ptr<uint>   d_concensPtr(concens);  
//   for(int i=153*110*30-10; i<153*110*30; i++)
//   {
//      d_concensPtr[i]++;
//   }
//   return;
  thrust::device_ptr<float4> dev_pos4((float4 *)pos);
  thrust::device_ptr<float4> dev_winP4((float4 *)winP);  

  timeval t;
  gettimeofday(&t,0);
  thrust::for_each(
      thrust::make_zip_iterator(
// 	thrust::make_tuple(dev_pos4, dev_winP4, devDebug.begin()) 
	thrust::make_tuple(dev_pos4, dev_winP4) 
      ), 
      thrust::make_zip_iterator(
// 	thrust::make_tuple(dev_pos4+numParticles, dev_winP4+numParticles, devDebug.end())
	thrust::make_tuple(dev_pos4+numParticles, dev_winP4+numParticles)
     ), 
      advect_functor( t.tv_sec * 1000LL + t.tv_usec / 1000 + getpid() )
  );
//   std::cout<<t.tv_sec * 1000LL + t.tv_usec / 1000<<"\n";
//   if(numParticles >= 500)
//   {
//     computeGridSize(numParticles, 64, numBlocks, numThreads); 
//     concentration_kernel<<< numBlocks, numThreads >>>((float4*)(pos),
// 						    concens,
// 						    numParticles 
// 						   ); 
//     std::cout<<"numeject"<<numeject<<"\n";
//     cal_concentration(dPos, m_dConcetration, numeject);    
//     b_print_concentration = false;
//   }
   
}     

void cal_concentration(float *pos, uint* concens, const uint &numParticles, const uint &total_particles)
{    
  
  computeGridSize(numParticles, 64, numBlocks, numThreads); 
  concentration_kernel<<< numBlocks, numThreads >>>((float4*)(pos),
						    concens,
						    numParticles 
// 						    , thrust::raw_pointer_cast(&devbug[0])
						   ); 
  
  
  if(numParticles >= total_particles)
  { 
    std::ofstream random_file;
    random_file.open ("cal_concentrations.csv", std::ios::out); 
//     
   int numBoxX=60, numBoxY=55, numBoxZ=25;
    thrust::host_vector<double>xBoxCen(numBoxX*numBoxY*numBoxZ),
			     yBoxCen(numBoxX*numBoxY*numBoxZ), 
			     zBoxCen(numBoxX*numBoxY*numBoxZ); 
  double quanX=2, quanY=2, quanZ=1.2;
  double xBoxSize=2, yBoxSize=2, zBoxSize=1.2;
  int lBndx =33, lBndy =0, lBndz=0;
  int id=0, zR=0;
  
  for(int k=0;k<numBoxZ;++k){
    int yR=0;
    for(int j=0;j<numBoxY;++j){
      int xR=0;
      for(int i=0;i<numBoxX;++i){
	id=k*numBoxY*numBoxX+j*numBoxX+i;
	
	xBoxCen[id]=lBndx+xR*(quanX)+xBoxSize/2.0;
	yBoxCen[id]=lBndy+yR*(quanY)+yBoxSize/2.0; 
	zBoxCen[id]=lBndz+zR*(quanZ)+zBoxSize/2.0;	

	xR++;
      }
      yR++;
    }
    zR++;
  }

    double conc = 0.1/(899*4.8*100000);
    thrust::device_ptr<uint> d_concensPtr(concens);  
    for(int index=0; index<60*55*25; index++)
    {  
	random_file<<xBoxCen[index]<<"  "<<yBoxCen[index]<<"  "<<zBoxCen[index]<<"  "<<(double)(d_concensPtr[index]) * conc<<"\n";  
    } 
    random_file << "[aa bb] = size(data); " << std::endl;
    random_file << "x = unique(data(:,1)); " << std::endl;
    random_file << "y = unique(data(:,2)); " << std::endl;
    random_file << "z = unique(data(:,3));" << std::endl;
    random_file << "nx = length(x);" << std::endl;
    random_file << "ny = length(y);" << std::endl;
    random_file << "for zht = 1:length(z)    %% or, you can select the z-height at which you want concentration contours " << std::endl;
    random_file << "   cc=1;" << std::endl;
    random_file << "   conc_vector_zht=0;" << std::endl;
    random_file << "   for ii = 1:aa " << std::endl;
    random_file << "      if data(ii,3) == z(zht,:)" << std::endl;
    random_file << "         conc_vector_zht(cc,1) = data(ii,4);" << std::endl;
    random_file << "         cc=cc+1;" << std::endl;
    random_file << "      end" << std::endl;
    random_file << "   end" << std::endl;
    random_file << "   conc_matrix_zht=0; " << std::endl;
    random_file << "   conc_matrix_zht = reshape(conc_vector_zht,nx,ny)';" << std::endl;
    random_file << "   figure(zht)" << std::endl;
    random_file << "   h = pcolor(x,y,log10(conc_matrix_zht));" << std::endl;
    random_file << "   set(h,'edgecolor','none');" << std::endl;
    random_file << "   shading interp;" << std::endl;
    random_file << "   hh=colorbar;" << std::endl;
    random_file << "   set(get(hh,'ylabel'),'string','log10(Concentration)','fontsize',20);" << std::endl;
    random_file << "   set(gcf,'color','w');" << std::endl;
    random_file << "   set(gcf,'visible','off'); %%this is to make sure the image is not displayed" << std::endl;
    random_file << "   xlabel('$x$','interpreter','latex','fontsize',20,'color','k'); " << std::endl;
    random_file << "   ylabel('$y$','interpreter','latex','fontsize',20,'color','k');" << std::endl;
    random_file << "   caxis([-8 3.5]);" << std::endl;
    random_file << "   string = strcat('log10(Concentration) Contours; Horizontal x-y plane; Elevation z = ',num2str(z(zht,:)));" << std::endl;
    random_file << "   h=title(string,'fontsize',12);" << std::endl;
    random_file << "   axis equal;" << std::endl;
    random_file.close();
    exit(0);
  }
//   random_file.open ("cal_concentration.csv", std::ios::out);
//   uint total = 0;
//   for(int i=0; i<153*110*30; i++)
//   {
//     if(d_concensPtr[i]>0) 
//     {
//       random_file<<i<<"  "<<(d_concensPtr[i])<<"\n"; 
//       total += d_concensPtr[i];
//     }
//   }
//   std::cout<<"total"<<total<<"  numParticles"<<numParticles<<"\n";
//   random_file.close();
}

void compareHstDev(const thrust::host_vector<float4> &hData, const uint &size, const int &texname)
{ 
    thrust::device_vector<float4> devVec;
    thrust::device_vector<int> devIndex;
    devVec.resize(size); 
    devIndex.resize(size); 
    thrust::sequence(devIndex.begin(), devIndex.end());  
    
    thrust::for_each(  
      thrust::make_zip_iterator( thrust::make_tuple(devVec.begin(), devIndex.begin())),
      thrust::make_zip_iterator( thrust::make_tuple(devVec.end(), devIndex.end())),
      copyDeviceData_functor(texname)
    ); 
    
    for(int i=0; i<devIndex.size(); i++)
    { 
      if(hData[i]!=(float4)devVec[i])
      {
	std::cout<<"error data found!!"<<i<<" x "<< hData[i].x<< " y "<<hData[i].y<< " z "<< hData[i].z<<"\n";
	std::cout<<"                  "<<i<<" x "<< ((float4)devVec[i]).x<< " y "<<((float4)devVec[i]).y<< " z "<< ((float4)devVec[i]).z<<"\n";
      } 
    }
//   }
 }
  
 
void randDevTest(thrust::host_vector<float4> &hData, const uint &size)
{
   thrust::device_vector<float4> devVec;
   thrust::device_vector<int> devIndex;
   devVec.resize(size); 
   devIndex.resize(size); 
   thrust::sequence(devIndex.begin(), devIndex.end()); 
   
   thrust::for_each(  
     thrust::make_zip_iterator( thrust::make_tuple(devVec.begin(), devIndex.begin())),
     thrust::make_zip_iterator( thrust::make_tuple(devVec.end(), devIndex.end())),
     rand_box_muller_f4()
   ); 
   std::ofstream file;
   file.open ("testboxmullerf4.csv", std::ios::out);
   for(int i=0; i<devIndex.size(); i++) 
     file<<" "<<((float4)devVec[i]).x<<", "<<((float4)devVec[i]).y<<", "<<((float4)devVec[i]).z<<", "<<((float4)devVec[i]).w<<"\n"; 
}

void copyTurbsToDevice(const thrust::host_vector<turbulence> &hData)
{ 
}

void global_kernel(float *pos, float *winP, const uint &numParticles,
		   turbulence* d_turbs_ptr)
{    
//   thrust::device_vector<float4> devDebug_vector(numParticles);  
  thrust::device_ptr<float4> dev_pos4((float4 *)pos); 
  thrust::device_ptr<float4> dev_winP4((float4 *)winP);  
//   thrust::device_ptr<bool> dev_seed_flag_ptr((bool *)seeds_flag);   
  
//   uint numThreads, numBlocks;
  computeGridSize(numParticles, 64, numBlocks, numThreads);
  
//   turbulence* d_turbs_ptr =  thrust::raw_pointer_cast(dev_turbs_vector); 
//   test_kernel<<< numBlocks, numThreads >>>(thrust::raw_pointer_cast(dev_pos4),
// 					   thrust::raw_pointer_cast(dev_winP4),
// 					   numParticles,
// // 					  thrust::raw_pointer_cast(dev_seed_flag_ptr),
// // 					   thrust::raw_pointer_cast(&devDebug_vector[0]),
// 					   d_turbs_ptr); 
  
}

void global_kernel_debug(float *pos, float *winP, bool* seeds_flag, const uint &numParticles,
		   const thrust::host_vector<turbulence> &hData)
{   
  
  random_file.open ("random_gen_by_global.txt", std::ios::out);
  thrust::device_vector<turbulence> dev_turbs_vector = hData; 
 
  thrust::device_vector<float4> devDebug_vector(numParticles);  
  thrust::device_ptr<float4> d_pos4((float4 *)pos);
  thrust::device_ptr<float4> d_winP4((float4 *)winP); 
  thrust::device_ptr<bool> dev_seed_flag_ptr((bool *)seeds_flag);   
  
//   uint numThreads, numBlocks;
  computeGridSize(numParticles, 64, numBlocks, numThreads);
  
  turbulence* d_turbs_ptr =  thrust::raw_pointer_cast(&dev_turbs_vector[0]); 
//   test_kernel<<< numBlocks, numThreads >>>(thrust::raw_pointer_cast(d_pos4),
// 					  thrust::raw_pointer_cast(d_winP4),
// 					  thrust::raw_pointer_cast(dev_seed_flag_ptr),
// 					  thrust::raw_pointer_cast(&devDebug_vector[0]),
// 					  d_turbs_ptr);    

   
}


}   // extern "C" 