/*
* main.cpp
* This file is part of CUDAPLUME
*
* Copyright (C) 2012 - Alex, Pete
*
*
* CUDAPLUME is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with CUDAPLUME. If not, see <http://www.gnu.org/licenses/>.
*/


#include <GL/glew.h>
#if defined (_WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <helper_cuda.h>  
#include <helper_cuda_gl.h> 
#include <helper_math.h>

#include <rendercheck_gl.h>

#include <iostream> 
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <ctime>

#include "GL_funs.hpp" 
#include "plumeSystem.h"
#include "paramgl.h"
#include "bw/Eulerian.h"
#include "bw/Dispersion.h"
#include "Turb_cp.cu" 
 
#include "util/handlePlumeArgs.h"
#include "quicutil/QUICProject.h"
#include <thrust/host_vector.h>  
const uint width = 800, height = 600;
 
  
bool bUseOpenGL = true;

bool bUseGlobal = false;

const char* output_file;

//////////////in gl_funs.h////////////////////////////////////////////////////////////////
// extern "C" void key(unsigned char, int, int);
// extern "C" void display();
// extern "C" void idle(void);
// extern "C" void special(int k, int x, int y);
// extern "C" void mouse(int button, int state, int x, int y);
// extern "C" void reshape(int w, int h);  
// extern "C" void keyUp(unsigned char key, int /*x*/, int /*y*/);
// extern "C" void cleanup() ;
// extern "C" void computeFPS();
// extern "C" GLuint loadTexture(char *filename);
// extern "C" void drawBuildings(float3 lowCorner, float3 highCorner);
//////////////in gl_funs.h///////////////////////////////////////////////////////////////


////////////////////////in kernel_interface.cu///////////////////////////////////////////
extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
// extern "C" void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);
////////////////////////in kernel_interface.cu///////////////////////////////////////////

void advectPar(const util&,dispersion&,eulerian&, const char*, const int);

sivelab::QUICProject *data = 0; 
Source source;   
// initialize particle system
// void initPlumeSystem(uint numParticles, uint3 gridSize, float4* &cellData)
void initPlumeSystem(const uint &numParticles, const uint3 &gridSize, const util &utl, const char* output_file)
{ 
  Building building;
  building.lowCorner = lowCorner; 
  building.highCorner = highCorner;  
  psystem = new PlumeSystem(numParticles, bUseOpenGL, bUseGlobal, building, domain, origin,
			    source, utl, output_file);   
} 

// initialize OpenGL
void initGL(int *argc, char **argv)
{   
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowSize(width, height);
  glutCreateWindow("CUDA Plume");

  glewInit();
  if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
    fprintf(stderr, "Required OpenGL extensions missing.");
    exit(-1);
  }
   
/////////////load  textures done start
////path should include 
////////////////skyBoxTex for skyBox Texture(need four ppm pics named by east,west,south, and north )
////////////////buildingTex  for building Texture(only need two ppm pics named by buliding and roof )
////////////////floorTex  for floor Texture(one ppm pic named by concrect.ppm here)
  std::string path = "../img/";    
  readTex(path);
//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, GL_REPEAT);
 // glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FITER,GL_LINEAR_MIPMAP_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
/////////////load  textures done! 
  

#if defined (_WIN32)
  if (wglewIsSupported("WGL_EXT_swap_control")) {
      // disable vertical sync
    wglSwapIntervalEXT(0);
  }
#endif

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.25, 0.25, 0.25, 1.0);

  glutReportErrors();
}
   
void loadQUICWindField(int nx, int ny, int nz, const std::string &quicFilesPath, 
		       //std::vector<WindFieldDomainData>& windFieldData)
		       float4* &windData, float3* &sig, float3* &U)
{  
  // 
  // for now, this only loads the ascii files... binary will be
  // incorporated into quicutil
  //
  
  assert( quicFilesPath.c_str() != NULL );
  std::string path = quicFilesPath + "QU_velocity.dat";

  std::ifstream QUICWindField;
  QUICWindField.open(path.c_str()); //opening the wind file  to read

  if(!QUICWindField){
    std::cerr<<"Unable to open QUIC Windfield file : QU_velocity.dat ";
    exit(1);
  }

  std::string header;  // I am just using a very crude method to read the header of the wind file
  QUICWindField>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header;
  QUICWindField>>header>>header>>header>>header>>header;  

  std::cout << "Domain Size: "  << nx << " X "   << ny << " X "    << nz << std::endl;
  uint kk = 0; 
  
  for(int k = 0; k < nz; k++){   
    for(int i = 0; i < ny; i++){
      for(int j = 0; j < nx; j++){
// 	int p2idx = k*nx*ny + i*nx + j; taudz1
	QUICWindField >> header;//windFieldData[p2idx].x;
	QUICWindField >> header;//windFieldData[p2idx].y;
	QUICWindField >> header;//windFieldData[p2idx].z;  	
	QUICWindField >> windData[kk].x;//windFieldData[p2idx].u;//((Cell)(d_ptr[kk])).wind.x;//cell.wind.x;////h_ptr[p2idx].wind.x; 
	QUICWindField >> windData[kk].y;//windFieldData[p2idx].v;//d_ptr[kk].wind.x;//cell.wind.y;//h_ptr[p2idx].wind.y; 
	QUICWindField >> windData[kk].z;//windFieldData[p2idx].w;//d_ptr[kk].wind.x;//c/ell.wind.z;//h_ptr[p2idx].wind.z; 
	windData[kk].w = drand48();
	
	kk++;  
      }
    }
  } 
  QUICWindField.close();
  
//   int row;
  uint arrSize = nx*ny*nz; 
  uint width = sqrt(arrSize); 
  int numInRow = width/nx;//(width - (width % nx))/nx; 
  std::cout<<"\n\n\n\n\n\n\n\n\n\nwidth :"<< width<<"\n"<< numInRow<<"\n" ;
  while(kk < 40*25*25) 
  {
    
  }
  return;
  /*
  
///////////////////////////////
////read turbulence data <not sure about if data correct yet> 
////////////////////////////////
  path = quicFilesPath + "QP_turbfield.dat"; 
  std::ifstream QUICturbfield;
  QUICturbfield.open(path.c_str()); //opening the wind file  to read
  if(!QUICturbfield){
    std::cerr<<"Unable to open QUIC Windfield file : QU_velocity.dat ";
    exit(1);
  }
  kk = 0;
  while(kk<43) 
  {
    QUICturbfield>>header;  
    kk++;
  }
  kk=0;
  float a; 
  while(kk < 40*25*25) 
  {
    QUICturbfield>>header>>header>>header; 
//     std::cout<<header<<" ";
    QUICturbfield>>sig[kk].x>>sig[kk].y>>sig[kk].z;
    QUICturbfield>>header>>header>>header
		 >>header>>header>>header ; 
//       std::cout<<sig[kk].x<< " "<<sig[kk].y<< " "<<sig[kk].z<<"\n";
    kk++;
  }
  kk=0;
  while(kk < 40*25*25) 
  {
     std::cout<<sig[kk].y/sig[kk].x<< " "<<sig[kk].z/sig[kk].x<< "\n";
     kk++;
    
  }
    std::cout<<"\n";*/

} 
 
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{    
    std::string quicFile = "";

  util utl;
  utl.readInputFile(quicFile); 
  float f_clock = ((float)std::clock())/CLOCKS_PER_SEC;
  std::cout<<"                    Going to UTL read end: "<<f_clock<<"\n"; 
  
  eulerian eul;
  eul.createEul(utl); 
  std::cout<<"                     Going to EUL read end: "<<((float)std::clock())/CLOCKS_PER_SEC -f_clock<<"\n"; 
  
//   std::cout<<"Going to Disp"<<std::endl;
 
  dispersion disp;
  disp.createDisp(eul);    
  
//   advectPar(utl,disp,eul,argv[1],argc);
//   return 1;
  
  
/*here you give the name of output_file of concentration 
 */
  output_file = "test.m";
  
/*
 * only for device global memory  
 * bUseGlobal must set true, false is default
 */  
  thrust::host_vector<turbulence> turbs;
/*
 * only for texture memory, default memory
 */
    thrust::host_vector<float4> windData;
    thrust::host_vector<float3> prime(disp.prime.begin(), disp.prime.end()); 
    
    thrust::host_vector<int> cellType;
    thrust::host_vector<float> CoEps;
    thrust::host_vector<float4> eigVal; 
    thrust::host_vector<float4> ka0; 
    thrust::host_vector<float4> g2nd; 
  ////////////////  matrix 9////////////////
    thrust::host_vector<float4> eigVec1;
    thrust::host_vector<float4> eigVec2;
    thrust::host_vector<float4> eigVec3;
    thrust::host_vector<float4> eigVecInv1;
    thrust::host_vector<float4> eigVecInv2;
    thrust::host_vector<float4> eigVecInv3;
    thrust::host_vector<float4> lam1;
    thrust::host_vector<float4> lam2;
    thrust::host_vector<float4> lam3;
  //////////////// matrix6 ////////////////
    thrust::host_vector<float4> sig1;
    thrust::host_vector<float4> sig2;
    thrust::host_vector<float4> taudx1;
    thrust::host_vector<float4> taudx2; 
    thrust::host_vector<float4> taudy1;
    thrust::host_vector<float4> taudy2; 
    thrust::host_vector<float4> taudz1;
    thrust::host_vector<float4> taudz2;
  if(bUseGlobal)
  {
      turbs = thrust::host_vector<turbulence> (eul.CoEps.size()); 
      turb_cp_2ndEdition(utl, eul, disp, turbs); 
  } else
  {    
  //copy CoEps start//////////////////////////////////////////////////////////////
    CoEps = thrust::host_vector<float> (eul.CoEps.begin(), eul.CoEps.end());  
    eul.CoEps.clear();
    
    cellType = thrust::host_vector<int> (eul.CellType.size()); 
    for(int i=0; i<eul.CellType.size(); i++)
    {
      cellType[i] =  eul.CellType[i].c;
    } 
    eul.CellType.clear(); 
  //copy CoEps end//////////////////////////////////////////////////////////////
    
    
    turb_cp(utl, eul, disp, windData, eigVal, ka0, g2nd,
  ////////////////  matrix 9////////////////
	    eigVec1, eigVec2, eigVec3,
	    eigVecInv1, eigVecInv2, eigVecInv3,
	    lam1, lam2, lam3,
  //////////////// matrix6 ////////////////
	    sig1, sig2, taudx1, taudx2, taudy1, taudy2, taudz1, taudz2);  
  }
  
   
 /* 
  
  /////////////////read files by args/////////////////////
  sivelab::PlumeArgs quicArgs;
  quicArgs.process( argc, argv );
  // ./plume -q ../../../quicdata/SBUE_small_bldg/SBUE_small_bldg.proj 

  // Must supply an argument containing the .prof file to be read.
  std::string quicInputFile;
  if (quicArgs.isSet("quicproj", quicInputFile))
    {
      std::cout << "Will read input from files in: \"" << quicInputFile << "\"" << std::endl;
    }
  else 
    {
      std::cerr << "Must provide QUIC .proj file for opening. Exiting." << std::endl;
      quicArgs.printUsage();
      exit(EXIT_FAILURE);
    }

  
///////////////////building ??????????/////////////////  
//   printf("sizeof constant memory: %d \n", sizeof(ConstParams));
  float3 localOrigin = make_float3(15.f, 12.5f, 0.f);
  float3 buildingwhl = make_float3(5.f, 5.f, 5.f); 
  lowCorner = make_float3(localOrigin.x, localOrigin.y - (buildingwhl.x/2.f), localOrigin.z);
  highCorner = make_float3(localOrigin.x +buildingwhl.y, localOrigin.y + (buildingwhl.x/2.f), localOrigin.z+buildingwhl.z);
////\\\\\\\\\\\\\\\\\\\\building ??????????/////////////////   
  
  std::cout<<"\n"<<"Initializing......          "<<"\n";

  

  data = new sivelab::QUICProject( quicInputFile );


  // At this point, we will have read in the particle number into
  // QP_params.inp, so get from the QUICProject:
  numParticles = data->qpParamData.numParticles;
  std::cout << "Number of particles: " << numParticles << std::endl;


  uint3 gridSize = make_uint3(data->nx, data->ny, data->nz);//.x = gridSize.y = gridSize.z = gridDim; 
  std::cout << "Done loading QUIC data.\n" << std::endl;
//   std::vector<WindFieldDomainData> windFieldData( data->nx * data->ny * data->nz );
  float4 *windData = (float4 *)malloc(gridSize.x*gridSize.y*gridSize.z*sizeof(float4));
  float3 *sigData  = (float3 *)malloc(gridSize.x*gridSize.y*gridSize.z*sizeof(float3));
  float3 *UData    = (float3 *)malloc(gridSizpose.x*gridSize.y*gridSize.z*sizeof(float3)); 
  loadQUICWindField(data->nx, data->ny, data->nz, data->m_quicProjectPath, windData, sigData, UData); 
*/
  numParticles = 100000;
 
  source.type = POINTSOURCE;
  if(source.type == SPHERESOURCE)
  {
    assert(source.type == SPHERESOURCE);
    float3 sourceOrigin = make_float3(utl.xSrc, utl.ySrc, utl.zSrc);
    source.info.sph.ori = sourceOrigin;
    source.info.sph.rad = .5f;
  }
  else if( source.type == LINESOURCE)
  {
    assert(source.type == LINESOURCE); 
    source.info.ln.start = make_float3(43.66666667, 75.0, 0.666666667);//6, 8.5, 0.5f); 
    source.info.ln.end = make_float3(43.66666667, 25.0, 0.666666667);//(6, 17.5, 0.5f); 
  } 
  else if( source.type == POINTSOURCE)
  {
    assert(source.type == POINTSOURCE);
    source.info.pt.ori = make_float3(utl.xSrc, utl.ySrc, utl.zSrc);
  } 
  source.speed = 0.5f;
  
  domain = make_uint3(utl.nx, utl.ny, utl.nz); 
   
//   return 1;
//   loadQUICWindField(data->nx, data->ny, data->nz, data->m_quicProjectPath, windFieldData);

  //
  // 3.  Build kernel to simply advect the particles...
  //
  // 4. Use time step in QPParams.h to determine the
  // loop... while (simulation duration is not yet complete) run
  // advection kernel again... 
 
///////////////////////////////Opengl Main section////
  if (!bUseOpenGL) 
  { 
    cudaInit(argc, argv);
    initGL(&argc, argv);
    cudaInit(argc, argv);
    initPlumeSystem(numParticles, gridSize, utl, output_file);
    
    if(bUseGlobal)
      psystem->copy_turbs_2_deviceGlobal(turbs); 
    else
       psystem->_initDeviceTexture(CoEps, cellType, windData, eigVal, ka0, g2nd,
  ////////////////  matrix 9////////////////
	    eigVec1, eigVec2, eigVec3,
	    eigVecInv1, eigVecInv2, eigVecInv3,
	    lam1, lam2, lam3,
  //////////////// matrix6 ////////////////
	    sig1, sig2, taudx1, taudx2, taudy1, taudy2, taudz1, taudz2); 
//       psystem->update(timestep); 
    psystem->reset(source.info.pt.ori, prime); 
    bool b_print_concentration = true;
    while(true)   
      psystem->update(timestep, b_print_concentration); 
//     psystem->dev_par_concentration();
    return 1; 
  } 
  else
  { 
    initGL(&argc, argv);
    cudaGLInit(argc, argv);

    initPlumeSystem(numParticles, gridSize, utl, output_file);
    if(!bUseGlobal)
    {
      psystem->_initDeviceTexture(CoEps, cellType, windData, eigVal, ka0, g2nd,
  ////////////////  matrix 9////////////////
	    eigVec1, eigVec2, eigVec3,
	    eigVecInv1, eigVecInv2, eigVecInv3,
	    lam1, lam2, lam3,
  //////////////// matrix6 ////////////////
	    sig1, sig2, taudx1, taudx2, taudy1, taudy2, taudz1, taudz2); 
//       psystem->_initialize();
    }
    else
    { 
      psystem->copy_turbs_2_deviceGlobal(turbs);  
    }
     
      psystem->reset(source.info.pt.ori, prime);
      
      renderer = new ParticleRenderer;
      renderer->setParticleRadius(psystem->getParticleRadius());
      renderer->setColorBuffer(psystem->getColorBuffer()); 

      initParams();
    
      initMenus(); 
      glutDisplayFunc(display);
      glutReshapeFunc(reshape);
      glutMouseFunc(mouse);
      glutMotionFunc(motion);
      glutKeyboardFunc(key);
      glutKeyboardUpFunc(keyUp);
      glutSpecialFunc(special);
      glutIdleFunc(idle);

      atexit(cleanup);

      glutMainLoop();  
  }
  if (psystem)
    delete psystem;
  if (data)
    delete data;  

  // cutilDeviceReset(); 
}
 
