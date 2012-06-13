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
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <cutil_gl_inline.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h
#include <rendercheck_gl.h>
#include <cutil_math.h>

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include "GL_funs.h"
#include "plumeSystem.h"
#include "paramgl.h"
#include "util/handleQUICArgs.h"
#include "quicutil/QUICProject.h"
 
 
#define NUM_PARTICLES   1000000 //00  //pow(2,20)    
const uint width = 800, height = 600;
 
  
bool bUseOpenGL = true;  

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
extern "C" void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);
////////////////////////in kernel_interface.cu///////////////////////////////////////////

sivelab::QUICProject *data = 0; 
Source source;   

// initialize particle system
void initPlumeSystem(uint numParticles, uint3 gridSize, float4* &cellData)
{ 
  Building building;
  building.lowCorner = lowCorner; 
  building.highCorner = highCorner;  
  psystem = new PlumeSystem(numParticles, gridSize, bUseOpenGL, building, domain, origin,
			       source, cellData);  
  psystem->reset(); 

  if (bUseOpenGL) {
    renderer = new ParticleRenderer;
    renderer->setParticleRadius(psystem->getParticleRadius());
    renderer->setColorBuffer(psystem->getColorBuffer());
  }

//   cutilCheckError(cutCreateTimer(&timer));
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
  std::string path = "../Img/";    
  readTex(path);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, GL_REPEAT);
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
	int p2idx = k*nx*ny + i*nx + j; 
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
  
  int row;
  uint arrSize = nx*ny*nz; 
  uint width = sqrt(arrSize); 
  int numInRow = width/nx;//(width - (width % nx))/nx; 
  std::cout<<"\n\n\n\n\n\n\n\n\n\nwidth :"<< width<<"\n"<< numInRow<<"\n" ;
  while(kk < 40*25*25) 
  {
    
  }
  return;
  
  
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
    std::cout<<"\n";

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{   
  
///////////////////building ??????????/////////////////  
//   printf("sizeof constant memory: %d \n", sizeof(ConstParams));
  float3 localOrigin = make_float3(15.f, 12.5f, 0.f);
  float3 buildingwhl = make_float3(5.f, 5.f, 5.f); 
  lowCorner = make_float3(localOrigin.x, localOrigin.y - (buildingwhl.x/2.f), localOrigin.z);
  highCorner = make_float3(localOrigin.x +buildingwhl.y, localOrigin.y + (buildingwhl.x/2.f), localOrigin.z+buildingwhl.z);
////\\\\\\\\\\\\\\\\\\\\building ??????????/////////////////   
  
  std::cout<<"\n"<<"Initializing......          "<<"\n";

  
  numParticles = NUM_PARTICLES;   
  
/////////////////read files by args/////////////////////
  sivelab::QUICArgs quicArgs;
  quicArgs.process( argc, argv );
  // ./plume -q ../../../quicdata/SBUE_small_bldg/SBUE_small_bldg.proj 

  data = new sivelab::QUICProject( quicArgs.quicproj );
  uint3 gridSize = make_uint3(data->nx, data->ny, data->nz);//.x = gridSize.y = gridSize.z = gridDim; 
  std::cout << "Done loading QUIC data.\n" << std::endl;
//   std::vector<WindFieldDomainData> windFieldData( data->nx * data->ny * data->nz );
  float4 *windData = (float4 *)malloc(gridSize.x*gridSize.y*gridSize.z*sizeof(float4));
  float3 *sigData  = (float3 *)malloc(gridSize.x*gridSize.y*gridSize.z*sizeof(float3));
  float3 *UData    = (float3 *)malloc(gridSize.x*gridSize.y*gridSize.z*sizeof(float3));
  loadQUICWindField(data->nx, data->ny, data->nz, data->m_quicProjectPath, windData, sigData, UData); 

  source.type = LINESOURCE;
  if(source.type == SPHERESOURCE)
  {
    assert(source.type == SPHERESOURCE);
    float3 sourceOrigin = make_float3(10.0f, 12.5f, .5f);
    source.info.sph.ori = sourceOrigin;
    source.info.sph.rad = .5f;
  }
  else if( source.type == LINESOURCE)
  {
    assert(source.type == LINESOURCE); 
    source.info.ln.start = make_float3(6, 5.5, 0.5f);//10.0f, 12.5f, .5f);
    source.info.ln.end = make_float3(6, 20.5, 0.5f);//(6.0f, 13.5f, 5.5f);
  } 
  else if( source.type == POINTSOURCE)
  {
    assert(source.type == POINTSOURCE);
  } 
  source.speed = 0.5f;
//   return 1;
//   loadQUICWindField(data->nx, data->ny, data->nz, data->m_quicProjectPath, windFieldData);

  //
  // 3.  Build kernel to simply advect the particles...
  //
  // 4. Use time step in QPParams.h to determine the
  // loop... while (simulation duration is not yet complete) run
  // advection kernel again...
 
///////////////////////////////Opengl Main section////
  if (!bUseOpenGL) {
      cudaInit(argc, argv);
  } 
  else
  { 
    initGL(&argc, argv);
    cudaGLInit(argc, argv);

    initPlumeSystem(numParticles, gridSize, windData);
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

  cutilDeviceReset(); 
}
