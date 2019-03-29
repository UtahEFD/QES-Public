#include <iostream>
#include <netcdf>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "Args.hpp"
#include "Urb.hpp"
#include "Turb.hpp"
#include "Input.hpp"
#include "Output.hpp"
#include "PlumeInputData.hpp"

#include <GL/glew.h>
#if defined (_WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <rendercheck_gl.h>

#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <ctime>

#include "GL_funs.hpp" 
#include "plumeSystem.h"
#include "paramgl.h"
#include "Eulerian.h"
#include "Dispersion.h"
//#include "Turb_cp.cu" 
 
#include "util/handlePlumeArgs.h"
#include "quicloader/QUICProject.h"

#include <thrust/host_vector.h>

using namespace netCDF;
using namespace netCDF::exceptions;

const uint width = 800, height = 600;
 
  
bool bUseOpenGL = false;

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

void advectPar(const util&,dispersion&,Eulerian&, const char*, const int);

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
    std::cout << "Calling initGL!" << std::endl;
    
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

PlumeInputData* parseXMLTree(const std::string fileName);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
    // set up time information
    double elapsed;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // print a nice little welcome message
    std::cout << std::endl;
    std::cout<<"##############################################################"<<std::endl;
    std::cout<<"#                                                            #"<<std::endl;
    std::cout<<"#                   Welcome to CUDA-PLUME                    #"<<std::endl;
    std::cout<<"#                                                            #"<<std::endl;
    std::cout<<"##############################################################"<<std::endl;
    
    // parse command line arguments
    Args arguments;
    arguments.processArguments(argc, argv);
    
    // parse xml settings
    PlumeInputData* PID = parseXMLTree(arguments.quicFile);
    if ( !PID ) {
        std::cerr << "QUIC-Plume input file: " << arguments.quicFile << " not able to be read successfully." << std::endl;
        exit(EXIT_FAILURE);
    }
        
    // Create instance of cudaUrb input class
    Input* inputUrb = new Input(arguments.inputFileUrb);
    
    // Create instance of cudaTurb input class
    Input* inputTurb = new Input(arguments.inputFileTurb);

    // Create instance of output class
    Output* output = new Output(arguments.outputFile);
    
    // Create instance of cudaUrb class
    Urb* urb = new Urb(inputUrb);
    
    // Create instance of cudaTurb class
    Turb* turb = new Turb(inputTurb);
    
    // Create instance of Eulerian class
    Eulerian* eul = new Eulerian(urb,turb);
  
    //Eulerian eul;
    //eul.createEul(utl); 
    //std::cout<<"                     Going to EUL read end: "<<((float)std::clock())/CLOCKS_PER_SEC -f_clock<<"\n"; 
  
//   std::cout<<"Going to Disp"<<std::endl;
 
  //dispersion disp;
  //disp.createDisp(eul);    
  
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
///*
// * only for texture memory, default memory
// */
//    thrust::host_vector<float4> windData;
//    thrust::host_vector<float3> prime(disp.prime.begin(), disp.prime.end()); 
//    
//    thrust::host_vector<int> cellType;
//    thrust::host_vector<float> CoEps;
//    thrust::host_vector<float4> eigVal; 
//    thrust::host_vector<float4> ka0; 
//    thrust::host_vector<float4> g2nd; 
//  ////////////////  matrix 9////////////////
//    thrust::host_vector<float4> eigVec1;
//    thrust::host_vector<float4> eigVec2;
//    thrust::host_vector<float4> eigVec3;
//    thrust::host_vector<float4> eigVecInv1;
//    thrust::host_vector<float4> eigVecInv2;
//    thrust::host_vector<float4> eigVecInv3;
//    thrust::host_vector<float4> lam1;
//    thrust::host_vector<float4> lam2;
//    thrust::host_vector<float4> lam3;
//  //////////////// matrix6 ////////////////
//    thrust::host_vector<float4> sig1;
//    thrust::host_vector<float4> sig2;
//    thrust::host_vector<float4> taudx1;
//    thrust::host_vector<float4> taudx2; 
//    thrust::host_vector<float4> taudy1;
//    thrust::host_vector<float4> taudy2; 
//    thrust::host_vector<float4> taudz1;
//    thrust::host_vector<float4> taudz2;
//  if(bUseGlobal)
//  {
//      turbs = thrust::host_vector<turbulence> (eul.CoEps.size()); 
//      turb_cp_2ndEdition(utl, eul, disp, turbs); 
//  } else
//  {    
//  //copy CoEps start//////////////////////////////////////////////////////////////
//    CoEps = thrust::host_vector<float> (eul.CoEps.begin(), eul.CoEps.end());  
//    eul.CoEps.clear();
//    
//    cellType = thrust::host_vector<int> (eul.CellType.size()); 
//    for(int i=0; i<eul.CellType.size(); i++)
//    {
//      cellType[i] =  eul.CellType[i].c;
//    } 
//    eul.CellType.clear(); 
//  //copy CoEps end//////////////////////////////////////////////////////////////
//    
//    
//    turb_cp(utl, eul, disp, windData, eigVal, ka0, g2nd,
//  ////////////////  matrix 9////////////////
//	    eigVec1, eigVec2, eigVec3,
//	    eigVecInv1, eigVecInv2, eigVecInv3,
//	    lam1, lam2, lam3,
//  //////////////// matrix6 ////////////////
//	    sig1, sig2, taudx1, taudx2, taudy1, taudy2, taudz1, taudz2);  
//  }
//
//  // this all needs to be pulled from the appropriate QUIC files!
//
//  numParticles = 100000;
// 
//  source.type = POINTSOURCE;
//  if(source.type == SPHERESOURCE)
//  {
//    assert(source.type == SPHERESOURCE);
//    float3 sourceOrigin = make_float3(utl.xSrc, utl.ySrc, utl.zSrc);
//    source.info.sph.ori = sourceOrigin;
//    source.info.sph.rad = .5f;
//  }
//  else if( source.type == LINESOURCE)
//  {
//    assert(source.type == LINESOURCE); 
//    source.info.ln.start = make_float3(43.66666667, 75.0, 0.666666667);//6, 8.5, 0.5f); 
//    source.info.ln.end = make_float3(43.66666667, 25.0, 0.666666667);//(6, 17.5, 0.5f); 
//  } 
//  else if( source.type == POINTSOURCE)
//  {
//    assert(source.type == POINTSOURCE);
//    source.info.pt.ori = make_float3(utl.xSrc, utl.ySrc, utl.zSrc);
//  } 
//  source.speed = 0.5f;
//  
//  domain = make_uint3(utl.nx, utl.ny, utl.nz); 
//   
////   return 1;
////   loadQUICWindField(data->nx, data->ny, data->nz, data->m_quicProjectPath, windFieldData);
//
//  //
//  // 3.  Build kernel to simply advect the particles...
//  //
//  // 4. Use time step in QPParams.h to determine the
//  // loop... while (simulation duration is not yet complete) run
//  // advection kernel again... 
// 
/////////////////////////////////Opengl Main section////
//  if (!bUseOpenGL) 
//  { 
//      
//    cudaInit(argc, argv);
//    initGL(&argc, argv);
//    cudaInit(argc, argv);
//    initPlumeSystem(numParticles, gridSize, utl, output_file);
//    
//    if(bUseGlobal)
//      psystem->copy_turbs_2_deviceGlobal(turbs); 
//    else
//       psystem->_initDeviceTexture(CoEps, cellType, windData, eigVal, ka0, g2nd,
//  ////////////////  matrix 9////////////////
//	    eigVec1, eigVec2, eigVec3,
//	    eigVecInv1, eigVecInv2, eigVecInv3,
//	    lam1, lam2, lam3,
//  //////////////// matrix6 ////////////////
//	    sig1, sig2, taudx1, taudx2, taudy1, taudy2, taudz1, taudz2); 
////       psystem->update(timestep); 
//    psystem->reset(source.info.pt.ori, prime); 
//    bool b_print_concentration = true;
//    while(true)   
//      psystem->update(timestep, b_print_concentration); 
////     psystem->dev_par_concentration();
//    return 1; 
//  } 
//  else
//  { 
//      std::cout << "Not using OpenGL for rendering the visuals." << std::endl;
//      
//      glutInit(&argc, argv);
//      glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
//  
//      GLenum glewErr = glewInit();
//      if (GLEW_OK != glewErr) {
//          /* Problem: glewInit failed, something is seriously wrong. */
//          std::cerr << "Error: " << glewGetErrorString(glewErr) << std::endl;
//          exit(EXIT_FAILURE);
//      }
//      // initGL(&argc, argv);
//      cudaGLInit(argc, argv);
//
//    initPlumeSystem(numParticles, gridSize, utl, output_file);
//    if(!bUseGlobal)
//    {
//      psystem->_initDeviceTexture(CoEps, cellType, windData, eigVal, ka0, g2nd,
//  ////////////////  matrix 9////////////////
//	    eigVec1, eigVec2, eigVec3,
//	    eigVecInv1, eigVecInv2, eigVecInv3,
//	    lam1, lam2, lam3,
//  //////////////// matrix6 ////////////////
//	    sig1, sig2, taudx1, taudx2, taudy1, taudy2, taudz1, taudz2); 
////       psystem->_initialize();
//    }
//    else
//    { 
//      psystem->copy_turbs_2_deviceGlobal(turbs);  
//    }
//     
//      psystem->reset(source.info.pt.ori, prime);
//      
//      renderer = new ParticleRenderer;
//      renderer->setParticleRadius(psystem->getParticleRadius());
//      renderer->setColorBuffer(psystem->getColorBuffer()); 
//
//      initParams();
//    
//      initMenus(); 
//      glutDisplayFunc(display);
//      glutReshapeFunc(reshape);
//      glutMouseFunc(mouse);
//      glutMotionFunc(motion);
//      glutKeyboardFunc(key);
//      glutKeyboardUpFunc(keyUp);
//      glutSpecialFunc(special);
//      glutIdleFunc(idle);
//
//      atexit(cleanup);
//
//      glutMainLoop();  
//  }
//  if (psystem)
//    delete psystem;
//  if (data)
//    delete data;  
//
  // cutilDeviceReset(); 
}

PlumeInputData* parseXMLTree(const std::string fileName)
{
	pt::ptree tree;

	try
	{
		pt::read_xml(fileName, tree);
	}
	catch (boost::property_tree::xml_parser::xml_parser_error& e)
	{
		std::cerr << "Error reading tree in" << fileName << "\n";
		return (PlumeInputData*)0;
	}

	PlumeInputData* xmlRoot = new PlumeInputData();
        xmlRoot->parseTree( tree );
	return xmlRoot;
}
 
