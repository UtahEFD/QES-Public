#include <iostream>
#include <netcdf>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "Advection.hpp"
#include "Args.hpp"
#include "Urb.hpp"
#include "Turb.hpp"
#include "Input.hpp"
#include "Output.hpp"
#include "PlumeInputData.hpp"

//#include <GL/glew.h>
//#if defined (_WIN32)
//#include <GL/wglew.h>
//#endif
//#if defined(__APPLE__) || defined(__MACOSX)
//#include <GLUT/glut.h>
//#else
//#include <GL/freeglut.h>
//#endif
//
//#include <rendercheck_gl.h>

#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <ctime>

//#include "GL_funs.hpp" 
//#include "plumeSystem.h"
//#include "paramgl.h"
#include "Eulerian.h"
#include "Dispersion.h"
//#include "Turb_cp.cu" 
 
//#include "util/handlePlumeArgs.h"
//#include "quicloader/QUICProject.h"

//#include <thrust/host_vector.h>
//
using namespace netCDF;
using namespace netCDF::exceptions;

//const uint width = 800, height = 600;
// 
//  
//bool bUseOpenGL = false;
//
//bool bUseGlobal = false;
//
//const char* output_file;

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
//extern "C" void cudaInit(int argc, char **argv);
//extern "C" void cudaGLInit(int argc, char **argv);
// extern "C" void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);
////////////////////////in kernel_interface.cu///////////////////////////////////////////
////
////void advectPar(const util&,Dispersion&,Eulerian&, const char*, const int);
////
////sivelab::QUICProject *data = 0; 
////Source source;   
////// initialize particle system
////// void initPlumeSystem(uint numParticles, uint3 gridSize, float4* &cellData)
////void initPlumeSystem(const uint &numParticles, const uint3 &gridSize, const util &utl, const char* output_file)
////{ 
////  Building building;
////  building.lowCorner = lowCorner; 
////  building.highCorner = highCorner;  
////  psystem = new PlumeSystem(numParticles, bUseOpenGL, bUseGlobal, building, domain, origin,
////			    source, utl, output_file);   
////} 
////
////// initialize OpenGL
////void initGL(int *argc, char **argv)
////{   
////    std::cout << "Calling initGL!" << std::endl;
////    
////  glutInit(argc, argv);
////  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
////  glutInitWindowSize(width, height);
////  glutCreateWindow("CUDA Plume");
////
////  glewInit();
////  if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
////    fprintf(stderr, "Required OpenGL extensions missing.");
////    exit(-1);
////  }
////   
/////////////////load  textures done start
////////path should include 
////////////////////skyBoxTex for skyBox Texture(need four ppm pics named by east,west,south, and north )
////////////////////buildingTex  for building Texture(only need two ppm pics named by buliding and roof )
////////////////////floorTex  for floor Texture(one ppm pic named by concrect.ppm here)
////  std::string path = "../img/";    
////  readTex(path);
//////  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
////  //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, GL_REPEAT);
//// // glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FITER,GL_LINEAR_MIPMAP_NEAREST);
////  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
////  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
////  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
/////////////////load  textures done! 
////  
////
////#if defined (_WIN32)
////  if (wglewIsSupported("WGL_EXT_swap_control")) {
////      // disable vertical sync
////    wglSwapIntervalEXT(0);
////  }
////#endif
////
////  glEnable(GL_DEPTH_TEST);
////  glClearColor(0.25, 0.25, 0.25, 1.0);
////
////  glutReportErrors();
////}
////
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
    
    // Create instance of Dispersion class
    Dispersion* dis = new Dispersion(urb,turb,eul,PID);
    
    // Create instance of Advection class
    Advection* adv = new Advection(urb,turb,eul,dis,PID);
    
    // compuet run time information
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    std::cout<<"[CUDA-Plume] \t Finished in "<<elapsed<<" seconds!"<<std::endl;
    std::cout<<"##############################################################"<<std::endl;
    
    exit(EXIT_SUCCESS);
    
    //Eulerian eul;
    //eul.createEul(utl); 
    //std::cout<<"                     Going to EUL read end: "<<((float)std::clock())/CLOCKS_PER_SEC -f_clock<<"\n"; 
  
//   std::cout<<"Going to Disp"<<std::endl;
 
  //Dispersion disp;
  //disp.createDisp(eul);    
  
//   advectPar(utl,disp,eul,argv[1],argc);
//   return 1;
  
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
 
