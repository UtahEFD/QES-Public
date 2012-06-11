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

#include "gl_funs.h"
#include "plumeSystem.h"
#include "Shader/Particle/render_particles.h"
#include "paramgl.h"
#include "util/handleQUICArgs.h"
#include "quicutil/QUICProject.h"
 
 
#define NUM_PARTICLES   pow(2,10)  
#define MAX(a,b) ((a > b) ? a : b)
 

const uint width = 800, height = 600;

// view params
int ox, oy;
bool keyDown[256];
int buttonState = 0;
float3 campos = make_float3(3.810398f, -9.522257f, -11.32808f); 
float camera_trans[] = {0, 0, 3};
float3 camera_rot   = make_float3(0.f, 0.f, 0.f);//{0, 0, 0};
float camera_trans_lag[] = {0, 0, 3};
float3 camera_rot_lag = make_float3(0.f, 0.f, 0.f);//{0, 0, 0};
const float inertia = .3f;
const float walkSpeed = 0.1f;
ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES;

//hard coding default values
Building building1;
float3 lowCorner = make_float3(3.f, -2.f, 3.f);
float3 highCorner = make_float3(8.f,  1.f, 8.f); 
extern GLuint buldingTex;
extern GLuint roofTex;

float3 domain = make_float3(40.f, 25.f, 26.f); 
float3 origin = make_float3(0.f, 0.f, 0.f); 
// float3 sourceOrigin = make_float3(10.0f, 12.5f, .5f);
Source source;
GLuint floorTex = 0; 



int mode = 0;
bool displayEnabled = true;
bool bPause = false;
bool displaySliders = false;  
int idleCounter = 0;
int demoCounter = 0;
float modelView[16];//for camera moving 


enum { M_VIEW = 0, M_MOVE };

extern uint numParticles;
uint3 gridSize;
int numIterations = 0; // run until exit

extern unsigned int timer;

// simulation parameters
float timestep = 0.5f;
float damping = 1.0f;
float gravity = 0.0003f;
int iterations = 1;
int ballr = 10;

// CheckFBO/BackBuffer class objects
extern CheckRender *g_CheckRender;
PlumeSystem *psystem = 0;
extern ParticleRenderer *renderer;  
bool bUseOpenGL = true;  

sivelab::QUICProject *data = 0;
  
extern "C" void computeFPS();
extern "C" GLuint loadTexture(char *filename);
extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);
extern "C" void drawBuildings(float3 lowCorner, float3 highCorner);



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

  cutilCheckError(cutCreateTimer(&timer));
}

void cleanup() 
{
  cutilCheckError( cutDeleteTimer( timer));
  
  if (buldingTex)  glDeleteTextures(1, &buldingTex);
  if (roofTex)  glDeleteTextures(1, &roofTex);
  if (floorTex)  glDeleteTextures(1, &floorTex);

  if (g_CheckRender) {
    delete g_CheckRender; g_CheckRender = NULL;
  }
}


// initialize OpenGL
void initGL(int *argc, char **argv)
{  
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowSize(width, height);
  glutCreateWindow("CUDA Particles");

  glewInit();
  if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
    fprintf(stderr, "Required OpenGL extensions missing.");
    exit(-1);
  }
  
// load  textures
  char* imagePath = cutFindFilePath("building.ppm", argv[0]);
  char* imagePath1 = cutFindFilePath("buildingRoof.ppm", argv[0]);
  char* imagePath2 = cutFindFilePath("concrete.ppm", argv[0]);
  if (imagePath == NULL) {
    fprintf(stderr, "Error finding floor image file\n"); 
  }
  buldingTex = loadTexture(imagePath);
  roofTex = loadTexture(imagePath1);
  floorTex = loadTexture(imagePath2);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, GL_REPEAT);

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
 


void drawFloor(float3 domainP, float3 originP)
{
  glPushMatrix(); 
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D,floorTex);
  glBegin(GL_QUADS); 
    glTexCoord2f(0.0f,0.0f);glVertex3f(originP.x, originP.y, originP.z);
    glTexCoord2f(0.0f,1.0f);glVertex3f(originP.x, domainP.y, originP.z);
    glTexCoord2f(1.0f,1.0f);glVertex3f(domainP.x, domainP.y, originP.z);
    glTexCoord2f(1.0f,0.0f);glVertex3f(domainP.x, originP.y, originP.z);
    
//     glTexCoord2f(0.0f,0.0f);glVertex3f(originP.x, originP.y, domainP.z);
//     glTexCoord2f(0.0f,1.0f);glVertex3f(originP.x, domainP.y, domainP.z);
//     glTexCoord2f(1.0f,1.0f);glVertex3f(domainP.x, domainP.y, domainP.z);
//     glTexCoord2f(1.0f,0.0f);glVertex3f(domainP.x, originP.y, domainP.z);
  glEnd();
  glPopMatrix(); 
}

void drawSphere()
{  
  glPushMatrix();
  glColor4f(1.0, 1.0, 0.0, .1f);
  glTranslatef(0.f, 0.f, 0.f);//(xpos,ypos,zpos);
  glutSolidSphere(1.f, 20, 16); 
  glPopMatrix(); 
}

void display()
{
  cutilCheckError(cutStartTimer(timer));   
  // update the simulation
  if (!bPause)
  {
    psystem->setIterations(iterations);
//     psystem->setDamping(damping);  
    psystem->update(timestep); 
    if (renderer) 
      renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
  }

  // render
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   

  // view transform 
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity(); 
  gluLookAt( 0, 25, 15,
	     0,  0,  0,
	     0,  0,  1);
  camera_rot_lag += (camera_rot - camera_rot_lag) * inertia;  
  glRotatef(camera_rot_lag.x, 1.0, 0.0, 0.0);
  glRotatef(camera_rot_lag.z, 0.0, 0.0, 1.0); 
  glTranslatef(campos.x, campos.y, campos.z); 
  glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
  
  lowCorner = make_float3(-100, -100, -100);
  highCorner = make_float3(140,  125, 125);
  drawBuildings(lowCorner, highCorner);
  drawFloor(domain, origin);
//   drawSphere();
  
  if (renderer && displayEnabled)
  {
    renderer->display(displayMode);
  }

  if (displaySliders) {
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
    glEnable(GL_BLEND); 
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
  }

  cutilCheckError(cutStopTimer(timer));   

  glutSwapBuffers();
  glutReportErrors();

  computeFPS();
//     glutPostRedisplay();
}


void reshape(int w, int h)
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (float) w / (float) h, 0.1, 100.0);

  glMatrixMode(GL_MODELVIEW);
  glViewport(0, 0, w, h);

  if (renderer) {
    renderer->setWindowSize(w, h);
    renderer->setFOV(60.0);
  }
}

void mouse(int button, int state, int x, int y)
{
  int mods;

  if (state == GLUT_DOWN)
    buttonState |= 1<<button;
  else if (state == GLUT_UP)
    buttonState = 0;

  mods = glutGetModifiers();
  if (mods & GLUT_ACTIVE_SHIFT) {
    buttonState = 2;
  } else if (mods & GLUT_ACTIVE_CTRL) {
    buttonState = 3;
  }

  ox = x; oy = y;
 
  idleCounter = 0;

  glutPostRedisplay();
}

// transfrom vector by matrix
void xform(float *v, float *r, GLfloat *m)
{
  r[0] = v[0]*m[0] + v[1]*m[4] + v[2]*m[8] + m[12];
  r[1] = v[0]*m[1] + v[1]*m[5] + v[2]*m[9] + m[13];
  r[2] = v[0]*m[2] + v[1]*m[6] + v[2]*m[10] + m[14];
} 

// transform vector by transpose of matrix
void ixform(float *v, float *r, GLfloat *m)
{
  r[0] = v[0]*m[0] + v[1]*m[1] + v[2]*m[2];
  r[1] = v[0]*m[4] + v[1]*m[5] + v[2]*m[6];
  r[2] = v[0]*m[8] + v[1]*m[9] + v[2]*m[10];
}

void ixformPoint(float *v, float *r, GLfloat *m)
{
  float x[4];
  x[0] = v[0] - m[12];
  x[1] = v[1] - m[13];
  x[2] = v[2] - m[14];
  x[3] = 1.0f;
  ixform(x, r, m);
}

void motion(int x, int y)
{
  float dx, dy;
  dx = (float)(x - ox);
  dy = (float)(y - oy);


  switch(mode) 
  {
  case M_VIEW:
    if (buttonState == 3) {
	// left+middle = zoom
      camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
    } 
    else if (buttonState & 2) {
	// middle = translate
      camera_trans[0] += dx / 100.0f;
      camera_trans[1] -= dy / 100.0f;
    } 
    else if (buttonState & 1) {
	// left = rotate
      camera_rot.z -= dx / 5.0f;
      camera_rot.x -= dy / 5.0f;
    }
    break;

  case M_MOVE:
    { 
    }
    break;
  }

  ox = x; oy = y;
 
  idleCounter = 0;

  glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
  switch (key) 
  { 
  case 13:
    psystem->update(timestep); 
    if (renderer)
	renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
    break;
  case '\033': 
    printf("campos: %f   %f   %f   \n", campos.x, campos.y, campos.z );
    exit(0);
    break;
  case 'v':
    mode = M_VIEW;
    break; 
  case 'm':
    mode = M_MOVE;
    break;
  case 'p':
    displayMode = ( ParticleRenderer::DisplayMode)
		  ((displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
    break;
  case 'z':
    psystem->dumpGrid();
    break;
  case 'u':
    psystem->dumpParticles(0, numParticles-1);//debugging
    break;

  case 'r':
    displayEnabled = !displayEnabled;
    break; 

  case 'h':
    displaySliders = !displaySliders;
    break;
  }
  keyDown[key] = true;
 
  idleCounter = 0;
  glutPostRedisplay();
}

void special(int k, int x, int y)
{
  
}

void idle(void)
{
  if (keyDown['w']) {
    // printf("adsfasdfasdf");
    campos.x += modelView[2] * walkSpeed;
    campos.y += modelView[6] * walkSpeed;
    campos.z += modelView[10] * walkSpeed;
  } else
  if (keyDown['s']) {
    campos.x -= modelView[2] * walkSpeed;
    campos.y -= modelView[6] * walkSpeed;
    campos.z -= modelView[10] * walkSpeed;
  } else
  if (keyDown['a']) {
    campos.x += modelView[0] * walkSpeed;
    campos.y += modelView[4] * walkSpeed;
    campos.z += modelView[8] * walkSpeed;
  } else
  if (keyDown['d']) {
    campos.x -= modelView[0] * walkSpeed;
    campos.y -= modelView[4] * walkSpeed;
    campos.z -= modelView[8] * walkSpeed;
  } else
  if (keyDown['e']) {
    campos.x += modelView[1] * walkSpeed;
    campos.y += modelView[5] * walkSpeed;
    campos.z += modelView[9] * walkSpeed;
  } else
  if (keyDown[' ']) {
    campos.x -= modelView[1] * walkSpeed;
    campos.y -= modelView[5] * walkSpeed;
    campos.z -= modelView[9] * walkSpeed;
  } 
  glutPostRedisplay();
}

void initParams()
{
}

void mainMenu(int i)
{
  key((unsigned char) i, 0, 0);
}

void initMenus()
{ 
}

void keyUp(unsigned char key, int /*x*/, int /*y*/)
{
  keyDown[key] = false;
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
	
	kk++;  
      }
    }
  } 
  QUICWindField.close();
  return;
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
  printf("sizeof constant memory: %d \n", sizeof(ConstParams));
  float3 localOrigin = make_float3(15.f, 12.5f, 0.f);
  float3 buildingwhl = make_float3(5.f, 5.f, 5.f);
  lowCorner = make_float3(localOrigin.x, localOrigin.y - (buildingwhl.x/2.f), localOrigin.z);
  highCorner = make_float3(localOrigin.x +buildingwhl.y, localOrigin.y + (buildingwhl.x/2.f), localOrigin.z+buildingwhl.z);
////\\\\\\\\\\\\\\\\\\\\building ??????????/////////////////  
  
  
  numParticles = NUM_PARTICLES; 
  numIterations = 0;  

  
/////////////////read files by args/////////////////////
  sivelab::QUICArgs quicArgs;
  quicArgs.process( argc, argv );
  // ./plume -q ../../../quicdata/SBUE_small_bldg/SBUE_small_bldg.proj 

  data = new sivelab::QUICProject( quicArgs.quicproj );
  gridSize = make_uint3(data->nx, data->ny, data->nz);//.x = gridSize.y = gridSize.z = gridDim; 
  std::cout << "Done loading QUIC data.\n" << std::endl;
//   std::vector<WindFieldDomainData> windFieldData( data->nx * data->ny * data->nz );
  float4 *windData = (float4 *)malloc(gridSize.x*gridSize.y*gridSize.z*sizeof(float4));
  float3 *sigData = (float3 *)malloc(gridSize.x*gridSize.y*gridSize.z*sizeof(float3));
  float3 *UData = (float3 *)malloc(gridSize.x*gridSize.y*gridSize.z*sizeof(float3));
  loadQUICWindField(data->nx, data->ny, data->nz, data->m_quicProjectPath, windData, sigData, UData); 

  source.type = SPHERESOURCE;
  if(source.type == SPHERESOURCE)
  {
    assert(source.type == SPHERESOURCE);
    float3 sourceOrigin = make_float3(10.0f, 12.5f, .5f);
    source.info.sph.ori = sourceOrigin;
    source.info.sph.rad = .5f;
  }else if( source.type == LINESOURCE)
  {
    assert(source.type == LINESOURCE);
  } else if( source.type == POINTSOURCE)
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
