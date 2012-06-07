/*
 * main.cpp
 * This file is part of CUDAPLUME
 *
 * Copyright (C) 2012 - Alex
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

#include "particleSystem.h"
#include "Shader/Particle/render_particles.h"
#include "paramgl.h"
 
#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

const int binIdx = 0;   // choose the proper sReferenceBin
#define NUM_PARTICLES   pow(2,10) 
#define GRID_SIZE 64

// Define the files that are to be save and the reference images for validation 

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
const float walkSpeed = 0.02f;
ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES;

Building building1;
float3 lowCorner = make_float3(3.f, -2.f, 3.f);
float3 highCorner = make_float3(8.f,  1.f, 8.f); 
GLuint buldingTex = 0;
GLuint roofTex = 0;

float3 domain = make_float3(40.f, 25.f, 25.f); 
float3 origin = make_float3(0.f, 0.f, 0.f); 
float3 sourceOrigin = make_float3(10.0f, 12.5f, .5f);
GLuint floorTex = 0;

int mode = 0;
bool displayEnabled = true;
bool bPause = false;
bool displaySliders = false;
bool wireframe = false;
bool demoMode = false;
int idleCounter = 0;
int demoCounter = 0;
float modelView[16];//for camera moving 


enum { M_VIEW = 0, M_MOVE };

uint numParticles = 0;
uint3 gridSize;
int numIterations = 0; // run until exit

// simulation parameters
float timestep = 0.5f;
float damping = 1.0f;
float gravity = 0.0003f;
int iterations = 1;
int ballr = 10;
 

ParticleSystem *psystem = 0;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
unsigned int timer;

ParticleRenderer *renderer = 0;

// ParamListGL *params;

// Auto-Verification Code
const int frameCheckNumber = 4;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0; 
bool g_Verify = false;
bool g_bQAReadback = false;
bool g_bGLVerify   = false;
bool g_bFBODisplay = false; 
//int ii=0;

// CheckFBO/BackBuffer class objects
CheckRender       *g_CheckRender = NULL;

#define MAX(a,b) ((a > b) ? a : b)

const char *sSDKsample = "CUDA Particles Simulation";

extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void* host, const void* device, unsigned int vbo, int size);



// initialize particle system
void initParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL)
{ 
  Building building;
  building.lowCorner = lowCorner; 
  building.highCorner = highCorner;  
  psystem = new ParticleSystem(numParticles, gridSize, bUseOpenGL, building, domain, origin,
				sourceOrigin
  );  
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

 
GLuint createTexture(GLenum target, GLint internalformat, GLenum format, int w, int h, void *data)
{
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(target, tex);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(target, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_UNSIGNED_BYTE, data);
    return tex;
} 
 
GLuint loadTexture(char *filename)
{
    unsigned char *data = 0;
    unsigned int width, height;
    cutilCheckError( cutLoadPPM4ub(filename, &data, &width, &height));
    if (!data) {
        printf("Error opening file '%s'\n", filename);
        return 0;
    } 

    return createTexture(GL_TEXTURE_2D, GL_RGBA8, GL_RGBA, width, height, data);
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

void runBenchmark(int iterations)
{ 
    cutilDeviceSynchronize();
    cutilCheckError(cutStartTimer(timer));  
    for (int i = 0; i < iterations; ++i)
    {
        psystem->update(timestep);
    }
    cutilDeviceSynchronize();
    cutilCheckError(cutStopTimer(timer));  
    float fAvgSeconds = ((float)1.0e-3 * (float)cutGetTimerValue(timer))/(float)iterations;
 

    if (g_bQAReadback) { 
        float *hPos = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());
        copyArrayFromDevice(hPos, psystem->getCudaPosVBO(), 
                            0, sizeof(float)*4*psystem->getNumParticles()); 
    }
}

void AutoQATest()
{
    if (g_CheckRender && g_CheckRender->IsQAReadback()) {
        char temp[256];
        sprintf(temp, "AutoTest: Particles");
	    glutSetWindowTitle(temp);
		exit(0);
    }
}

void computeFPS()
{
    frameCount++;
    fpsCount++;
    if (fpsCount == fpsLimit-1) {
        g_Verify = true;
    }
    if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
        sprintf(fps, "%s CUDA Particles (%d particles): %3.1f fps", 
                ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "AutoTest: " : ""), numParticles, ifps);  

        glutSetWindowTitle(fps);
        fpsCount = 0; 
        if (g_CheckRender && !g_CheckRender->IsQAReadback()) fpsLimit = (int)MAX(ifps, 1.f);

        cutilCheckError(cutResetTimer(timer));  

        AutoQATest();
    }
}

void drawBuildings() 
{ 
    glPushMatrix(); 
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,buldingTex);
    glBegin(GL_QUADS);
      glTexCoord2f(0.0f,0.0f);glVertex3f(lowCorner.x,lowCorner.y,highCorner.z);
      glTexCoord2f(0.0f,1.0f);glVertex3f(lowCorner.x,highCorner.y,highCorner.z);
      glTexCoord2f(1.0f,1.0f);glVertex3f(highCorner.x,highCorner.y,highCorner.z);
      glTexCoord2f(1.0f,0.0f);glVertex3f(highCorner.x,lowCorner.y,highCorner.z);
      
      glTexCoord2f(0.0f,0.0f);glVertex3f(highCorner.x,lowCorner.y,highCorner.z);
      glTexCoord2f(0.0f,1.0f);glVertex3f(highCorner.x,highCorner.y,highCorner.z);
      glTexCoord2f(1.0f,1.0f);glVertex3f(highCorner.x,highCorner.y,lowCorner.z);
      glTexCoord2f(1.0f,0.0f);glVertex3f(highCorner.x,lowCorner.y,lowCorner.z);
       
      glTexCoord2f(0.0f,0.0f);glVertex3f(highCorner.x,lowCorner.y,lowCorner.z);
      glTexCoord2f(0.0f,1.0f);glVertex3f(highCorner.x,highCorner.y,lowCorner.z);
      glTexCoord2f(1.0f,1.0f);glVertex3f(lowCorner.x,highCorner.y,lowCorner.z);
      glTexCoord2f(1.0f,0.0f);glVertex3f(lowCorner.x,lowCorner.y,lowCorner.z);
      
      glTexCoord2f(0.0f,0.0f);glVertex3f(lowCorner.x,lowCorner.y,lowCorner.z);
      glTexCoord2f(0.0f,1.0f);glVertex3f(lowCorner.x,highCorner.y,lowCorner.z);
      glTexCoord2f(1.0f,1.0f);glVertex3f(lowCorner.x,highCorner.y,highCorner.z);
      glTexCoord2f(1.0f,0.0f);glVertex3f(lowCorner.x,lowCorner.y,highCorner.z);
    glEnd();  
    
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,roofTex);
    glBegin(GL_QUADS); 
      glTexCoord2f(0.0f,0.0f);glVertex3f(lowCorner.x,highCorner.y,highCorner.z);
      glTexCoord2f(0.0f,1.0f);glVertex3f(lowCorner.x,highCorner.y,lowCorner.z);
      glTexCoord2f(1.0f,1.0f);glVertex3f(highCorner.x,highCorner.y,lowCorner.z);
      glTexCoord2f(1.0f,0.0f);glVertex3f(highCorner.x,highCorner.y,highCorner.z);
    glEnd();
    glPopMatrix();  
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
    glEnd();
    glPopMatrix(); 
}
 
 void drawSphere()
 {
//       glDisable(GL_TEXTURE_RECTANGLE_ARB);
//   //glEnable(GL_LIGHTING);
//   //glEnable(GL_LIGHT0);
//    
//   glEnable(GL_COLOR_MATERIAL);
//   glEnable(GL_BLEND);
//   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 

  //glPointSize(2.0);
  
  glPushMatrix();
  glColor4f(1.0, 0.0, 0.0, .1f);
  glTranslatef(0.f, 0.f, 0.f);//(xpos,ypos,zpos);
  glutSolidSphere(1.25f, 20, 16); 
  glPopMatrix();
  
  glPushMatrix();
  glColor4f(.0, 1.0, 0.0, .1f);
  glTranslatef(20.f, 0.f, 0.f);//(xpos,ypos,zpos);
  glutSolidSphere(1.25f, 20, 16);
  glPopMatrix();
  
  glPushMatrix();
  glColor4f(.0, 0.0, 1.0, .1f);
  glTranslatef(20.f, 0.f, 20.f);//(xpos,ypos,zpos);
  //glTranslatef(-6.f, 0.f, 5.f);//(xpos,ypos,zpos);
  glutSolidSphere(1.25f, 20, 16); 
  glPopMatrix();
  
  glPushMatrix();
  glColor4f(1.0, 0.0, 0.0, .1f);
  glTranslatef(0.f, 20.f, 0.f);//(xpos,ypos,zpos);
  glutSolidSphere(1.25f, 20, 16); 
  glPopMatrix();
  
  glPushMatrix();
  glColor4f(.0, 1.0, 0.0, .1f);
  glTranslatef(20.f, 20.f, 0.f);//(xpos,ypos,zpos);
  glutSolidSphere(1.25f, 20, 16);
  glPopMatrix();
  
  glPushMatrix();
  glColor4f(.0, 0.0, 1.0, .1f);
  glTranslatef(20.f, 20.f, 20.f);//(xpos,ypos,zpos);
  //glTranslatef(-6.f, 0.f, 5.f);//(xpos,ypos,zpos);
  glutSolidSphere(1.25f, 20, 16); 
  glPopMatrix();
 
//   glDisable(GL_BLEND);
//   glDisable(GL_COLOR_MATERIAL);
//   //glDisable(GL_LIGHT0);*/
//   //glDisable(GL_LIGHTING);
//    glEnable(GL_TEXTURE_RECTANGLE_ARB); 
//    
 }
void display()
{
    cutilCheckError(cutStartTimer(timer));  
// printf("%d, ", ii++);
    // update the simulation
    if (!bPause)
    {
        psystem->setIterations(iterations);
        psystem->setDamping(damping);
//       psystem->setGravity(-gravity);
//         psystem->setCollideSpring(collideSpring);
//         psystem->setCollideDamping(collideDamping);
//         psystem->setCollideShear(collideShear);
//         psystem->setCollideAttraction(collideAttraction);

        psystem->update(timestep); 
        if (renderer) 
            renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
    }

    // render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   

    // view transform
   //glMatrixMode(GL_PROJECTION);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity(); 
    camera_rot_lag += (camera_rot - camera_rot_lag) * inertia;  
    glRotatef(camera_rot_lag.x, 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag.y, 0.0, 1.0, 0.0); 
    glTranslatef(campos.x, campos.y, campos.z); 
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    
    drawBuildings();
    drawFloor(domain, origin);
    drawSphere();
    
    if (renderer && displayEnabled)
    {
        renderer->display(displayMode);
    }

    if (displaySliders) {
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        glEnable(GL_BLEND);
//         params->Render(0, 0);
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

    demoMode = false;
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
            camera_rot.x += dy / 5.0f;
            camera_rot.y += dx / 5.0f;
        }
        break;

    case M_MOVE:
        { 
        }
        break;
    }

    ox = x; oy = y;

    demoMode = false;
    idleCounter = 0;

    glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key) 
    {
//     case ' ':
//     campos.y -= 0.1; 
//        // bPause = !bPause;
//         break;
    case 13:
        psystem->update(timestep); 
        if (renderer)
            renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
        break;
    case '\033':
   // case 'q':
  printf(" %f   %f   %f   \n", campos.x, campos.y, campos.z );
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

    demoMode = false;
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
    }
    if (keyDown['s']) {
        campos.x -= modelView[2] * walkSpeed;
        campos.y -= modelView[6] * walkSpeed;
        campos.z -= modelView[10] * walkSpeed;
    }
    if (keyDown['a']) {
        campos.x += modelView[0] * walkSpeed;
        campos.y += modelView[4] * walkSpeed;
        campos.z += modelView[8] * walkSpeed;
    }
    if (keyDown['d']) {
        campos.x -= modelView[0] * walkSpeed;
        campos.y -= modelView[4] * walkSpeed;
        campos.z -= modelView[8] * walkSpeed;
    }
    if (keyDown['e']) {
        campos.x += modelView[1] * walkSpeed;
        campos.y += modelView[5] * walkSpeed;
        campos.z += modelView[9] * walkSpeed;
    }
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


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv) 
{  
    
  printf("sizeof constant memory: %d \n", sizeof(ConstParams));
 // return 1; 
    //campos.x=0.200000, campos.y=0.000000, campos.z = -16.599997;  
    numParticles = NUM_PARTICLES; 
    gridSize = make_uint3(40, 25, 25);//.x = gridSize.y = gridSize.z = gridDim; 
    numIterations = 0;
    g_bQAReadback = false;

    if (argc > 1) {
//         cutGetCmdLineArgumenti( argc, (const char**) argv, "n", (int *) &numParticles);
//         cutGetCmdLineArgumenti( argc, (const char**) argv, "grid", (int *) &gridDim);
// 
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "qatest") ||
			cutCheckCmdLineFlag(argc, (const char **)argv, "noprompt")) 
		{
            g_bQAReadback = true;
            fpsLimit = frameCheckNumber;
            numIterations = 1;
        }
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "glverify"))
		{
            numIterations = 1;
            g_bGLVerify = true;
            fpsLimit = frameCheckNumber;
        }
    }

    gridSize = make_uint3(40, 25, 25);//.x = gridSize.y = gridSize.z = gridDim; 

    bool benchmark = cutCheckCmdLineFlag(argc, (const char**) argv, "benchmark") != 0;
    cutGetCmdLineArgumenti( argc, (const char**) argv, "i", &numIterations);

    if (g_bQAReadback) {
        cudaInit(argc, argv);
    } else {
		if ( cutCheckCmdLineFlag(argc, (const char **)argv, "device") ) { 
			printf("[%s]\n", argv[0]);
			printf("   Does not explicitly support -device=n in OpenGL mode\n");
			printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
			printf(" > %s -device=n -qatest\n", argv[0]);
			printf("exiting...\n"); 
		}

		initGL(&argc, argv);
        cudaGLInit(argc, argv);
    }


    initParticleSystem(numParticles, gridSize, !g_bQAReadback);
    initParams();

    if (!g_bQAReadback) 
        initMenus();
 

    if (benchmark || g_bQAReadback)
    {
        if (numIterations <= 0) 
            numIterations = 300;

        runBenchmark(numIterations);
    }
    else
    { 
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

    cutilDeviceReset(); 
}
