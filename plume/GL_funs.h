/*
 * gl_funs.h
 * This file is part of CUDAPLUME
 *
 * Copyright (C) 2012 - Alex
 *
 * CUDAPLUME is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * CUDAPLUME is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CUDAPLUME. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __GL_FUNS_H__
#define __GL_FUNS_H__


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
#include "Shader/Particle/render_particles.h"
#include "plumeSystem.h"

extern "C"
{
  GLuint buldingTex = 0;
  GLuint roofTex = 0;
  GLuint floorTex = 0; 
  enum BoxType{SKYBOX, BUILDING};
  struct BoxTex
  {
    GLuint e; GLuint w; GLuint n; GLuint s;
    GLuint u; GLuint d; 
    BoxType type;
  }skyBox, building;
   
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

  void readBoxTex(std::string path, BoxTex boxTex)
  {  
    path = "../Img/";  
      
    
    std::string imgPath = path;
    boxTex.type == BUILDING ? imgPath.append("building.ppm"):imgPath.append("east.ppm"); 
    boxTex.e = loadTexture((char*)imgPath.c_str());
    
    imgPath = path;
    imgPath = path.append("west.ppm"); 
    boxTex.type == BUILDING ? 1:boxTex.w = loadTexture((char*)imgPath.c_str());
    
    imgPath = path;
    imgPath = path.append("north.ppm"); 
    boxTex.type == BUILDING ? 1:boxTex.n = loadTexture((char*)imgPath.c_str());
    
    imgPath = path;
    imgPath = path.append("south.ppm"); 
    boxTex.type == BUILDING ? 1:boxTex.s = loadTexture((char*)imgPath.c_str());
    
    imgPath = path;
    boxTex.type == BUILDING ? imgPath.append("roof.ppm"):imgPath.append("up.ppm"); 
    boxTex.type == BUILDING ? 1:boxTex.u = loadTexture((char*)imgPath.c_str());
    
    imgPath = path;
    imgPath = path.append("down.ppm"); 
    boxTex.type == BUILDING ? 1:boxTex.d = loadTexture((char*)imgPath.c_str()); 
  }
  
  void drawScene()
  {
    
    
  }
    
  void drawBuildings(float3 lowCorner, float3 highCorner, bool bottom) 
  { 
    glPushMatrix(); 
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,buldingTex);
    glBegin(GL_QUADS);
      glTexCoord2f(0.0f,0.0f);glVertex3f(lowCorner.x,lowCorner.y,lowCorner.z);
      glTexCoord2f(0.0f,1.0f);glVertex3f(lowCorner.x,lowCorner.y,highCorner.z);
      glTexCoord2f(1.0f,1.0f);glVertex3f(highCorner.x,lowCorner.y,highCorner.z);
      glTexCoord2f(1.0f,0.0f);glVertex3f(highCorner.x,lowCorner.y,lowCorner.z);
      
      glTexCoord2f(0.0f,0.0f);glVertex3f(highCorner.x,lowCorner.y,lowCorner.z);
      glTexCoord2f(0.0f,1.0f);glVertex3f(highCorner.x,lowCorner.y,highCorner.z);
      glTexCoord2f(1.0f,1.0f);glVertex3f(highCorner.x,highCorner.y,highCorner.z);
      glTexCoord2f(1.0f,0.0f);glVertex3f(highCorner.x,highCorner.y,lowCorner.z);
	
      glTexCoord2f(0.0f,0.0f);glVertex3f(highCorner.x,highCorner.y,lowCorner.z);
      glTexCoord2f(0.0f,1.0f);glVertex3f(highCorner.x,highCorner.y,highCorner.z);
      glTexCoord2f(1.0f,1.0f);glVertex3f(lowCorner.x,highCorner.y,highCorner.z);
      glTexCoord2f(1.0f,0.0f);glVertex3f(lowCorner.x,highCorner.y,lowCorner.z);
      
      glTexCoord2f(0.0f,0.0f);glVertex3f(lowCorner.x,highCorner.y,lowCorner.z);
      glTexCoord2f(0.0f,1.0f);glVertex3f(lowCorner.x,highCorner.y,highCorner.z);
      glTexCoord2f(1.0f,1.0f);glVertex3f(lowCorner.x,lowCorner.y,highCorner.z);
      glTexCoord2f(1.0f,0.0f);glVertex3f(lowCorner.x,lowCorner.y,lowCorner.z);
    glEnd();   
    
    if(bottom)
    {
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D,roofTex);
      glBegin(GL_QUADS); 
	glTexCoord2f(0.0f,0.0f);glVertex3f(lowCorner.x,lowCorner.y,lowCorner.z);
	glTexCoord2f(0.0f,1.0f);glVertex3f(lowCorner.x,highCorner.y,lowCorner.z);
	glTexCoord2f(1.0f,1.0f);glVertex3f(highCorner.x,highCorner.y,lowCorner.z);
	glTexCoord2f(1.0f,0.0f);glVertex3f(highCorner.x,lowCorner.y,lowCorner.z);
      glEnd();
    }
    
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D,roofTex);
    glBegin(GL_QUADS); 
      glTexCoord2f(0.0f,0.0f);glVertex3f(lowCorner.x,lowCorner.y,highCorner.z);
      glTexCoord2f(0.0f,1.0f);glVertex3f(lowCorner.x,highCorner.y,highCorner.z);
      glTexCoord2f(1.0f,1.0f);glVertex3f(highCorner.x,highCorner.y,highCorner.z);
      glTexCoord2f(1.0f,0.0f);glVertex3f(highCorner.x,lowCorner.y,highCorner.z);
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


// fps
//   static int fpsCount = 0;
//   static int /*fpsLimit*/ = 1;
//   unsigned int timer;
//   unsigned int frameCount = 0;   
  uint numParticles = 0;
  
  CheckRender *g_CheckRender = NULL;	
  ParticleRenderer *renderer = 0;  
      
  void computeFPS()
  {
//     frameCount++;
//     fpsCount++; 
//     if (fpsCount == fpsLimit) {
//       char fps[256];
//       float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
//       sprintf(fps, "%s CUDA Plume (%d particles): %3.1f fps", 
// 	      ((g_CheckRender && g_CheckRender->IsQAReadback()) ? "AutoTest: " : ""), numParticles, ifps);  
// 
//       glutSetWindowTitle(fps);
//       fpsCount = 0; 
//       if (g_CheckRender && !g_CheckRender->IsQAReadback()) 
// 	fpsLimit = (int)MAX(ifps, 1.f);
// 
//       cutilCheckError(cutResetTimer(timer));   
//     }

	//update the delta time for animation
    static int lastFrameTime = 0; 
    if (lastFrameTime == 0) 
      lastFrameTime = glutGet(GLUT_ELAPSED_TIME); 

    int now = glutGet(GLUT_ELAPSED_TIME);
    int elapsedMilliseconds = now - lastFrameTime;
    float delta_t = elapsedMilliseconds / 1000.0f;
    lastFrameTime = now;
    char fps[256];
    sprintf(fps, "CUDA Plume (%d particles) %4.1f fps", numParticles, 1/delta_t);   
    glutSetWindowTitle(fps);
  }
  
  void reshape(int w, int h)
  {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 10000.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    if (renderer) {
      renderer->setWindowSize(w, h);
      renderer->setFOV(60.0);
    }
  }

  void cleanup() 
  {
//     cutilCheckError( cutDeleteTimer( timer));
    
    if (buldingTex)  glDeleteTextures(1, &buldingTex);
    if (roofTex)  glDeleteTextures(1, &roofTex);
    if (floorTex)  glDeleteTextures(1, &floorTex);

    if (g_CheckRender) {
      delete g_CheckRender; g_CheckRender = NULL;
    }
  } 
  
  int mode = 0;
  int ox, oy;
  bool keyDown[256];
  int buttonState = 0;
  float camera_trans[] = {0, 0, 3};
  float3 campos = make_float3(3.810398f, -9.522257f, -11.32808f); 
  float3 camera_rot_lag = make_float3(0.f, 0.f, 0.f);//{0, 0, 0};
  const float inertia = .3f;
  float3 camera_rot   = make_float3(0.f, 0.f, 0.f);
  float modelView[16];//for camera moving 
  enum { M_VIEW = 0, M_MOVE };
  
  PlumeSystem *psystem = 0;
  bool displayEnabled = true;
  bool displaySliders = false; 
  bool bPause = false;
  const float walkSpeed = 0.1f;
  ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES; 
  
  float3 lowCorner = make_float3(3.f, -2.f, 3.f);
  float3 highCorner = make_float3(8.f,  1.f, 8.f); 
  float3 domain = make_float3(40.f, 25.f, 26.f); 
  float3 origin = make_float3(0.f, 0.f, 0.f); 
    
  // simulation parameters
  float timestep = 0.5f;
  float damping = 1.0f;
  float gravity = 0.0003f;
  int iterations = 1;
  int ballr = 10;

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
  
//     idleCounter = 0;

    glutPostRedisplay();
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

    glutPostRedisplay();
  }

  
  // commented out to remove unused parameter warnings in Linux
  void key(unsigned char key, int /*x*/, int /*y*/)
  {
    switch (key) 
    { 
  //   case 13:
  //     psystem->update(timestep); 
  //     if (renderer)
  // 	renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
  //     break;
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
  
  //   idleCounter = 0;
    glutPostRedisplay();
  }

  void keyUp(unsigned char key, int /*x*/, int /*y*/)
  {
    keyDown[key] = false;
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
    
  void display()
  {
//     cutilCheckError(cutStartTimer(timer));   
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
    
    lowCorner = make_float3(-1130, -1130, -1130);
    highCorner = make_float3(1180,  1165, 1165);
    drawBuildings(lowCorner, highCorner, true);
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

//     cutilCheckError(cutStopTimer(timer));   

    glutSwapBuffers();
    glutReportErrors();

    computeFPS();
  //     glutPostRedisplay();
  }
   
  void special(int k, int x, int y)
  {
    
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
}
#endif /* __GL_FUNS_H__ */

