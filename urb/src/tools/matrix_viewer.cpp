
#include <GL/glew.h>
#include <GL/glut.h>

#include "matrixViewer.h"

#include "../util/matrixIO.h"

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <cstdio>

/**
 * Dumps the memory on long file names with underscores. 
 * Don't know why, and not sure that I'm <that> interested
 * in finding out.
 */

void initGL(void);
void initMV(float*, float*, int, int, int);
void display(void);
void idle(void);
void keyboard(unsigned char key,int x,int y);
void mouse(int button, int state, int x, int y);
void reshape(int width, int height);
void OpenGLText(int x, int y, char* s);

int window_id;
int win_width = 700, win_height = 700;

point default_eye = point(0.0f,25.0f,-25.0f);
point default_focus = point(0.0f,10.0f,0.0f);
QUIC::matrixViewer* mv = 0;

float inchScreenWidth, inchScreenHeight;
int   inchesFromScreen;

char text_buffer[128];

int main( int argc, char *argv[] ) 
{
	std::cout << std::endl << "<===> Compare two matrices via OpenGL/CUDA <===>" << std::endl;
	//Arguments passed should be: filename, filename, nx, ny, nz.

	char* matFile1 = (char*) malloc(100 * sizeof(char));
	char* matFile2 = (char*) malloc(100 * sizeof(char));

	int nx, ny, nz;
	int nx_1, ny_1, nz_1;
	int nx_2, ny_2, nz_2;

	if(argc == 3) 
	{
		strcpy(matFile1, argv[1]);
		strcpy(matFile2, argv[2]);
	}

	else 
	{
		std::cout << "Please specify two files (that contain properly formatted data) to compare." << std::endl;
		exit(1);
	}

	// Test the file names.
	FILE* fileTester = fopen(matFile1, "r");
	if(fileTester == NULL) 
	{
		std::cout << "Unable to open " << matFile1 << ", exiting..." << std::endl;
		exit(1);
	}
	else {fclose(fileTester);}
	fileTester = fopen(matFile2, "r");
	if(fileTester == NULL) 
	{
		std::cout << "Unable to open " << matFile2 << ", exiting..." << std::endl;
		exit(1);
	}
	else {fclose(fileTester);}

	// Get things rolling.
		std::cout << "Loading matrices to host memory..." << std::flush;
	float* M_1 = inputMatrix(matFile1, &nx_1, &ny_1, &nz_1);
	float* M_2 = inputMatrix(matFile2, &nx_2, &ny_2, &nz_2);
		std::cout << "done." << std::endl;

	if(nx_1 != nx_2 || ny_1 != ny_2 || nz_1 != nz_2) 
	{
		std::cout << "Matrices for comparison do not have the same dimensions. Exiting...\n" << std::endl;
		exit(1);
	}
	else {nx = nx_1; ny = ny_1; nz = nz_1;}

		std::cout << "Comparing " << matFile1 << " and " << matFile2 << std::flush;
		std::cout << " with dimensions " << nx << "x" << ny << "x" << nz << "..." << std::endl;		


  glutInitDisplayMode( GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE );

  glutInit(&argc, argv);
  glutInitWindowPosition(0, 0);
  glutInitWindowSize(win_width, win_height);

	int len1 = strlen(matFile1);
	int len2 = strlen(matFile2);
	char* wndw_ttl = (char*) malloc((len1 + len2 + 5) * sizeof(char));
	sprintf(wndw_ttl, "%s and %s", matFile1, matFile2);
  window_id = glutCreateWindow(wndw_ttl);
  
  	initGL();
    initMV(M_1, M_2, nx, ny, nz);

	free(matFile1);
	free(matFile2);
	free(M_1);
	free(M_2);
	//free(wndw_ttl);

  glutDisplayFunc(display);
  glutIdleFunc(idle);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutReshapeFunc(reshape);

  glutMainLoop();
  return 0;
}

void initGL(void) 
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    glEnable(GL_DEPTH_TEST);
      
    glEnable(GL_TEXTURE_2D);
    
    int width  = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    glViewport(0, 0, width, height);
    gluPerspective(45.0f,width / height,.5f,100.0f);
}

void initMV(float* M_1, float* M_2, int nx, int ny, int nz) 
{
	mv = new QUIC::matrixViewer(M_1, M_2, nx, ny, nz);

	mv->setEye(default_eye);
	mv->setFocus(default_focus);
}

void reshape(int width, int height) 
{
    glutReshapeWindow(width,height);
    glViewport(0,0,width,height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void display(void) 
{
  glClearColor(0.3, 0.3, 0.3, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
	if(mv->isFlat()) 
	{
		glMatrixMode(GL_PROJECTION);
  		glLoadIdentity();
  		gluOrtho2D(0,1,0,1);
  		glMatrixMode(GL_MODELVIEW);
  		glLoadIdentity();
	}

	else 
	{
		glMatrixMode(GL_PROJECTION);
	  	glLoadIdentity();
		int width  = glutGet(GLUT_WINDOW_WIDTH);
		int height = glutGet(GLUT_WINDOW_HEIGHT);
		gluPerspective(45.0f,width / height,.5f,100.0f);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	  	gluLookAt(mv->getEye(), mv->getFocus(), mv->getUp());
	}

	mv->glDrawMatrixView();		

	if(!mv->areSeperate()) 
	{
		sprintf(text_buffer, "Threshold: %0.3f", mv->getThreshold());
  	OpenGLText(5,5,text_buffer);
	}
	else
	{
		float ib = mv->getIntervalBound();
		sprintf(text_buffer, "Values on [%f, %f]", -ib, ib);
	 	OpenGLText(5,5,text_buffer);
	}
  
  if(mv->isRelDiff()) 
  {
  	sprintf(text_buffer, "Displaying Relative Difference");
  	OpenGLText(5, glutGet(GLUT_WINDOW_HEIGHT) - 20, text_buffer);
  }
  else
  {
  	sprintf(text_buffer, "Displaying Absolute Difference");
  	OpenGLText(5, glutGet(GLUT_WINDOW_HEIGHT) - 20, text_buffer);
  }
  
  glutSwapBuffers();
}

void idle(void) {}

void keyboard(unsigned char key,int x,int y) 
{
  if (key==27) // the escape key
  {
	  delete mv; mv = 0;
      glutDestroyWindow(window_id);
      exit(0);
  }
  switch(key) 
  {
		//Movement
    case 'w': mv->moveForward (.5); break;
    case 'W': mv->moveForward (1.); break;
    case 's': mv->moveBackward(.5); break;
    case 'S': mv->moveBackward(1.); break;

    case 'd': mv->strafeRight (.5); break;
    case 'a': mv->strafeLeft  (.5); break;
    case 'A': mv->strafeLeft  (1.); break;
    
    case 'r': mv->strafeUp 	 (.5); break;
    case 'f': mv->strafeDown  (.5); break;

		//Look
	  case 'q': mv->lookLeft    ( 5.); break;
    case 'Q': mv->lookLeft    (10.); break;
    case 'e': mv->lookRight   ( 5.); break;
    case 'E': mv->lookRight   (10.); break;

	  case ',': mv->lookUp  (5.); break;
    case '.': mv->lookDown(5.); break;  

		//Views
	  case 'D': if(mv->areSeperate()) {mv->showDiff();}
				else				  {mv->showSeperate();}
				break;
	  case 'R': case 'p': mv->showRelDiff(!mv->isRelDiff()); break;	

    case 'F': mv->setFlat(!mv->isFlat()); break;
	  case 'L': mv->toggleSmoothTex(); break;
     
		//Threshold
	  case 't': mv->setThreshold(mv->getThreshold() + .005); break;
	  case 'T': mv->setThreshold(mv->getThreshold() + .05); break;
	  case 'y': mv->setThreshold(mv->getThreshold() - .005); break;
	  case 'Y': mv->setThreshold(mv->getThreshold() - .05); break;


		//Slices
    case 'k': mv->showNextSlice(); break;
	  case 'K': while(mv->getSliceToShow() != -1) {mv->showNextSlice();} break;
	  case 'l': mv->showPreviousSlice(); break;
      
		//Slice Movement
	  case '{': mv->setSeperation(mv->getSeperation() - 0.1f); break;
	  case '}': mv->setSeperation(mv->getSeperation() + 0.1f); break;

	  case '(': mv->setShift(mv->getShift() - 0.1f); break;
	  case ')': mv->setShift(mv->getShift() + 0.1f); break;

		case '-': mv->setAlpha(mv->getAlpha() - 0.05f); break;
    case '=': mv->setAlpha(mv->getAlpha() + 0.05f); break;
		
      default: /*do nothing*/;
  }
  glutPostRedisplay();
}

int downX = 0;
int downY = 0;
void mouse(int button, int state, int x, int y) 
{
    if(state == GLUT_DOWN && button == GLUT_LEFT_BUTTON) 
	{
        downX = x;
        downY = y;
    }
    if(state == GLUT_UP && button == GLUT_LEFT_BUTTON) 
	{
        mv->lookRight((downX - x) / 15.);
        mv->lookUp   ((y - downY) / 15.);
    }
    glutPostRedisplay();
}


void OpenGLText(int x, int y, char* s) 
{
	int lines;
	char* p;
	GLint* vp = new GLint[4];
	glGetIntegerv(GL_VIEWPORT,vp);

	// glDisable(GL_LIGHTING);
	// glDisable(texType);
	glDisable(GL_DEPTH_TEST);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, vp[2], 
	0, vp[3], -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glColor3ub(0, 0, 0);
	glRasterPos2i(x+1, y-1);
	for (p=s, lines=0; *p; p++) 
	{
		if (*p == '\n') 
		{
			lines++;
			glRasterPos2i(x+1, y-1-(lines*18));
		}
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
	}
	glColor3ub(255, 255, 0);
	glRasterPos2i(x, y);
	for (p=s, lines=0; *p; p++) 
	{
		if (*p == '\n') 
		{
			lines++;
			glRasterPos2i(x, y-(lines*18));
		}
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *p);
	}
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glEnable(GL_DEPTH_TEST);
	// glEnable(texType);
	// glEnable(GL_LIGHTING);
}

