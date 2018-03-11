/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Driver for running urbViewer.
*/

#include <GL/glew.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "urbModule.h"
#include "urbViewer.h"

#include "../util/directory.h"


void initGL(void);
void initDM(void);
void display(void);
void idle(void);
void keyboard(unsigned char key,int x,int y);
void mouse(int button, int state, int x, int y);
void reshape(int width, int height);
void OpenGLText(int x, int y, char* s);

int window_id;
int win_width = 700, win_height = 700;


QUIC::urbViewer* umv = 0;

float inchScreenWidth, inchScreenHeight;
int   inchesFromScreen;

bool run = false;
char text_buffer[128];

std::string inp_dir;
std::string out_dir;
std::string shdr_dir;

int main( int argc, char *argv[] ) 
{
	getDirectories(argc, argv, inp_dir, out_dir);

	// fourth arg should be shader directory
	shdr_dir = "./";
	if(argc == 4)
	{
		shdr_dir.assign(argv[3]);
	}

  glutInitDisplayMode( GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE );

  glutInit(&argc, argv);
  glutInitWindowPosition(0, 0);
  glutInitWindowSize(win_width, win_height);

  window_id = glutCreateWindow(argv[0]);
  
  initGL();
  initDM();
  
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
  gluPerspective(45.,width / height,.5,100.);
}

void initDM(void) 
{
	std::cout << "Initializing urbViewer..." << std::endl;
 
	umv = new QUIC::urbViewer();
	umv->beQuiet(true);
 
  //QUIC::urbSetup::usingFortran(umv, inp_dir);

 	QUIC::urbParser::parse(umv, inp_dir);
 	QUIC::urbSetup::usingCPP(umv);

 	umv->initialize(); 	
 	umv->loadShaders(shdr_dir);

 
	umv->setEye(point(0.,20.,-12.5));
	umv->setFocus(point(0.,10.,0.));

	std::cout << "urbViewer initialization done." << std::endl;
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
	glClearColor(.8, .8, .8, .5);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
	if(umv->isFlat()) 
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
		gluPerspective(45.,width / height,.5,100.);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	  	gluLookAt(umv->getEye(), umv->getFocus(), umv->getUp());
	}

	umv->glDrawData();

  	sprintf(text_buffer, "Iterations  = %d", umv->getIteration());
  	OpenGLText(5, 65, text_buffer);

  	sprintf(text_buffer, "Tolerance = %f", umv->getEpsilon());
  	OpenGLText(5, 45, text_buffer);

  	sprintf(text_buffer, "Error         = %f", umv->getError());
  	OpenGLText(5, 25, text_buffer);

  	float ib = umv->getIntervalBound();
  	sprintf(text_buffer, "Interval     = [%f, %f]", -ib, ib);
  	OpenGLText(5, 5, text_buffer);
  	
 		sprintf(text_buffer, "Using Compression");
 		OpenGLText(5, glutGet(GLUT_WINDOW_HEIGHT) - 20, text_buffer);
  
  	glutSwapBuffers();
}

void idle(void) 
{
	if(run && !umv->isConvergedQ()) 
	{
		umv->nextIteration();
		display();
	}
	else 
	{
		run = false;
	}
}

void keyboard(unsigned char key,int x,int y) 
{
  if (key==27) // the escape key
  {
      glutDestroyWindow(window_id);
	  delete umv; umv = 0;
      exit(0);
  }
  
  switch(key) 
  {
   	// Movement
      case 'w': umv->moveForward (.5); break;
      case 'W': umv->moveForward (1.); break;
      case 's': umv->moveBackward(.5); break;
      case 'S': umv->moveBackward(1.); break;

      case 'd': umv->strafeRight (.5); break;
      //case 'D': umv->strafeRight (1.); break;
      case 'a': umv->strafeLeft  (.5); break;
      case 'A': umv->strafeLeft  (1.); break;
      
      case 'r': umv->strafeUp 	 (.5); break;
      case 'f': umv->strafeDown  (.5); break;

	// Look
      case 'q': umv->lookLeft    ( 5.); break;
      case 'Q': umv->lookLeft    (10.); break;
      case 'e': umv->lookRight   ( 5.); break;

		  case ',': umv->lookUp  (5.); break;
      case '.': umv->lookDown(5.); break;  

	// Data Manipulation
		  case 'R': umv->reset(); break;
      case 'F': umv->setFlat(!umv->isFlat()); break;

		  case 'D': std::cout << "Dumping iteration..." << std::flush;
				umv->dumpIteration(out_dir); 
				std::cout << "done." << std::endl;
			break;

      case 'E': umv->setEuler(!umv->isEuler()); break;
     
      case 'k': umv->showNextSlice(); break;
		  case 'l': umv->showPrevSlice(); break;
		  case 'K': while(umv->getSliceToShow() != -1) {umv->showNextSlice();} break;
 
      case '-': umv->setAlpha(umv->getAlpha() - .05); break;
      case '=': umv->setAlpha(umv->getAlpha() + .05); break;
      
		  case '{': umv->setSeperation(umv->getSeperation() - .1); break;
		  case '}': umv->setSeperation(umv->getSeperation() + .1); break;

		  case '(': umv->setShift(umv->getShift() - .1); break;
		  case ')': umv->setShift(umv->getShift() + .1); break;

		  case 'i': umv->nextIteration(); break;
		  case 'I': run = !run; break;
	  
      default: ; // do nothing
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
        umv->lookRight((downX - x) / 15.);
        umv->lookUp   ((y - downY) / 15.);
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
	/*
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
	*/
	glColor3ub(0, 0, 0);
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

