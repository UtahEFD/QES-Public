#include "viewer.h"

#ifdef __APPLE__
  #include <OpenGL/gl.h>
  #include <OpenGL/glu.h>
#else 
  #include <GL/glu.h>
#endif

#include <cmath>

#include "matrix.h"

viewer::viewer(void) 
{
	eye = point( 0., 0.,-10.);
	foc = point( 0., 0., 0.);
	up  = normal( 0., 1., 0.);
}

viewer::viewer(viewer const&  v) 
{
	eye = v.eye;
	foc = v.foc;
	up  = v.up;
}

viewer::viewer(point const& l) 
{
	eye = l;
	foc = point( 0.,0.,0.);
	up  = normal(0.,1.,0.);
}

//May not be working
viewer::viewer(point const& l, point const& f) 
{
	eye = l;
	foc = f;
	up  = normal(0.,1.,0.);
}
        
viewer::viewer(point const& l, point const& f, normal const& u) 
{
	eye = l;
	foc = f;
	up  = u;
}

point viewer::getEye() const             {return eye;}
void  viewer::setEye  (point const& loc)  {eye = loc;}
void  viewer::shiftEye(point const& disp) {eye += disp;}

point viewer::getFocus() const             {return foc;}
void  viewer::setFocus  (point const& f)    {foc = f;}
void  viewer::shiftFocus(point const& disp) {foc += disp;}

normal viewer::getGaze() const        {return normal(foc - eye);}
void   viewer::setGaze(normal const& g) {foc = g * pow(foc * foc, .5);}

normal viewer::getRight() const      {return normal(up / foc);}
normal viewer::getUp()    const      {return up;}
void  viewer::setUp(normal const& n) {up = n;}


void viewer::lookRight(float const& deg) 
{
	foc = eye + makeRotation(up, deg) * (foc - eye);
}

void viewer::lookUp(float const& deg) 
{
	normal right(up / (foc - eye));
	foc = eye + makeRotation(right, deg) * (foc - eye);
	//up  =       makeRotation(right, deg) * up;
}

void viewer::rollRight(float const& deg) 
{
	up = makeRotation(foc - eye, deg) * up;
}

void viewer::lookLeft(float const& deg) {lookRight(-deg);}
void viewer::lookDown(float const& deg) {lookUp(-deg);}
void viewer::rollLeft(float const& deg) {rollRight(-deg);}

void viewer::moveForward(float const& disp) 
{
	normal gaze(foc - eye);
	eye += gaze * disp;
	foc += gaze * disp;
}

void viewer::strafeRight(float const& disp) 
{
	normal right(up / (foc - eye));
	eye += right * -disp;
	foc += right * -disp;
}

void viewer::strafeUp(float const& disp) 
{
	eye += up * disp;
	foc += up * disp;
}

void viewer::moveBackward(float const& disp) {moveForward(-disp);}
void viewer::strafeLeft(float const& disp)   {strafeRight(-disp);}
void viewer::strafeDown(float const& disp)   {strafeUp(-disp);}

void viewer::operator =(viewer const& v) 
{
	eye = v.getEye();
	foc = v.getFocus();
	up  = v.getUp();
}

void gluLookAt(point const& eye, point const& focus, normal const& up) 
{
  gluLookAt
  (
    eye.x,  eye.y,  eye.z, 
	  focus.x, focus.y, focus.z, 
	  up.getX(), up.getY(), up.getZ()
	);
}

