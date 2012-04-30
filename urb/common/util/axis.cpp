#include "axis.h"

#include <GL/gl.h>
#include "OpenGLExtensions.h"
//#include <cmath>


axis::axis(float start, float finish, float increment, point loc, normal ori)
{
	strt = start;
	fnsh = finish;
	lngth = fabs(fnsh - strt);

	ncrmnt = fabs(increment);
	lctn = loc;
	rnttn = ori;
	r = g = b = 1.0f;
	a = 0.4f;
	
	point bump = point(1.0f, 1.0f, 1.0f);
	//point bump = point(rnttn.getX(), rnttn.getY(), rnttn.getZ());
	//if(bump.x < bump.y && bump.x < bump.z) {bump.x += 0.5f;}
	//else if(bump.y < bump.x && bump.y < bump.z) {bump.y += 0.5f;}
	//else {bump.z += 0.5f;}
	
	r1 = rnttn / bump;
	r2 = rnttn / r1;
}

axis::~axis()
{
	// do nothing so far
}

void axis::glDraw()
{
	//glDisable(GL_LIGHTING);
	glPushMatrix();
	glTranslatef(lctn);
	glColor4f(r, g, b, a);

	// Draw axis
	glBegin(GL_LINES);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(rnttn * lngth);
	glEnd();

	// Draw end ticks
	float end_ticklength = lngth / 100.0f;
	glBegin(GL_LINE_STRIP);
	glVertex3f( r1 * end_ticklength);
	glVertex3f( r2 * end_ticklength);
	glVertex3f(-r1 * end_ticklength);
	glVertex3f(-r2 * end_ticklength);
	glVertex3f( r1 * end_ticklength);
	glEnd();
	
	glBegin(GL_LINE_STRIP);
	glVertex3f( r1 * end_ticklength + rnttn * lngth);
	glVertex3f( r2 * end_ticklength + rnttn * lngth);
	glVertex3f(-r1 * end_ticklength + rnttn * lngth);
	glVertex3f(-r2 * end_ticklength + rnttn * lngth);
	glVertex3f( r1 * end_ticklength + rnttn * lngth);
	glEnd();

	// Draw ticks
	float tick_length = lngth / 200.0f;
	for(float i = 0.0f; i <= lngth; i += lngth / (ncrmnt + 1.0f))
	{
		glBegin(GL_LINE_STRIP);
		glVertex3f( r1 * tick_length + rnttn * i);
		glVertex3f( r2 * tick_length + rnttn * i);
		glVertex3f(-r1 * tick_length + rnttn * i);
		glVertex3f(-r2 * tick_length + rnttn * i);
		glVertex3f( r1 * tick_length + rnttn * i);
		glEnd();
	}

	glPopMatrix();
	//glEnable(GL_LIGHTING);
}

void axis::setColor(float r, float g, float b)
{
	this->setColor(r, g, b, a);
}

void axis::setColor(float _r, float _g, float _b, float _a)
{
	r = _r;
	g = _g;
	b = _b;
	a = _a;
}

void axis::setLength(float len)
{
	lngth = fabs(len);
}

void axis::setLocation(point loc)
{
	lctn = loc;
}

void axis::setOrientation(normal ori)
{
	rnttn = ori;
}

