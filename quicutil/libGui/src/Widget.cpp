/* File: Widget.cpp
 * Author: Matthew Overby
 */

#include "Widget.h"

using namespace SLUI;

Widget::Widget(){

	m_borderColor = sf::Color(70,130,180);
	m_bgColor = sf::Color(255, 255, 255, 200);
	m_textColor = sf::Color(0,0,0);
	m_highlightColor = sf::Color(255, 153, 051, 200);
	m_paddingL = 10;
	m_paddingR = 10;
	m_paddingT = 10;
	m_paddingB = 10;
	m_App = 0;
}

Widget::~Widget(){

}

void Widget::changeColors(sf::Color border, sf::Color bg, sf::Color highlight, sf::Color text){

	m_borderColor = border;
	m_bgColor = bg;
	m_highlightColor = highlight;
	m_textColor = text;

}

bool Widget::isMouseOver(){

	const sf::Input &input = m_App->GetInput();
	float mouseX = input.GetMouseX();
	float mouseY = input.GetMouseY();

	if((mouseX < m_posX + m_width) && (m_posX < mouseX) && 
	(mouseY < m_posY + m_height) && (m_posY < mouseY)) {
		return true;
	}

	return false;
}

bool Widget::isMouseOver(int posX, int posY, int width, int height){

	const sf::Input &input = m_App->GetInput();
	float mouseX = input.GetMouseX();
	float mouseY = input.GetMouseY();

	if((mouseX < posX + width) && (posX < mouseX) && 
	(mouseY < posY + height) && (posY < mouseY)) {
		return true;
	}

	return false;
}

sf::Shape Widget::createBackground(float x, float y, float height, float width){

	// Rounded Rectangles by Astrof
	// http://en.sfml-dev.org/forums/index.php?topic=973.0

	sf::Shape newBg;
	newBg.SetOutlineWidth(2); 
	float radius = 7;

	float x2 = 0;
	float y2 = 0;

	for(int i=0; i<10; i++) { 
		x2 += radius/10; 
		y2 = sqrt(radius*radius - x2*x2); 
		newBg.AddPoint(x2+x+width-radius, y-y2+radius, m_bgColor, m_borderColor); 
	} 

	y2=0; 
	for(int i=0; i<10; i++) { 
		y2 += radius/10; 
		x2 = sqrt(radius*radius - y2*y2); 
		newBg.AddPoint(x+width+x2-radius, y+height-radius+y2, m_bgColor, m_borderColor); 
	} 

	x2=0; 
	for(int i=0; i<10; i++) { 
		x2 += radius/10; 
		y2 = sqrt(radius*radius - x2*x2); 
		newBg.AddPoint(x+radius-x2, y+height-radius+y2, m_bgColor, m_borderColor); 
	} 

	y2=0; 
	for(int i=0; i<10; i++) { 
		y2 += radius/10; 
		x2 = sqrt(radius*radius - y2*y2); 
		newBg.AddPoint(x-x2+radius, y+radius-y2, m_bgColor, m_borderColor); 
	} 

	return newBg; 
}

void Widget::setHighlight(bool val){

	if(val){ m_background.SetColor(m_highlightColor); }
	else{ m_background.SetColor(m_bgColor); }
}

/*
void Widget::setPosition( int posX, int posY ){

	m_posX = posX;
	m_posY = posY;
}
*/


