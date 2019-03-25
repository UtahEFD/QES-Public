/* File: Scrollbar.cpp
 * Author: Matthew Overby
 */


#include "Scrollbar.h"

using namespace SLUI;

Scrollbar::Scrollbar() {}

Scrollbar::Scrollbar(int x, int y, int h, sf::RenderWindow* app){

	m_posX = x;
	m_posY = y;
	m_height = h;
	m_width = 15;
	m_App = app;
	m_bgColor = sf::Color(100, 100, 100, 100);
	sliderPosY = m_posY+15;
	sliderHeight = m_height-2*15;

	m_background = sf::Shape::Rectangle(m_posX, m_posY, m_posX+m_width, 
		m_posY+m_height, sf::Color(255, 255, 255, 255));
	m_background.SetColor(m_bgColor);

	upBg = sf::Shape::Rectangle(m_posX, m_posY, m_posX+m_width, 
		m_posY+m_width, sf::Color(255, 255, 255, 255));
	upBg.SetColor(sf::Color(120, 120, 120, 255));

	downBg = sf::Shape::Rectangle(m_posX, m_posY+m_height-m_width, m_posX+m_width, 
		m_posY+m_height, sf::Color(255, 255, 255, 255));
	downBg.SetColor(sf::Color(120, 120, 120, 255));

	slider = sf::Shape::Rectangle(m_posX, sliderPosY, m_posX+m_width, 
		sliderPosY+sliderHeight, sf::Color(255, 255, 255, 255));
	slider.SetColor(sf::Color(120, 120, 120, 255));

	upArrow.AddPoint(m_posX+7.5f, m_posY+2.5f, sf::Color::Black, sf::Color::Black);
	upArrow.AddPoint(m_posX+12, m_posY+11, sf::Color::Black, sf::Color::Black);
	upArrow.AddPoint(m_posX+3, m_posY+11, sf::Color::Black, sf::Color::Black);
	upArrow.SetOutlineWidth(1.f);

	downArrow.AddPoint(m_posX+3, m_posY+m_height-11, sf::Color::Black, sf::Color::Black);
	downArrow.AddPoint(m_posX+12, m_posY+m_height-11, sf::Color::Black, sf::Color::Black);
	downArrow.AddPoint(m_posX+7.5f, m_posY+m_height-2.5f, sf::Color::Black, sf::Color::Black);
	downArrow.SetOutlineWidth(1.f);
}

Scrollbar::~Scrollbar() {}

void Scrollbar::makeSlider(int newLinesTotal, int newLinesVisible){

	float sliderRatio = 1;
	linesTotal = newLinesTotal;
	linesVisible = newLinesVisible;
	sliderHeight = m_height-2*15;

	if(linesTotal > linesVisible){
		sliderRatio = (float)linesVisible/(float)linesTotal;
		sliderHeight = sliderRatio * sliderHeight;
	}

	slider = sf::Shape::Rectangle(m_posX, sliderPosY, m_posX+m_width, 
		sliderPosY+sliderHeight, sf::Color(255, 255, 255, 255));
	slider.SetColor(sf::Color(120, 120, 120, 255));
}

void Scrollbar::highlight(const sf::Input *input) {

	if(isMouseOverUp(input)) {
		upBg.SetColor(m_highlightColor);
	}
	else {
		upBg.SetColor(sf::Color(120, 120, 120, 100));
	}

	if(isMouseOverDown(input)) {
		downBg.SetColor(m_highlightColor);
	}
	else{
		downBg.SetColor(sf::Color(120, 120, 120, 100));
	}
}

bool Scrollbar::isMouseOverUp(const sf::Input *input){

	if(isMouseOver(m_posX, m_posY, 15, 15)) {
		return true;
	}
	return false;
}

bool Scrollbar::isMouseOverDown(const sf::Input *input){

	if(isMouseOver(m_posX, m_posY+m_height-m_width, 15, 15)) {
		return true;
	}
	return false;
}

bool Scrollbar::isMouseOverBar(){

	if(isMouseOver(m_posX, sliderPosY, 15, sliderHeight)) {
		return true;
	}
	return false;
}

void Scrollbar::scroll(bool up){

	if(linesTotal > linesVisible){

		float sliderRatio = 1/((float)linesVisible);

		if(sliderPosY+sliderHeight <= m_posY+m_height-15 && sliderPosY >= m_posY+15){

			float tempSliderPosY = sliderPosY;
			if(!up) tempSliderPosY += (sliderHeight*sliderRatio);
			else if(up) tempSliderPosY -= (sliderHeight*sliderRatio);

			if(tempSliderPosY+sliderHeight >= m_posY+m_height-15){
				sliderPosY = (m_posY+m_height-15)-sliderHeight;
			}
			else if(tempSliderPosY <= m_posY+15){
				sliderPosY = m_posY+15;
			}
			else{
				sliderPosY = tempSliderPosY;
			}

			slider = sf::Shape::Rectangle(m_posX, sliderPosY, m_posX+m_width, 
				sliderPosY+sliderHeight, sf::Color(255, 255, 255, 255));
			slider.SetColor(sf::Color(120, 120, 120, 255));
		}
	}
}


void Scrollbar::moveSliderToTop(){

	sliderPosY = m_posY+15;
}

void Scrollbar::resizeEvent(int x, int y, int h){

	m_posX = x;
	m_posY = y;
	m_height = h;

	sliderPosY = m_posY+15;
	sliderHeight = m_height-2*15;

	m_background = sf::Shape::Rectangle(m_posX, m_posY, m_posX+m_width, 
		m_posY+m_height, sf::Color(255, 255, 255, 255));
	m_background.SetColor(m_bgColor);

	upBg = sf::Shape::Rectangle(m_posX, m_posY, m_posX+m_width, 
		m_posY+m_width, sf::Color(255, 255, 255, 255));
	upBg.SetColor(sf::Color(120, 120, 120, 255));

	downBg = sf::Shape::Rectangle(m_posX, m_posY+m_height-m_width, m_posX+m_width, 
		m_posY+m_height, sf::Color(255, 255, 255, 255));
	downBg.SetColor(sf::Color(120, 120, 120, 255));

	slider = sf::Shape::Rectangle(m_posX, sliderPosY, m_posX+m_width, 
		sliderPosY+sliderHeight, sf::Color(255, 255, 255, 255));
	slider.SetColor(sf::Color(120, 120, 120, 255));

	upArrow = sf::Shape();
	upArrow.AddPoint(m_posX+7.5f, m_posY+2.5f, sf::Color::Black, sf::Color::Black);
	upArrow.AddPoint(m_posX+12, m_posY+11, sf::Color::Black, sf::Color::Black);
	upArrow.AddPoint(m_posX+3, m_posY+11, sf::Color::Black, sf::Color::Black);
	upArrow.SetOutlineWidth(1.f);

	downArrow = sf::Shape();
	downArrow.AddPoint(m_posX+3, m_posY+m_height-11, sf::Color::Black, sf::Color::Black);
	downArrow.AddPoint(m_posX+12, m_posY+m_height-11, sf::Color::Black, sf::Color::Black);
	downArrow.AddPoint(m_posX+7.5f, m_posY+m_height-2.5f, sf::Color::Black, sf::Color::Black);
	downArrow.SetOutlineWidth(1.f);
	
}

void Scrollbar::draw(){
	
	m_App->Draw(m_background);
	m_App->Draw(upBg);
	m_App->Draw(downBg);
	m_App->Draw(upArrow);
	m_App->Draw(downArrow);
	m_App->Draw(slider);
}


void Scrollbar::checkEvent(){

	//switch(m_eventTracker->eventType){
	switch(-1){

		case EventType::W_Resize:{
			//resizeEvent();
		} break;

		case EventType::M_Dragleft:{
		} break;

		case EventType::M_Scrollup:{
			//scroll(true);
		} break;

		case EventType::M_Scrolldown:{
			//scroll(false);
		} break;

		case EventType::M_Clickleft:{
			//mouseClicked();
		} break;

		case -1:{
		} break;
	}
}



