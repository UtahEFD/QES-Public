/* File: FisheyeWidget.cpp
 * Author: Matthew Overby and Scot Halverson
 */

#include "FisheyeWidget.h"

using namespace SLUI;

FisheyeWidget::FisheyeWidget(EventTracker *evt, OptionTracker *opt, sf::RenderWindow* app){

	int h = app->GetHeight();
	int w = app->GetWidth();
	m_App = app;
	colors = 0;

	m_width = 120;
	m_height = 120;

	m_paddingR = m_width+15;
	m_paddingB = m_height+100;

//	m_posX = w-m_paddingR;
//	m_posY = h-m_paddingB;

	m_paddingL = 10;
	m_paddingT = 40;
	m_posX = m_paddingL;
	m_posY = m_paddingT;

	m_background = createBackground(m_posX, m_posY, m_height, m_width);

	m_optTracker = opt;
	m_eventTracker = evt;
}


FisheyeWidget::~FisheyeWidget(){

}


void FisheyeWidget::draw(){

	if( !m_optTracker->getActive("drawfisheye") ){ return; }

	m_App->Draw( m_background );
	m_App->Draw( m_sprite );
}


void FisheyeWidget::resizeEvent(){

	int h = m_App->GetHeight();
	int w = m_App->GetWidth();

//	int xMove = w-m_paddingR - m_posX;
//	m_posX += xMove;

//	int yMove = h-m_paddingB - m_posY;
//	m_posY += yMove;

	m_posX = m_paddingL;
	m_posY = m_paddingT;

	m_background = createBackground( m_posX, m_posY, m_height, m_width );
}


void FisheyeWidget::checkEvent(){

	switch(m_eventTracker->eventType){

		case EventType::W_Resize:{
			resizeEvent();
		} break;

		case -1:{
		} break;

	} // end switch eventType
}


void FisheyeWidget::setPosition( int posX, int posY ){

	m_posX = posX;
	m_posY = posY;
	m_background = createBackground(m_posX, m_posY, m_height, m_width);
}


void FisheyeWidget::setData( PixelMap newMap ){

	if( newMap.size() > 10000 ){
		std::cout << "**FisheyeWidget Error:  new PixelMap too large: " << 
		newMap.size() << std::endl;
		return;
	}

	int size = sqrt( newMap.size() );
	m_image = sf::Image( size, size );
	if( colors ){ delete colors; }
	colors = new sf::Color[newMap.size()];

	PixelMap::iterator it = newMap.begin();
	int i = 0;
	while( it != newMap.end() ){

		std::pair<int,int> pair = it->first;
		sf::Color newColor = sf::Color( it->second.x, it->second.y, it->second.z );	
		colors[i] = newColor;

		m_image.SetPixel( pair.first, pair.second, colors[i] );
		it++; i++;
	}

	m_sprite = sf::Sprite( m_image );
	m_sprite.SetPosition( m_posX+10, m_posY+10 );

}

