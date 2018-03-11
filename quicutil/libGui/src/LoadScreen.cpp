/* File: LoadScreen.cpp
 * Author: Matthew Overby
 */

#include "LoadScreen.h"

using namespace SLUI;

LoadScreen::LoadScreen(sf::RenderWindow* app){

	int h = app->GetHeight();
	int w = app->GetWidth();
	m_width = 200;
	m_height = 100;
	m_App = app;

	m_posX = w/2-m_width/2;
	m_posY = h/2-m_height/2;
	m_background = createBackground(m_posX, m_posY, m_height, m_width);

	loading.SetPosition(w/2-60, h/2-16);
	loading.SetColor(m_textColor);
	loading.SetSize(20);
	loading.SetText("Loading...");

}

void LoadScreen::draw(){

	m_App->Draw(m_background);
	m_App->Draw(loading);

}

void LoadScreen::resizeEvent(){

	int h = m_App->GetHeight();
	int w = m_App->GetWidth();

	m_posX = w/2-m_width/2;
	m_posY = h/2-m_height/2;
	m_background = createBackground(m_posX, m_posY, m_height, m_width);
	loading.SetPosition(w/2-60, h/2-18);

}



