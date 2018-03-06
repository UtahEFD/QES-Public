/* File: TextWindow.cpp
 * Author: Matthew Overby
 */

#include "TextWindow.h"

using namespace SLUI;

TextWindow::TextWindow(int newId, WindowController *wnC, OptionTracker *opT, EventTracker *evt, sf::RenderWindow* app){

	int w = app->GetWidth();
	int h = app->GetHeight();
	id = newId;

	defaultColor = sf::Color(100, 100, 100);
	menuItemDown = sf::Color(205, 183, 158);

	m_App = app;
	m_optTracker = opT;
	m_eventTracker = evt;
	m_winController = wnC;
	m_height = h-Margin;
	m_width = w-Margin;
	m_posX = Margin/2;
	m_posY = Margin/2;

	closeButton = new StandardButton("Close", m_eventTracker, m_App);
	closeButton->setPosition( m_posX+m_width-Button::Width-20, m_posY+m_height-50 );

	m_background = createBackground(m_posX, m_posY, m_height, m_width);

	textBox = TextBox(m_width-40, m_height-95, m_posX+20, m_posY+20, app);

	sf::String textFormat;
	textFormat.SetSize(16);
	textFormat.SetColor(sf::Color::Black);
	textBox.setTextFormat(textFormat);
	
}

TextWindow::~TextWindow(){

	delete closeButton;
}

void TextWindow::loadText(std::string newFile){

	txtFile = newFile;
	textBox.loadText(newFile);
}

void TextWindow::highlight() {

	const sf::Input *input = &m_App->GetInput();
	closeButton->highlight();
	textBox.highlight(input);
}

void TextWindow::mouseClicked() {
	
	const sf::Input *input = &m_App->GetInput();
	closeButton->checkEvent();

	if(closeButton->updated){
		closeButton->updated = false;
		close();
	}

	textBox.mouseClicked(input);
}

void TextWindow::scroll(bool up){
	textBox.scroll(up);
}

void TextWindow::draw(){

	m_App->Draw(m_background);
	textBox.draw();
	closeButton->draw();
}

void TextWindow::close(){

	m_winController->close(id);
}

void TextWindow::open(){

	m_winController->open(id);
}

void TextWindow::resizeEvent(){

	m_height = m_App->GetHeight()-Margin;
	m_width = m_App->GetWidth()-Margin;
	m_posX = Margin/2;
	m_posY = Margin/2;

	closeButton->setPosition(m_posX+m_width-Button::Width-20, m_posY+m_height-50);

	m_background = createBackground(m_posX, m_posY, m_height, m_width);
	textBox = TextBox(m_width-40, m_height-95, m_posX+20, m_posY+20, m_App);

	sf::String textFormat;
	textFormat.SetSize(16);
	textFormat.SetColor(sf::Color::Black);
	textBox.setTextFormat(textFormat);

	if(txtFile.length() > 0) textBox.loadText(txtFile);
}

void TextWindow::checkEvent(){

	switch(m_eventTracker->eventType){

		case EventType::W_Resize:{
			resizeEvent();
		} break;

		case EventType::M_Scrollup:{
			scroll(true);
		} break;

		case EventType::M_Scrolldown:{
			scroll(false);
		} break;

		case EventType::M_Clickleft:{
			mouseClicked();
		} break;

		case EventType::M_Dragleft:{
			if(textBox.mouseOverScrollbar()){
				int diff = m_eventTracker->oldPos.y - m_eventTracker->newPos.y;
				//textBox.scroll(diff);
			}
		} break;

		case -1:{
		} break;

	}
}



