/* File: PrefWindow.cpp
 * Author: Matthew Overby
 */

#include "PrefWindow.h"

using namespace SLUI;

PrefWindow::PrefWindow(Window *win){}

PrefWindow::PrefWindow(int newId, WindowController *wnC, OptionTracker *opT, EventTracker *evt, sf::RenderWindow* app){

	int w = app->GetWidth();
	int h = app->GetHeight();
	id = newId;

	newKeyListening = false;
	m_paddingL = 50;
	m_paddingR = 50;
	m_paddingT = 50;
	m_paddingB = 50;
	m_posX = Margin/2;
	m_posY = Margin/2;
	m_height = h-Margin;
	m_width = Button::Width*2+m_paddingL*4;

	m_App = app;
	m_optTracker = opT;
	m_eventTracker = evt;
	m_winController = wnC;
	leftButtonIndex = m_posY+m_paddingT;
	rightButtonIndex = m_posY+m_paddingT;
	rightButtonPosX = m_posX+Button::Width+m_paddingL*2;
	leftButtonPosX = m_posX+m_paddingL;

	linesVisible = (m_height-m_paddingB*2-20)/(Button::Height+10);
	linesTotal = oldLinesTotal = std::max((leftButtonIndex-m_posY+m_paddingT)/(Button::Height+10), (rightButtonIndex-m_posY+m_paddingT)/(Button::Height+10));
	lineIndex = 0;

	closeButton = new StandardButton("Close", m_eventTracker, m_App);
	closeButton->setPosition( rightButtonPosX, m_posY+m_height-m_paddingB-Button::Height );

	applyButton = new StandardButton("Save Settings", m_eventTracker, m_App);
	applyButton->setPosition(leftButtonPosX, m_posY+m_height-m_paddingB-Button::Height);

	m_background = createBackground(m_posX, m_posY, m_height, m_width);

	scrollbar = new Scrollbar(m_posX+m_width-m_paddingR, m_posY+m_paddingT, (m_height-m_paddingB*2-20), m_App);
	scrollbar->makeSlider(linesTotal, linesVisible);
}

PrefWindow::~PrefWindow(){

	delete scrollbar;
	delete closeButton;
	delete applyButton;

	std::map< std::string, Button* >::iterator it1;
	for (it1 = leftButtons.begin(); it1 != leftButtons.end(); it1++) {
		Button *delPtr = it1->second;
		delete delPtr;
		leftButtons.erase( it1 );
	}

	std::map< std::string, Button* >::iterator it2;
	for (it2 = rightButtons.begin(); it2 != rightButtons.end(); it2++) {
		Button *delPtr = it2->second;
		delete delPtr;
		rightButtons.erase( it2 );
	}
}

void PrefWindow::highlight() {

	const sf::Input *input = &m_App->GetInput();

	closeButton->highlight();
	applyButton->highlight();
	scrollbar->highlight(input);

	std::map< std::string, Button* >::iterator it1;
	for (it1 = leftButtons.begin(); it1 != leftButtons.end(); it1++) {
		it1->second->highlight();
	}

	std::map< std::string, Button* >::iterator it2;
	for (it2 = rightButtons.begin(); it2 != rightButtons.end(); it2++) {
		it2->second->highlight();
	}
}

void PrefWindow::mouseClicked() {

	const sf::Input *input = &m_App->GetInput();
	float mouseX = m_App->GetInput().GetMouseX();
	float mouseY = m_App->GetInput().GetMouseY();
	newKeyListening = false;
	bool listButtonClicked = false;
	closeButton->checkEvent();
	applyButton->checkEvent();

	if(scrollbar->isMouseOverUp(input)) {
		scroll(true);
	}
	else if(scrollbar->isMouseOverDown(input)) {
		scroll(false);
	}

	if(closeButton->updated){
		closeButton->updated = false;
		close();
		m_eventTracker->callFunction( "updateconfig" );
	}
	else {
		std::map< std::string, Button* >::iterator it1;
		for (it1 = leftButtons.begin(); it1 != leftButtons.end(); it1++) {

			it1->second->checkEvent();

			if( it1->second->updated ){

				it1->second->updated = false;

				if (it1->second->type == ButtonType::Radio){
					m_optTracker->toggle(it1->first);
				}
				else if (it1->second->type == ButtonType::Value){
					float newVal = it1->second->getValue();
					m_optTracker->setValue(it1->first, newVal);
				}
				else if(it1->second->type == ButtonType::List){
					listButtonClicked = true;

					m_optTracker->setListValue(it1->first, it1->second->getDropSelected());
					chopButtons();
					if(!it1->second->active){
						linesTotal -= ( it1->second->getValue() );
						scrollbar->moveSliderToTop();
					}
					// I really should rework this function so I don't have to use a break
					break;
				}


			}
		}

		std::map< std::string, Button* >::iterator it2;
		for (it2 = rightButtons.begin(); it2 != rightButtons.end(); it2++) {
			it2->second->active = false;
			it2->second->checkEvent();
			if( it2->second->active ){
				newKeyListening = true;
				newKeyCommand = it2->first;
			}
		}

	}
	if(!listButtonClicked){
		if(applyButton->updated){
			applyButton->updated = false;
			m_eventTracker->showMessage("Settings Updated", 2);
			m_eventTracker->callFunction( "updateconfig" );
			m_eventTracker->callFunction( "updatevisuals" );
			//m_eventTracker->m_eventFlags["retrace"] = 1;
		}
	}

	linesTotal = oldLinesTotal;

	std::map< std::string, Button* >::iterator forLines;
	for (forLines = leftButtons.begin(); forLines != leftButtons.end(); forLines++) {
		if( forLines->second->type == ButtonType::List && forLines->second->active ){
			linesTotal += forLines->second->getValue();
		}
	}

	scrollbar->makeSlider(linesTotal, linesVisible);

}

void PrefWindow::scroll(bool up){

	if(up && lineIndex > 0){
		// scroll up = decrement index
		scrollbar->scroll(true);
		lineIndex--;
		chopButtons();
	}
	else if(!up && lineIndex < linesTotal-linesVisible){
		// scroll down = increase index
		scrollbar->scroll(false);
		lineIndex ++;
		chopButtons();
	}
}

void PrefWindow::setButtonBounds(std::string command, float minVal, float maxVal){

	std::map< std::string, Button* >::iterator it = leftButtons.find( command );
	if (it != leftButtons.end()) {
		it->second->setMinMax( minVal, maxVal );
	}
}

void PrefWindow::update(){

	// Options
	std::map<std::string, Option*> optionList = m_optTracker->getOptions();
	for(std::map<std::string, Option*>::iterator it = optionList.begin(); it != optionList.end(); it++){

		std::string command = it->first;

		std::map< std::string, Button* >::iterator buttonIt = leftButtons.find( command );

		if (buttonIt != leftButtons.end()) {
			if( buttonIt->second->type == ButtonType::Radio && it->second->type == OptionType::Bool){
				buttonIt->second->active = m_optTracker->getActive(command);
			}
			else if( buttonIt->second->type == ButtonType::Value && it->second->type == OptionType::Value){
				buttonIt->second->setValue( m_optTracker->getValue(command) );
			}
			else if( buttonIt->second->type == ButtonType::List && it->second->type == OptionType::List){
				buttonIt->second->setDropSelected( m_optTracker->getListValue(command) );
			}
		}
	}

	// Keys
	std::map<std::string, Keys*> keyList = m_optTracker->getKeys();
	for(std::map<std::string, Keys*>::iterator it = keyList.begin(); it != keyList.end(); it++){

		std::map< std::string, Button* >::iterator buttonId = rightButtons.find( it->first );
		if( buttonId != rightButtons.end() ){
			buttonId->second->setNewKey( m_optTracker->getKeyStr(it->first) );
		}
	}

}

void PrefWindow::addRadioButton(std::string command, std::string lab){

	Button *newButton = new RadioButton(lab, 0, m_eventTracker, m_App);
	newButton->setPosition(leftButtonPosX, leftButtonIndex);
	leftButtons[command] = newButton;

	leftButtonIndex += 40;
	linesTotal = oldLinesTotal = std::max((leftButtonIndex-m_posY+m_paddingT)/(Button::Height+10), (rightButtonIndex-m_posY+m_paddingT)/(Button::Height+10));
	scrollbar->makeSlider(linesTotal, linesVisible);
	chopButtons();
}

void PrefWindow::addValueButton(std::string command, std::string lab, float init){

	Button *newButton = new ValueButton(lab, init, m_eventTracker, m_App);
	newButton->setPosition(leftButtonPosX, leftButtonIndex);

	leftButtons[command] = newButton;

	leftButtonIndex += 40;
	linesTotal = oldLinesTotal = std::max((leftButtonIndex-m_posY+m_paddingT)/(Button::Height+10), (rightButtonIndex-m_posY+m_paddingT)/(Button::Height+10));
	scrollbar->makeSlider(linesTotal, linesVisible);
	chopButtons();
}

void PrefWindow::addKeyButton(std::string command, std::string lab, std::string key){

	Button *newButton = new KeyButton(lab, key, m_eventTracker, m_App);
	newButton->setPosition(rightButtonPosX, rightButtonIndex);
	rightButtons[command] = newButton;

	rightButtonIndex += 40;
	linesTotal = oldLinesTotal = std::max((leftButtonIndex-m_posY+m_paddingT)/(Button::Height+10), (rightButtonIndex-m_posY+m_paddingT)/(Button::Height+10));
	scrollbar->makeSlider(linesTotal, linesVisible);
	chopButtons();
}

void PrefWindow::addListButton(std::string command, std::string lab, std::vector<std::string> opts){

	Button *newButton = new ListButton(lab, m_eventTracker, m_App);
	newButton->setDropList(opts);
	newButton->setPosition(leftButtonPosX, leftButtonIndex);
	leftButtons[command] = newButton;

	leftButtonIndex += 40;
	linesTotal = oldLinesTotal = std::max((leftButtonIndex-m_posY+m_paddingT)/(Button::Height+10), (rightButtonIndex-m_posY+m_paddingT)/(Button::Height+10));
	scrollbar->makeSlider(linesTotal, linesVisible);
	chopButtons();
}

void PrefWindow::draw(){

	m_App->Draw(m_background);
	closeButton->draw();
	applyButton->draw();
	std::map< std::string, Button* >::iterator it1;

	for (it1 = leftButtons.begin(); it1 != leftButtons.end(); it1++) {
		it1->second->draw();
	}

	std::map< std::string, Button* >::iterator it2;
	for (it2 = rightButtons.begin(); it2 != rightButtons.end(); it2++) {
		it2->second->draw();
	}

	// DRAW LIST BUTTONS LAST 
	// ESPECIALLY IF THEY'RE ACTIVE
	for (it1 = leftButtons.begin(); it1 != leftButtons.end(); it1++) {
		if( it1->second->type == ButtonType::List && !it1->second->active ){ it1->second->draw(); }
	}
	for (it1 = leftButtons.begin(); it1 != leftButtons.end(); it1++) {
		if( it1->second->type == ButtonType::List && it1->second->active ){ it1->second->draw(); }
	}

	if(linesTotal > linesVisible) scrollbar->draw();
}

void PrefWindow::close(){

	std::map< std::string, Button* >::iterator it1;
	for (it1 = leftButtons.begin(); it1 != leftButtons.end(); it1++) {
		if( it1->second->type == ButtonType::Value || it1->second->type == ButtonType::Key ){
			it1->second->active = false;
		}
	}

	std::map< std::string, Button* >::iterator it2;
	for (it2 = rightButtons.begin(); it2 != rightButtons.end(); it2++) {
		if( it2->second->type == ButtonType::Value || it2->second->type == ButtonType::Key ){
			it2->second->active = false;
		}
	}

	m_winController->close(id);
	chopButtons();
	newKeyListening = false;
	linesTotal = oldLinesTotal;
	scrollbar->makeSlider(linesTotal, linesVisible);
	scrollbar->moveSliderToTop();
}

void PrefWindow::open(){

	m_winController->open(id);
}

void PrefWindow::resizeEvent(){

	int h = m_App->GetHeight();

	m_height = h-Margin;
	m_posX = Margin/2;
	m_posY = Margin/2;

	linesVisible = (m_height-m_paddingB*2-20)/40;
	linesTotal = std::max((leftButtonIndex-m_posY+m_paddingT)/40, (rightButtonIndex-m_posY+m_paddingT)/40);
	lineIndex = 0;

	closeButton->setPosition(rightButtonPosX, m_posY+m_height-m_paddingB);
	applyButton->setPosition(leftButtonPosX, m_posY+m_height-m_paddingB);

	m_background = createBackground(m_posX, m_posY, m_height, m_width);
	scrollbar->resizeEvent(m_posX+m_width-m_paddingR, m_posY+m_paddingT, (m_height-m_paddingB*2-20));
	scrollbar->makeSlider(linesTotal, linesVisible);

	chopButtons();
}

void PrefWindow::chopButtons(){

	float leftPosition = m_posY+m_paddingT;
	float rightPosition = m_posY+m_paddingT;

	std::map< std::string, Button* >::iterator it1;
	int currLine = 0;
	for (it1 = leftButtons.begin(); it1 != leftButtons.end(); it1++) {

		if(currLine >= lineIndex && currLine < lineIndex+linesVisible) {

			it1->second->setPosition(leftButtonPosX, leftPosition);

			if( (leftPosition+Button::Height) >= (applyButton->getPosition().y-10)){
				it1->second->setPosition(m_App->GetWidth(), m_App->GetHeight());
			}

			if(it1->second->type == ButtonType::List){

				if(currLine >= lineIndex+linesVisible-1){
					it1->second->setPosition(m_App->GetWidth(), m_App->GetHeight());
				}

				if( it1->second->active ){
					leftPosition += it1->second->getHeight() + 10;
				}
				else leftPosition += 40;

			}
			else leftPosition += 40;

		}
		else{
			// Move it to off the render window
			it1->second->setPosition(m_App->GetWidth(), m_App->GetHeight());
		}
		currLine++;
	}

	std::map< std::string, Button* >::iterator it2;
	currLine = 0;
	for (it2 = rightButtons.begin(); it2 != rightButtons.end(); it2++) {

		if(currLine >= lineIndex && currLine < lineIndex+linesVisible) {
			it2->second->setPosition(rightButtonPosX, rightPosition);
			rightPosition += 40;
		}
		else{
			// Move it to off the render window
			it2->second->setPosition(m_App->GetWidth(), m_App->GetHeight());
		}
		currLine++;
	}
}

void PrefWindow::checkEvent(){

	switch(m_eventTracker->eventType){

		case EventType::W_Resize:{
			resizeEvent();
		} break;

		case EventType::K_Press:{
			if(newKeyListening){

				std::map< std::string, Button* >::iterator it2;
				for (it2 = rightButtons.begin(); it2 != rightButtons.end(); it2++) {
					if( it2->second->type == ButtonType::Key) it2->second->checkEvent();
				}

				Keys tempKey;
				std::string desiredKey = tempKey.keyToString(m_eventTracker->lastEvent->Key.Code);
				if(desiredKey.compare("bad")!=0){
					if(m_optTracker->bind(newKeyCommand, desiredKey)){
						update();
					}
				}

				newKeyListening = false;
			}
		} break;

		case EventType::M_Dragleft:{
			std::map< std::string, Button* >::iterator it1;
			for (it1 = leftButtons.begin(); it1 != leftButtons.end(); it1++) {
				if( it1->second->type == ButtonType::Value) it1->second->checkEvent();
			}
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

		case -1:{
		} break;

	}
}

//int PrefWindow::getId(){
//	return id;
//}


