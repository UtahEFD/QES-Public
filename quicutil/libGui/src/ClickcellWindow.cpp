/* File: ClickcellWindow.cpp
 * Author: Matthew Overby
 */

#include "ClickcellWindow.h"

using namespace SLUI;

ClickcellWindow::ClickcellWindow(int newId, WindowController *wnC, CellTracker* clT, 
	OptionTracker *opT, EventTracker *evt, sf::RenderWindow* app){

	int w = app->GetWidth();
	int h = app->GetHeight();
	id = newId;

	defaultColor = sf::Color( 100, 100, 100 );
	menuItemDown = sf::Color( 205, 183, 158 );

	m_App = app;
	m_optTracker = opT;
	m_eventTracker = evt;
	m_winController = wnC;
	m_cellTracker = clT;
	m_height = h-160;
	m_width = w-160;
	m_posX = 160/2;
	m_posY = 160/2;

	textBox = TextBox(m_width-90-Button::Width*2, m_height-Button::Height-60, m_posX+20, m_posY+20, m_App);

	closeButton = new StandardButton("Close", m_eventTracker, m_App);
	closeButton->setPosition( m_posX+m_width-Button::Width-20, m_posY+m_height-50 );

	cellButton = new StandardButton("Graph Selected", m_eventTracker, m_App);
	cellButton->setPosition( m_posX+m_width-Button::Width-20, m_posY+20 );

	refreshButton = new StandardButton("Refresh", m_eventTracker, m_App);
	refreshButton->setPosition( m_posX+m_width-Button::Width-20, m_posY+30+Button::Height );

	saveButton = new StandardButton("Save to XML", m_eventTracker, m_App);
	saveButton->setPosition( m_posX+m_width-Button::Width-20, m_posY+40+Button::Height*2 );

	propButton = new ListButton("Set Material", m_eventTracker, m_App);
	propButton->setPosition( m_posX+m_width-Button::Width-20, m_posY+50+Button::Height*3 );

	showButton = new ListButton("Show Cell", m_eventTracker, m_App);
	showButton->setPosition( m_posX+m_width-Button::Width*2-40, m_posY+20 );

	std::vector<std::string> empty;
	empty.push_back("Last Selected");
	showButton->setDropList(empty);

	std::vector<std::string> patchSettings;
	patchSettings.push_back("Red Brick");
	patchSettings.push_back("Concrete");
	patchSettings.push_back("Glass");
	patchSettings.push_back("Wood");
	patchSettings.push_back("Gravel");
	patchSettings.push_back("Sand");
	patchSettings.push_back("Grass");
	patchSettings.push_back("Soil");
	patchSettings.push_back("White Pigment");
	patchSettings.push_back("Tar Paper");
	patchSettings.push_back("Black Body");
	propButton->setDropList(patchSettings);

	m_background = createBackground(m_posX, m_posY, m_height, m_width);
	
}

ClickcellWindow::~ClickcellWindow(){

	delete closeButton;
	delete cellButton;
	delete refreshButton;
	delete showButton;
	delete propButton;
	delete saveButton;
}

void ClickcellWindow::highlight() {

	const sf::Input *input = &m_App->GetInput();
	closeButton->highlight();
	cellButton->highlight();
	refreshButton->highlight();
	showButton->highlight();
	propButton->highlight();
	saveButton->highlight();
}

void ClickcellWindow::mouseClicked() {
	
	const sf::Input *input = &m_App->GetInput();
	closeButton->checkEvent();
	cellButton->checkEvent();
	refreshButton->checkEvent();
	showButton->checkEvent();
	propButton->checkEvent();
	saveButton->checkEvent();


	if(closeButton->updated){
		closeButton->updated = false;
		close();
	}
	else if(cellButton->updated){
		cellButton->updated = false;
		m_eventTracker->callFunction( "creategraph" );
	}
	else if(refreshButton->updated){
		refreshButton->updated = false;
		m_eventTracker->callFunction( "updatepatches" );
		update();
	}
	else if(saveButton->updated){
		saveButton->updated = false;
		m_eventTracker->callFunction( "updatepatches" );
		update();
		m_eventTracker->callFunction( "savepatchsettings" );
	}
	else if(propButton->updated){
		propButton->updated = false;
		std::string currentChoice = propButton->getDropSelected();
		m_optTracker->setListValue( "temp_patchMaterial", currentChoice );
	}
	else if(showButton->updated){
		showButton->updated = false;

		std::string currentChoice = showButton->getDropSelected();
		if(currentChoice == "Last Selected"){
			update();
		}
		else{
			std::vector<unsigned int> selected = m_cellTracker->getSelected();
			for(int i=0; i<selected.size(); i++){
				std::stringstream name("");
				name << "Patch ID: " << selected.at(i);
				if(currentChoice.compare( name.str() )==0){
					textBox.setText( m_cellTracker->getStats( selected.at(i) ));
				}
			}
		} // end another selected
	}
}

void ClickcellWindow::draw(){

	m_App->Draw(m_background);
	closeButton->draw();
	cellButton->draw();
	refreshButton->draw();
	showButton->draw();
	propButton->draw();
	saveButton->draw();
	textBox.draw();
}

void ClickcellWindow::update(){

	std::vector<unsigned int> selected = m_cellTracker->getSelected();
	if( selected.size() > 0 ){
		textBox.setText( m_cellTracker->getStats( m_cellTracker->getSelected().back() ) );
		std::vector<std::string> selections;
		selections.push_back("Last Selected");

		for( int i=0; i<selected.size(); i++){
			std::stringstream name("");
			name << "Patch ID: " << selected.at(i);
			selections.push_back(name.str());
		}

		showButton->setDropList(selections);
		showButton->setDropSelected("Last Selected");
	}
	else{
		textBox.setText( "No Cells Selected" );
		std::vector<std::string> selections;
		selections.push_back("Last Selected");
		showButton->setDropList(selections);
	}
}

void ClickcellWindow::close(){

	propButton->clearSelected();
	m_winController->close(id);
}

void ClickcellWindow::open(){

	m_winController->open(id);
}

void ClickcellWindow::resizeEvent(){

	m_height = m_App->GetHeight()-160;
	m_width = m_App->GetWidth()-160;
	m_posX = 160/2;
	m_posY = 160/2;

	textBox = TextBox(m_width-90-Button::Width*2, m_height-Button::Height-60, m_posX+20, m_posY+20, m_App);

	closeButton->setPosition(m_posX+m_width-Button::Width-20, m_posY+m_height-50);
	cellButton->setPosition(m_posX+m_width-Button::Width-20, m_posY+20);
	refreshButton->setPosition(m_posX+m_width-Button::Width-20, m_posY+30+Button::Height);
	saveButton->setPosition(m_posX+m_width-Button::Width-20, m_posY+40+Button::Height*2);
	propButton->setPosition( m_posX+m_width-Button::Width-20, m_posY+50+Button::Height*3 );
	showButton->setPosition( m_posX+m_width-Button::Width*2-40, m_posY+20 );

	m_background = createBackground(m_posX, m_posY, m_height, m_width);

	update();
}

void ClickcellWindow::checkEvent(){

	switch(m_eventTracker->eventType){

		case EventType::W_Resize:{
			resizeEvent();
		} break;

		case EventType::M_Clickleft:{
			mouseClicked();
		} break;

		case -1:{
		} break;

	}
}



