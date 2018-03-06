/* File: GraphWindow.cpp
 * Author: Matthew Overby
 */

#include "GraphWindow.h"

using namespace SLUI;

GraphWindow::GraphWindow(int newid, WindowController *wnC, GraphTracker *grT, OptionTracker *opT, EventTracker *evt, sf::RenderWindow* app){

	int w = app->GetWidth();
	int h = app->GetHeight();
	id = newid;

	m_App = app;
	m_optTracker = opT;
	m_graphTracker = grT;
	m_eventTracker = evt;
	m_winController = wnC;
	m_height = h-90;
	m_width = w-60;
	m_posX = 30;
	m_posY = 60;

	int graphHeight = m_height - 50;
	int graphWidth = m_width - Button::Width - 100;
	int graphPosX = m_posX + 25;
	int graphPosY = m_posY + 25;
	m_graph = new Graph(graphWidth, graphHeight, graphPosX, graphPosY);
	m_graph->setInterval( m_optTracker->getValue( "xinterval" ), m_optTracker->getValue( "yinterval" ) );

	exportButton = new StandardButton("Save to PNG", m_eventTracker, m_App);
	exportButton->setPosition(  m_posX+m_width-Button::Width-50, m_posY+20 );

	printButton = new StandardButton("Save to Text", m_eventTracker, m_App);
	printButton->setPosition( m_posX+m_width-Button::Width-50, m_posY+60 );

	redrawButton = new StandardButton("Refresh", m_eventTracker, m_App);
	redrawButton->setPosition( m_posX+m_width-Button::Width-50, m_posY+100 );

	newWinButton = new StandardButton("Open in Window", m_eventTracker, m_App);
	newWinButton->setPosition( m_posX+m_width-Button::Width-50, m_posY+140 );

	clearPlotsButton = new StandardButton("Clear Graph", m_eventTracker, m_App);
	clearPlotsButton->setPosition( m_posX+m_width-Button::Width-50, m_posY+180 );

	showButton = new ListButton("      Show Plot", m_eventTracker, m_App);
	showButton->setPosition( m_posX+m_width-Button::Width-50, m_posY+220 );

	std::vector<std::string> empty;
	empty.push_back("Show All");
	showButton->setDropList(empty);
	showButton->setDropSelected("Show All");

	settingsButton = new StandardButton("Settings", m_eventTracker, m_App);
	settingsButton->setPosition( m_posX+m_width-Button::Width-50, m_posY+m_height-90 );

	closeButton = new StandardButton("Close", m_eventTracker, m_App);
	closeButton->setPosition( m_posX+m_width-Button::Width-50, m_posY+m_height-50 );

	m_background = createBackground(m_posX, m_posY, m_height, m_width);
}

GraphWindow::~GraphWindow(){

	delete m_graph;
	delete closeButton;
	delete exportButton;
	delete redrawButton;
	delete newWinButton;
	delete settingsButton;
	delete clearPlotsButton;
	delete printButton;
	delete showButton;
}

void GraphWindow::update() {

	m_graph->setXMinMax(m_graphTracker->m_xMin, m_graphTracker->m_xMax);
	m_graph->setYMinMax(m_graphTracker->m_yMin, m_graphTracker->m_yMax);
	m_graph->setInterval( m_optTracker->getValue( "xinterval" ), m_optTracker->getValue( "yinterval" ) );

	m_graph->setSaveLocation(m_graphTracker->saveLocation);
	m_graph->setLabels(m_graphTracker->xLabel, m_graphTracker->yLabel);

	m_graphTracker->updateGraph();
	m_graph->update(*m_graphTracker->getPlotMap());

	std::map< std::string, GraphLine > *plotmap = m_graphTracker->getPlotMap();
	std::map< std::string, GraphLine >::iterator iter;

	std::vector<std::string> labels;
	labels.push_back("Show All");

	for( iter = plotmap->begin(); iter != plotmap->end(); iter++){
		labels.push_back(iter->first);
	}
	showButton->setDropList(labels);
	showButton->setDropSelected("Show All");

	m_graphTracker->updateWindow = false;
}

void GraphWindow::highlight() {

	closeButton->highlight();
	newWinButton->highlight();
	exportButton->highlight();
	redrawButton->highlight();
	settingsButton->highlight();
	clearPlotsButton->highlight();
	showButton->highlight();
	printButton->highlight();
}

void GraphWindow::mouseClicked() {
	
	const sf::Input *input = &m_App->GetInput();

	closeButton->checkEvent();
	newWinButton->checkEvent();
	exportButton->checkEvent();
	redrawButton->checkEvent();
	settingsButton->checkEvent();
	clearPlotsButton->checkEvent();
	showButton->checkEvent();
	printButton->checkEvent();

	if(newWinButton->updated){
		newWinButton->updated = false;
		m_graphTracker->launchWindow(m_graph);
	}
	else if(redrawButton->updated){
		redrawButton->updated = false;
		update();
	}
	else if(clearPlotsButton->updated){
		clearPlotsButton->updated = false;
		m_graphTracker->clearPlots();
		update();
	}

	else if(exportButton->updated){
		exportButton->updated = false;
		std::string fn = m_graph->printGraph(m_App);
		std::string message = "Graph exported as \""+fn+"\"";
		m_eventTracker->showMessage(message, 3);
	}
	else if(showButton->updated){
		showButton->updated = false;

		std::string currentChoice = showButton->getDropSelected();

		if(currentChoice == "Show All"){
			update();
		}
		else{
			//std::vector< GraphLine > *plots = m_graphTracker->getPlots();
			std::map< std::string, GraphLine > *plotmap = m_graphTracker->getPlotMap();
			std::map< std::string, GraphLine >::iterator iter;
			for( iter = plotmap->begin(); iter != plotmap->end(); iter++){

			if(currentChoice.compare( iter->first )==0){
					std::map< std::string, GraphLine > tempPlotlist;
					tempPlotlist.insert( std::pair< std::string, GraphLine >( iter->first, iter->second ) );
					m_graph->update(tempPlotlist);
				}
			}
		}
	}
	else if(closeButton->updated){
		closeButton->updated = false;
		close();
	}
	else if(settingsButton->updated){
		settingsButton->updated = false;
		m_winController->open(GRAPHING);
	}
	else if(printButton->updated){
		printButton->updated = false;
		std::string fn = m_graph->printGraphData();
		std::string message = "Graph exported as \""+fn+"\"";
		m_eventTracker->showMessage(message, 3);
	}
}

void GraphWindow::draw(){

	m_App->Draw(m_background);
	closeButton->draw();
	newWinButton->draw();
	exportButton->draw();
	clearPlotsButton->draw();
	redrawButton->draw();
	settingsButton->draw();
	showButton->draw();
	printButton->draw();

	if(m_graphTracker->updateWindow) update();
	m_graph->draw(m_App);
}

void GraphWindow::close(){

	m_winController->close(id);
}

void GraphWindow::open(){

	m_winController->open(id);
}

void GraphWindow::resizeEvent(){

	m_height = m_App->GetHeight()-90;
	m_width = m_App->GetWidth()-60;
	m_posX = 30;
	m_posY = 60;

	int graphHeight = m_height - 50;
	int graphWidth = m_width - Button::Width - 100;
	int graphPosX = m_posX + 25;
	int graphPosY = m_posY + 25;
	m_graph->resizeEvent(graphWidth, graphHeight, graphPosX, graphPosY);

	exportButton->setPosition(m_posX+m_width-Button::Width-50, m_posY+20);
	printButton->setPosition(m_posX+m_width-Button::Width-50, m_posY+60);
	redrawButton->setPosition(m_posX+m_width-Button::Width-50, m_posY+100);
	newWinButton->setPosition(m_posX+m_width-Button::Width-50, m_posY+140);
	clearPlotsButton->setPosition(m_posX+m_width-Button::Width-50, m_posY+180);
	showButton->setPosition(m_posX+m_width-Button::Width-50, m_posY+220);
	settingsButton->setPosition(m_posX+m_width-Button::Width-50, m_posY+m_height-90);
	closeButton->setPosition(m_posX+m_width-Button::Width-50, m_posY+m_height-50);

	m_background = createBackground(m_posX, m_posY, m_height, m_width);
}

void GraphWindow::checkEvent(){

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




