/* File: GraphTracker.cpp
 * Author: Matthew Overby
 */

#include "GraphTracker.h"

using namespace SLUI;

/*
*	GraphApp Functions
*/

GraphApp::GraphApp(Graph *g, std::string name){

	// Create the Render Window
	g_App = new sf::RenderWindow();
	g_App->Create(sf::VideoMode(800, 630), name);
	g_App->SetActive(true);

	// Make a copy of the graph
	if(g) graph = *g;
	graph.resizeEvent(800, 600, 0, 30);
	graph.draw(g_App);

	// Create the menu button background
	menuBg = sf::Shape::Rectangle(0, 0, 800, MenuItem::DefaultHeight, sf::Color(100, 100, 100, 255));
	g_App->Draw(menuBg);

	// Create the menu button
	graphOptions = new MenuItem(2, 2, "Graph Options", 0, 0, 0, g_App);
	graphOptions->addIndexItem(0, "Close Window");
	graphOptions->addIndexItem(1, "Save to PNG");
	graphOptions->addIndexItem(2, "Save to Text");
	graphOptions->draw();

	// Display the window
	g_App->Display();
}

GraphApp::~GraphApp(){

	delete g_App;
	delete graphOptions;
}

void GraphApp::resizeEvent(sf::Event Event){

	// Fix the view
	g_App->Clear();
	static sf::View view; 
	view.SetFromRect(sf::FloatRect(0, 0, Event.Size.Width, Event.Size.Height)); 
	g_App->SetView(view);

	// Resize the graph
	graph.resizeEvent(Event.Size.Width, Event.Size.Height, 0, 30);
	graph.draw(g_App);

	// Resize the menu background
	menuBg = sf::Shape::Rectangle(0, 0, Event.Size.Width, MenuItem::DefaultHeight, sf::Color(100, 100, 100, 255));
	g_App->Draw(menuBg);

	graphOptions->draw();
			
	// Display the window			
	g_App->Display();
}

void GraphApp::redraw(){

	g_App->Clear();
	graph.draw(g_App);
	g_App->Draw(menuBg);
	graphOptions->draw();
	g_App->Display();
}

/*
*	GraphTracker Functions
*/

GraphTracker::GraphTracker(OptionTracker *opT, EventTracker *evT){

	m_optTracker = opT;
	m_eventTracker = evT;
	updateWindow = true;
	graphsOpened = 0;
}

GraphTracker::~GraphTracker(){

	plotmap.clear();

	for(int i=0; i < g_Apps.size(); i++){
		GraphApp *delPtr = g_Apps.back();
		delete delPtr;
		g_Apps.pop_back();
	}
}

void GraphTracker::copy( GraphTracker *copyGraphTracker ){

	plotmap.clear();
	plotmap = *copyGraphTracker->getPlotMap();
	saveLocation = copyGraphTracker->saveLocation;
	xLabel = copyGraphTracker->xLabel;
	yLabel = copyGraphTracker->yLabel;

	updateWindow = true;
}

std::map< std::string, GraphLine >* GraphTracker::getPlotMap(){

	return &plotmap;
}

GraphLine GraphTracker::getLine( std::string label ){

	GraphLine result;
	std::map< std::string, GraphLine >::iterator finder = plotmap.find( label );	
	if( finder != plotmap.end() ){
		result = finder->second;
	}
	return result;
}

void GraphTracker::clearPlots(){

	plotmap.clear();
	updateWindow = true;
}

void GraphTracker::updateGraph(){

	std::map< std::string, GraphLine > temp = plotmap;
	plotmap.clear();

	std::map< std::string, GraphLine >::iterator iter;
	for( iter = temp.begin(); iter != temp.end(); iter++ ){
		plotmap.insert( std::pair< std::string, GraphLine >( iter->first, iter->second ) );
	}
	updateWindow = true;
}

void GraphTracker::addPlot(std::string label, std::vector<float> newX, std::vector<float> newY){

	GraphLine newLine;
	newLine.yPoints = newY;
	newLine.xPoints = newX;
	newLine.setColor(plotmap.size());
	newLine.label = label;
	updateWindow = true;

	plotmap.insert( std::pair< std::string, GraphLine >( label, newLine ) );
}

void GraphTracker::launchWindow(Graph *graph){

	graphsOpened++;
	std::stringstream name;
	name << "Graph " << graphsOpened;
	GraphApp *graphApp = new GraphApp(graph, name.str());
	g_Apps.push_back(graphApp);
	sf::Sleep(0.3);
}

void GraphTracker::checkEvent(sf::RenderWindow *m_App){

	if(g_Apps.size() > 0){

		m_App->SetActive(false);
		int delIndex = -1;

		for(int i=0; i<g_Apps.size(); i++){

			GraphApp *current = g_Apps.at(i);

			if(current->g_App->IsOpened()){

				current->g_App->SetActive(true);

				sf::Event newEvent;
				while(current->g_App->GetEvent(newEvent)){

					current->graphOptions->highlight();
					current->graphOptions->draw();
					current->g_App->Display();

					if(newEvent.Type == sf::Event::Closed){
						delIndex = i;
						current->g_App->Close();
					}
					else if(newEvent.Type == sf::Event::KeyPressed && newEvent.Key.Code == sf::Key::Escape){
						delIndex = i;
						current->g_App->Close();
					}
					else if(newEvent.Type == sf::Event::Resized){
						current->resizeEvent(newEvent);
					}
					else if(newEvent.Type == sf::Event::MouseButtonPressed && 
						newEvent.MouseButton.Button == sf::Mouse::Left){

						int clicked = current->graphOptions->getIndexClicked();

						switch(clicked){

							case 0:{
								delIndex = i;
								current->g_App->Close();
							} break;

							case 1:{ // Save to PNG
								sf::Sleep(0.3f);
								current->graphOptions->dropped = false;
								current->redraw();
								std::string fn = current->graph.printGraph(current->g_App);
								std::string message = "Graph exported as \""+fn+"\"";
								m_eventTracker->showMessage(message, 3);
							} break;

							case 2:{ // Save as text file
								sf::Sleep(0.3f);
								current->redraw();
								std::string fn = current->graph.printGraphData();
								std::string message = "Graph exported as \""+fn+"\"";
								m_eventTracker->showMessage(message, 3);
							} break;

							case -1:{
								current->redraw();
							} break;
						}

					}

				} // end while event

				current->g_App->SetActive(false);
			}

		}

		if(delIndex >= 0){
			GraphApp *delPtr = g_Apps.at(delIndex);
			delete delPtr;
			g_Apps.erase(g_Apps.begin() + delIndex);
		}

		m_App->SetActive(true);
	}
}

void GraphTracker::setSaveLocation(std::string str){

	saveLocation = str;
	updateWindow = true;
}

void GraphTracker::setLabels(std::string x, std::string y){

	xLabel = x;
	yLabel = y;
	updateWindow = true;
}

void GraphTracker::setXMinMax(float min, float max){

	if( min < max ){
		m_xMin = min;
		m_xMax = max;
	}
	updateWindow = true;
}

void GraphTracker::setYMinMax(float min, float max){

	if( min < max ){
		m_yMin = min;
		m_yMax = max;
	}
	updateWindow = true;
}













