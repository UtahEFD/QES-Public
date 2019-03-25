/* File: Graph.cpp
 * Author: Matthew Overby
 */

#include "Graph.h"

using namespace SLUI;

/*
*	GraphLine Functions
*/

GraphLine::GraphLine(){

	color = sf::Color(255, 0, 0);
}

void GraphLine::setColor(int numPlots){

        int numColors = 6;
	sf::Color *lineColors = new sf::Color[numColors];

	lineColors[0] = sf::Color(255, 0, 0);
	lineColors[1] = sf::Color(0, 255, 0);
	lineColors[2] = sf::Color(0, 0, 255);
	lineColors[3] = sf::Color(255, 255, 0);
	lineColors[4] = sf::Color(255, 0, 255);
	lineColors[5] = sf::Color(0, 255, 255);

	color = lineColors[numPlots % numColors];
	delete [] lineColors;
}

/*
*	Graph Functions
*/

Graph::Graph(){

	m_width = 800;
	m_height = 600;
	m_posX = 0;
	m_posY = 0;

	m_xMin = 0.f;
	m_xMax = 10.f;
	m_yMin = 0.f;
	m_yMax = 10.f;

	m_xInterval = 10;
	m_yInterval = 10;

	m_xLabel = sf::String("Time (hr)");
	m_yLabel = sf::String("Temperature (F)");

	m_yLabel.Rotate(90);
	plotsUpdated = true;
}

Graph::Graph(int w, int h, int px, int py) {

	m_width = w;
	m_height = h;
	m_posX = px;
	m_posY = py;

	m_xInterval = 10;
	m_yInterval = 10;

	m_xLabel = sf::String("Time (hr)");
	m_yLabel = sf::String("Temperature (F)");

	m_yLabel.Rotate(90);
	plotsUpdated = true;
}

Graph::~Graph() {

	removeVisuals();
	plotmap.clear();
}

void Graph::setXMinMax( float min, float max ){

	m_xMin = min;
	m_xMax = max;
}

void Graph::setYMinMax( float min, float max ){

	m_yMin = min;
	m_yMax = max;
}

void Graph::setInterval( int x, int y ){

	if( x <= 0 ) x = 1;
	if( y <= 0 ) y = 1;
	m_xInterval = x;
	m_yInterval = y;
}

void Graph::setLabels(std::string x, std::string y){

	m_xLabel = sf::String(x);
	m_yLabel = sf::String(y);
	m_yLabel.Rotate(90);
}

void Graph::draw(sf::RenderWindow *g_App){

	if(plotsUpdated){
		removeVisuals();
		calcVisuals();
		plotsUpdated = false;
	}

	for(int i=0; i<m_graph.size(); i++){
		g_App->Draw(m_graph[i]);
	}

	for(int i=0; i<m_labels.size(); i++){
		g_App->Draw(m_labels[i]);
	}
}

std::string Graph::printGraph(sf::RenderWindow *g_App){

	time_t ltime;
	ltime=time(NULL);
	std::stringstream timestamp(asctime( localtime(&ltime) ));
	std::string fileName = "";
	for(int i=0; i<5; i++){
		std::string temp = "";
		timestamp >> temp;
		fileName += temp;
		if(i<4) fileName += "-";
	}
	fileName += ".png";


	const sf::Image& screenshot = g_App->Capture();
	sf::Image graph;
	graph.Create(m_width, m_height);
	graph.SetSmooth(true);
	graph.Copy(screenshot, 0, 0, sf::IntRect(m_posX, m_posY, m_posX+m_width, m_posY+m_height));
	graph.SaveToFile(saveLocation+fileName);

	return saveLocation+fileName;
}

std::string Graph::printGraphData(){

	std::string xlabel = m_xLabel.GetText();
	std::string ylabel = m_yLabel.GetText();

	time_t ltime;
	ltime=time(NULL);
	std::stringstream timestamp(asctime( localtime(&ltime) ));
	std::string fileName = "";
	for(int i=0; i<5; i++){
		std::string temp = "";
		timestamp >> temp;
		fileName += temp;
		if(i<4) fileName += "-";
	}
	fileName += ".txt";
	std::string fullOutfile = saveLocation+fileName;

	std::ofstream outfile;
	outfile.open( fullOutfile.c_str()) ;

	std::stringstream header("");
	header << xlabel;

	std::map<float,std::vector<float> > values;

	std::map< std::string, GraphLine >::iterator iter;
	for( iter = plotmap.begin(); iter != plotmap.end(); iter++ ){

		std::map<float,std::vector<float> >::iterator finder;
		GraphLine currLine = iter->second;
		header << "\t\t" << currLine.label;

		for(int i=0; i<currLine.xPoints.size(); i++){

			finder = values.find( currLine.xPoints[i] );

			if( finder == values.end() ){
				std::vector<float> newYvals;
				newYvals.push_back( currLine.yPoints[i] );
				values.insert( std::pair<float,std::vector<float> >(
					currLine.xPoints[i],
					newYvals ) );
			}
			else{
				finder->second.push_back( currLine.yPoints[i] );
			}

		} // end loop x points
	} // end loop plots

	outfile << header.str();

	std::map<float,std::vector<float> >::iterator vals = values.begin();
	while( vals != values.end() ){

		std::stringstream liness("");
		std::vector<float> yvalsout = vals->second;
		liness << vals->first;
		
		for( int i=0; i<yvalsout.size(); i++ ){
			liness << "\t\t" << yvalsout[i];
		}

		outfile << liness.str() << "\n";

		vals++;
	}


	outfile.close();
	return saveLocation+fileName;
}

void Graph::update(std::map< std::string, GraphLine > newPlots){

	plotmap.clear();
	plotmap = newPlots;
	plotsUpdated = true;
}

void Graph::resizeEvent(int w, int h, int posX, int posY){

	m_posX = posX;
	m_posY = posY;
	m_height = h;
	m_width = w;
	plotsUpdated = true;
}

void Graph::setSaveLocation(std::string str){

	if(str.length() > 1){
		if(str.at(str.length()-1) == '/')
			saveLocation = str;
		else
			saveLocation += '/';
	}
}

void Graph::removeVisuals(){

	m_graph.clear();
	m_labels.clear();
}


void Graph::calcVisuals(){

	// Colors!
	sf::Color m_bgColor = sf::Color(255, 255, 255);
	sf::Color m_lineColor = sf::Color(0, 0, 0);
	sf::Color m_lineOpColor = sf::Color(0, 0, 0, 80);

	// Set visualization specs
	int lineThickness = 2;
	int lineLength = 10;
	int padding = 50;
	int margin = 25;
	float keyWidth = 100;
	float keyHeight = 100;
	float keyPadding = 20;

	// Create the Key:  Expands key background as needed
	float keyIndex = padding + 8;
	std::map<std::string, GraphLine>::iterator iter;
	for(iter = plotmap.begin(); iter != plotmap.end(); iter++){

		GraphLine currentLine = iter->second;
		sf::String plotKey(iter->first);

		plotKey.SetSize(16);
		plotKey.SetColor(currentLine.color);

		keyIndex += plotKey.GetRect().GetHeight();

		float keyElementSize = plotKey.GetRect().GetWidth() + keyPadding*2;
		if(keyElementSize > keyWidth){
			keyWidth = keyElementSize;
		}
		if(keyIndex+plotKey.GetRect().GetHeight() > (keyHeight + padding)){
			keyHeight += plotKey.GetRect().GetHeight();
		}

		plotKey.SetPosition(m_posX+m_width-keyWidth-padding+keyPadding, m_posY+keyIndex);

		m_labels.push_back(plotKey);
	}

	
	float numLinesX = m_xInterval;
	float numLinesY = m_yInterval;
	float xInterval = (m_width-padding*3-margin-keyWidth)/numLinesX;
	float yInterval = (m_height-padding*2-margin)/numLinesY;


	// Initialize labels/sprites
	sf::Shape m_background = sf::Shape::Rectangle(m_posX, m_posY, m_posX+m_width, m_posY+m_height, m_bgColor, 0, m_lineColor);

	sf::Shape m_keyBackground = sf::Shape::Rectangle(m_posX+m_width-padding-keyWidth, 
		m_posY+padding, m_posX+m_width-padding, m_posY+keyHeight+padding, sf::Color(240,240,240), lineThickness, m_lineColor);

	sf::String num;

	// Set label values
	sf::String keyLabel("Key");
	keyLabel.SetSize(14);
	keyLabel.SetColor(m_lineColor);
	keyLabel.SetPosition(m_posX+m_width-(keyWidth/2)-padding-keyLabel.GetRect().GetWidth()/2, m_posY+padding + 7);
	m_xLabel.SetSize(14);
	m_xLabel.SetColor(m_lineColor);
	m_xLabel.SetPosition(m_posX+m_width/2 - m_xLabel.GetRect().GetWidth()/2 - padding*2, m_posY+m_height-padding );
	m_yLabel.SetSize(14);
	m_yLabel.SetColor(m_lineColor);
	m_yLabel.SetPosition(m_posX+padding-30, m_posY+m_height/2 + 50);
	num.SetSize(12);
	num.SetColor(m_lineColor);

	// Set background/static stuff
	m_graph.push_back(m_background);
	m_graph.push_back(m_keyBackground);
	m_labels.push_back(m_xLabel);
	m_labels.push_back(m_yLabel);
	m_labels.push_back(keyLabel);

	// Calc x axis grid
	float xIndex = 0;
	float xVal = m_xMin;
	//bool xVals_are_fractions = false;
	//if( m_xSteps > m_xRange ){ xVals_are_fractions = true; }
	for(int x=0; x<=numLinesX; x++){

		sf::Shape line = sf::Shape::Line(xIndex, m_height, 
			xIndex, m_height-lineLength, lineThickness, m_lineColor, 0, m_lineColor);


		sf::Shape line2 = sf::Shape::Line(xIndex, m_height, 
			xIndex, padding*2+margin, lineThickness, m_lineOpColor, 0, m_lineOpColor);

		if(x == 0){ // black "tally marks"
			line = sf::Shape::Line(xIndex, m_height, 
			xIndex, padding*2+margin, lineThickness, m_lineColor, 0, m_lineColor);
		}


		if(xVal < 10 && xVal > -10){
			num.SetPosition(xIndex-4, m_height+6);
		}
		else {
			num.SetPosition(xIndex-6, m_height+6);
		}

		std::stringstream ss;
		char number[64];
		sprintf(number, "%.1f", xVal);
		ss << number;
		num.SetText(ss.str());	

		// Center the number on the line
		sf::Vector2f newPos = getLabelPos( 0, num, sf::Vector2f( xIndex, m_height-lineLength ) );
		num.SetPosition( newPos );

		line.Move(m_posX+margin+padding, m_posY-margin-padding);
		line2.Move(m_posX+margin+padding, m_posY-margin-padding);
		num.Move(m_posX+margin+padding, m_posY-margin-padding);

		m_graph.push_back((line));
		m_graph.push_back((line2));
		m_labels.push_back((num));

		xIndex += xInterval;
		xVal += (m_xMax - m_xMin)/numLinesX;
	}

	// Calc y axis grid
	float yIndex = m_height;
	float yVal = m_yMin;
	for(int y=0; y<=numLinesY; y++){

		sf::Shape line = sf::Shape::Line(0, yIndex, lineLength, yIndex, lineThickness, m_lineColor, 0, m_lineColor);
		sf::Shape line2 = sf::Shape::Line(0, yIndex, m_width-padding*3-margin-keyWidth, yIndex, lineThickness, m_lineOpColor, 0, m_lineOpColor);

		if(y == 0){
			line = sf::Shape::Line(0, yIndex, m_width-padding*3-margin-keyWidth, yIndex, lineThickness, m_lineColor, 0, m_lineColor);
		}


		if(yVal < 10 && yVal > -10)
			num.SetPosition(-12, yIndex-8);
		else if(yVal < 100 && yVal > -100)
			num.SetPosition(-20, yIndex-8);
		else
			num.SetPosition(-28, yIndex-8);

		std::stringstream ss;
		char number[64];
		sprintf(number, "%.1f", yVal);
		ss << number;
		num.SetText(ss.str());

		sf::Vector2f newPos = getLabelPos( 1, num, sf::Vector2f( 0, yIndex ) );
		num.SetPosition( newPos );

		line.Move(m_posX+margin+padding, m_posY-margin-padding);
		line2.Move(m_posX+margin+padding, m_posY-margin-padding);
		num.Move(m_posX+margin+padding, m_posY-margin-padding);

		m_graph.push_back((line));
		m_graph.push_back((line2));
		m_labels.push_back((num));

		yIndex -= yInterval;
		yVal += (m_yMax - m_yMin)/numLinesY;
	}

	// Calc Plots
	for(iter = plotmap.begin(); iter != plotmap.end(); iter++){

		GraphLine currentLine = iter->second;

		xIndex = 0;
		xVal = 0;
		float lastPointX = 0;
		float lastPointY = m_height;
		float yPos = m_height;
		float xPos = 0;
		float xIncrement = (m_xMax-m_xMin)/(float)currentLine.yPoints.size();

		// Points on each plot
		for(int j=0; j < currentLine.yPoints.size(); j++){

			float offset = 0.f;

			offset = 0.f - m_yMin;
			yVal = currentLine.yPoints[j];
			yPos = m_height - (yVal+offset)*(yInterval/((m_yMax+offset)/numLinesY));

			offset = 0.f - m_xMin;
			xVal = currentLine.xPoints[j];
			xPos = (xVal+offset)*(xInterval/((m_xMax+offset)/numLinesX));

			sf::Shape line;
			if(lastPointX >= m_xMin && lastPointY >= m_yMin && j > 0){
				line = sf::Shape::Line(xPos, yPos, 
				lastPointX, lastPointY, lineThickness, currentLine.color, 0, currentLine.color);
			}

			line.Move(m_posX+margin+padding, m_posY-margin-padding);

			if(yVal <= m_yMax && yVal >= m_yMin && xVal <= m_xMax && xVal >= m_xMin ){

				m_graph.push_back((line));

				lastPointX = xPos;
				lastPointY = yPos;
			}
			else{
				lastPointX = m_xMin-1;
				lastPointY = m_yMin-1;
			}

		}
	} // end calc plots

}


sf::Vector2f Graph::getLabelPos( int x_or_y, sf::String num, sf::Vector2f linePos ){

	sf::Vector2f position = num.GetPosition();
	float height = num.GetRect().GetHeight();
	float width = num.GetRect().GetWidth();

	if( x_or_y == 0 ){ // x

		position.y = linePos.y;
		position.y += (height + 4.f);

		position.x = linePos.x;
		position.x -= (width/2.f);

	}
	else if( x_or_y == 1 ){ // y

		position.x = linePos.x;
		position.x -= (width + 5.f);

		position.y = linePos.y;
		position.y -= (width/2.f);

	}

	return position;
}







