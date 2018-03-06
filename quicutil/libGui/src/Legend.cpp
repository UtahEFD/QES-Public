/* File: Legend.cpp
 * Author: Matthew Overby and Scot Halverson
 */

#include "Legend.h"

using namespace SLUI;

// TODO: Make createGradiant and createText based off of m_height/width and m_posX/Y
//	instead of the render window height/width
//	Also, have the gradiant be drawn with SFML instead of a displayList

Legend::Legend(EventTracker *evt, OptionTracker *opt, sf::RenderWindow* app){
    
    drawable = false;

	setMinMaxID( "minDrawValue", "maxDrawValue" );
	setLegendTextID( "legendtext" );

	int transparency = 200;
	m_colors.push_back( sf::Color( 255, 0, 0, transparency ) );
	m_colors.push_back( sf::Color( 255, 255, 0, transparency ) );
	m_colors.push_back( sf::Color( 0, 255, 0, transparency ) );
	m_colors.push_back( sf::Color( 0, 0, 255, transparency ) );
	m_colors.push_back( sf::Color( 128, 0, 128, transparency ) );


	int h = app->GetHeight();
	int w = app->GetWidth();
	m_App = app;
	gradiant = 99;  // initialized to 99 because if I don't init it to something,
			// the gradiant is sometimes not generated

	m_width = 120;
	m_height = 420;

	m_paddingR = m_width+15;
//	m_paddingB = m_height+230;
	m_paddingT = 40;

	m_posX = w-m_paddingR;
	m_posY = m_paddingT;
//	m_posY = h-m_paddingB;
	m_background = createBackground(m_posX, m_posY, m_height, m_width);

	m_optTracker = opt;
	m_eventTracker = evt;

	textColor = sf::Color(0, 0, 0);
	newCreateGradient();
	createText();
}


Legend::~Legend(){

	glDeleteLists(gradiant, 1);
}


void Legend::createGradiant(){

	unsigned int colors[5][3];
	colors[0][0] = 128; 	colors[0][1] = 0; 	colors[0][2] = 128; //purple
	colors[1][0] = 0; 	colors[1][1] = 0; 	colors[1][2] = 255; //blue
	colors[2][0] = 0; 	colors[2][1] = 255; 	colors[2][2] = 0; //green
	colors[3][0] = 255; 	colors[3][1] = 255; 	colors[3][2] = 0; //yellow
	colors[4][0] = 255; 	colors[4][1] = 0; 	colors[4][2] = 0; //red

	float minDrawValue = m_optTracker->getValue( minlabel );
	float maxDrawValue = m_optTracker->getValue( maxlabel );

	glNewList(gradiant, GL_COMPILE);

	// Delete the old display list
	glDeleteLists(gradiant, 1);

	glEnable (GL_BLEND); 
	glDisable(GL_DEPTH_TEST);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//here we set the viewport (portion of the window dedicated to the legend)
	glViewport( m_App->GetWidth() - 130, 230 , 130, 200 );
	
		int xMax = 130; //right edge of window
		int yMax = 200; //bottom edge of window
		int gradWidth = 100, gradHeight; //how wide to make the key
	
		static float transparency = .9; //transparency value
	
		//color values for the gradient
		unsigned int colorValues[3][3];
		for (int a = 0; a < 3; a++){
			for (int b = 0; b < 3; b++){
				colorValues[a][b] = 0;
			}
		}
		unsigned int *colorPtr;
		int numColors;
		//declare a color for the text displayed in the legend
		textColor = sf::Color(0, 0, 0);
	
		//then set what the color values are, depending on if we are in color rendering mode or B&W
		if ( !m_optTracker->getActive( "colormode" ) ){
			textColor.r = 255;
			colorValues[1][0] = 128;
			colorValues[1][1] = 128;
			colorValues[1][2] = 128;
			colorValues[2][0] = 255;
			colorValues[2][1] = 255;
			colorValues[2][2] = 255;
		
			colorPtr = *colorValues;
			numColors = 3;
		}
		else{
			colorPtr = *colors;
			numColors = 5;	
		}
	
		gradHeight = (gradWidth * 2) / (numColors - 1);
	
		//Here we push the current matrix to save it, then load the identity matrix
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
	
		//then we do some modifications to the matrix to set it in orthographic mode
		//looking at the area where we will render the legend
		glOrtho(0, 100, 0, 200, -1, 1);
		glMatrixMode(GL_MODELVIEW);
		// then we push the matrix again, and load the identity... 
		// Not entirely sure why at this point, seems to be neccessary though.  -Scot
		glPushMatrix();
		glLoadIdentity();
	
		//declare minimum and maximum values for the legend to display
		float maxKeyValue = maxDrawValue;
		float minKeyValue = minDrawValue;
	
		//draw gradient quads
		for (int a = 0; a < numColors - 1; a++){

			glBegin(GL_QUADS);
		
			glColor4f(
				colorPtr[(numColors - 1 - a) * 3 + 0]/256.f, 
				colorPtr[(numColors - 1 - a) * 3 + 1]/256.f, 
				colorPtr[(numColors - 1 - a) * 3 + 2]/256.f, 
				transparency); //top color, opaque
			
			glVertex2f(xMax -60, yMax - (a * gradHeight));  //top right corner
			
			glColor4f(
				colorPtr[(numColors -1 - a) * 3 + 0]/256.f, 
				colorPtr[(numColors -1 - a) * 3 + 1]/256.f, 
				colorPtr[(numColors - 1 - a) * 3 + 2]/256.f, 
				  transparency); //top color, transparent

			glVertex2f(xMax - gradWidth -30, yMax - (a * gradHeight)); //top left corner

			glColor4f(
				colorPtr[(numColors - (a + 2)) * 3 + 0]/256.f, 
				colorPtr[(numColors - (a + 2)) * 3 + 1]/256.f, 
				colorPtr[(numColors - (a + 2)) * 3 + 2]/256.f, 
				  transparency); //bottom color, transparent 

			glVertex2f(xMax - gradWidth -30, yMax - ((a + 1)*gradHeight)); //bottom left corner

			glColor4f(
				colorPtr[(numColors - (a + 2)) * 3 + 0]/256.f, 
				colorPtr[(numColors - (a + 2)) * 3 + 1]/256.f, 
				colorPtr[(numColors - (a + 2)) * 3 + 2]/256.f, 
				transparency); //bottom color, opaque

			glVertex2f(xMax -60, yMax - ((a + 1)*gradHeight) ); //bottom right corner

			glEnd();	
		}
		//draw a white bar next to the gradient
		glBegin(GL_QUADS);
		glColor4f(1,1,1,.6); //white
		glVertex2f(xMax - 45, yMax);
		glVertex2f(xMax - 45, yMax - 200);
		glVertex2f(xMax - 60, yMax - 200);
		glVertex2f(xMax - 60, yMax);
		glEnd();

	//return to previous openGL state, and other OpenGL cleanup
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	
	glEnable(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	
	glViewport(0,0, m_App->GetWidth() , m_App->GetHeight());
	
	glDepthMask(GL_TRUE);
	glDisable(GL_BLEND);
	

	glEndList();
}


void Legend::createText(){

	int h = m_App->GetHeight();
	int w = m_App->GetWidth();

	int label_paddingL = 45;


	//create some stringstreams and strings for text output
	std::stringstream textss1, textss2, textss3;
	float minDrawValue = m_optTracker->getValue( minlabel );
	float maxDrawValue = m_optTracker->getValue( maxlabel );
	int legendTextVal = (int)m_optTracker->getValue( legendtext );
	
	//and then write out some data to these stringstreams/strings and display them to the screen
	textss1 << minDrawValue;
	minVal.SetText(textss1.str());
	minVal.SetFont(sf::Font::GetDefaultFont());
	minVal.SetColor(textColor);
	minVal.SetScale(.5, .5);
	minVal.SetPosition( m_posX+label_paddingL, m_posY+m_height-20 );
	
	textss2 << (maxDrawValue - minDrawValue) / 2 + minDrawValue;
	midVal.SetText(textss2.str());
	midVal.SetFont(sf::Font::GetDefaultFont());
	midVal.SetColor(textColor);
	midVal.SetScale(.5, .5);
	midVal.SetPosition( m_posX+label_paddingL, m_posY+m_height/2.f-10 );
	
	textss3 << maxDrawValue ;
	maxVal.SetText(textss3.str());
	maxVal.SetFont(sf::Font::GetDefaultFont());
	maxVal.SetColor(textColor);
	maxVal.SetScale(.5, .5);
	maxVal.SetPosition( m_posX+label_paddingL, m_posY+2 );

	std::string legendText = "";

	switch( legendTextVal ){
		case 0:{
			legendText = "Temperature (K)";
		} break;
		case 1:{
			legendText = "Sky View Ratio";
		} break;
		case 2:{
			legendText = "Sun View Ratio";
		} break;
		case 3:{
			legendText = "Energy Balance (W/m^2)";
			minVal.Move( -15.0, 0.0 );
			maxVal.Move( -10.0, 0.0 );
		} break;
		case 4:{
			legendText = "Sunlit View Factor";
		} break;
		case 5:{
			legendText = "Absorptance";
		} break;
		case 6:{
			legendText = "Watts";
		} break;
		case 7:{
			legendText = "Watts / m^2";
		} break;
	}

	label = sf::String();
	label.SetText(legendText);
	label.SetFont(sf::Font::GetDefaultFont());
	label.SetColor(textColor);
	label.SetScale(.5, .5);
	label.Rotate(-90);
	label.SetPosition( m_posX+label_paddingL+70, m_posY+10 );

}


void Legend::draw(){
    
    if (!drawable) {
        newCreateGradient();
    }

	m_App->Draw( m_background );
	m_App->Draw( gradientSprite );
	m_App->Draw( minVal );
	m_App->Draw( maxVal );
	m_App->Draw( midVal );
	m_App->Draw( label );
}


void Legend::resizeEvent(){

	int h = m_App->GetHeight();
	int w = m_App->GetWidth();

	int xMove = w-m_paddingR - m_posX;
	m_posX += xMove;

//	int yMove = h-m_paddingB - m_posY;
	int yMove = m_paddingT - m_posY;
	m_posY += yMove;

//	m_posX = w-m_paddingR;
//	m_posY = h-m_paddingB;

	newCreateGradient();
	createText();
	m_background = createBackground( m_posX, m_posY, m_height, m_width );
}


void Legend::checkEvent(){

	if( m_optTracker->stateChanged( "colormode" ) ||
		m_optTracker->stateChanged( minlabel ) || 
		m_optTracker->stateChanged( maxlabel ) ){
		newCreateGradient();
		createText();
	}

	switch(m_eventTracker->eventType){

		case EventType::W_Resize:{
			resizeEvent();
		} break;

		case -1:{
		} break;

	} // end switch eventType
}


void Legend::newCreateGradient(){

	ColorScale imgGradient;
	int transparency = 200;

    std::string colormapName = m_optTracker->getListValue("colormap");
    if (colormapName != "") {
        imgGradient = ColorScale(colormapName);
    } else {
        drawable = false;
        return;
    }

 	int gHeight = m_height - 4;
	int gWidth = m_width - 35;

	sf::Color* tab =new sf::Color[ gHeight ];
	imgGradient.fillTab( tab, gHeight );

	gradientImage = sf::Image( gWidth, gHeight );
	for( int i=0; i < gWidth; i++ ){
		for( int j=0; j < gHeight; j++ ){
			gradientImage.SetPixel( i, (gHeight - 1) - j, tab[j] );
		}		
	}

	gradientSprite = sf::Sprite( gradientImage );
	gradientSprite.SetPosition( m_posX+5, m_posY+2 );

	delete[] tab;
    drawable = true;
}


void Legend::setPosition( int posX, int posY ){

	m_posX = posX;
	m_posY = posY;

	//m_paddingR = m_App->GetWidth() - m_width;
	//m_paddingB = m_App->GetHeight()  - m_height;

	m_background = createBackground(m_posX, m_posY, m_height, m_width);
	newCreateGradient();
	createText();
}


void Legend::setColors( std::vector<sf::Color> colors ){

	m_colors.clear();
	m_colors = colors;
	newCreateGradient();
}









