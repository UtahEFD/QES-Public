/* File: Button.cpp
 * Author: Matthew Overby
 */

#include "Button.h"

//TODO have a type string const (like the int type) for config to write
// out/load

using namespace SLUI;

DropOption::DropOption(std::string newLabel, float x, float y){

	m_posX = x;
	m_posY = y;

	label = sf::String(newLabel);
	label.SetPosition(m_posX+30, m_posY+2);
	label.SetSize(18);
	label.SetColor(sf::Color(0,0,0));

	m_background = sf::Shape::Rectangle(0, 0, Button::Width, Button::Height, sf::Color(255,255,255,200));
	m_background.Move(m_posX, m_posY);

	icon = sf::Shape::Circle(m_posX+15, m_posY+15, 6, sf::Color(255, 255,255,200), 1, sf::Color(0,0,0));
	icon_on = sf::Shape::Circle(m_posX+15, m_posY+15, 4, sf::Color(0,180,0), 1, sf::Color(0,0,0));
}

void DropOption::draw(sf::RenderWindow *m_App, bool active){

	m_App->Draw(m_background);
	m_App->Draw(label);
	m_App->Draw(icon);
	if(active) m_App->Draw(icon_on);
}

void DropOption::highlight(sf::RenderWindow *m_App){

	const sf::Input &input = m_App->GetInput();
	float mouseX = input.GetMouseX();
	float mouseY = input.GetMouseY();

	if((mouseX < m_posX + Button::Width) && (m_posX < mouseX) && 
	(mouseY < m_posY + Button::Height) && (m_posY < mouseY)) {
		m_background.SetColor(sf::Color(255, 153, 051, 200));
	}
	else {
		m_background.SetColor(sf::Color(255,255,255,200));
	}
}

bool DropOption::mouseClicked(sf::RenderWindow *m_App){

	const sf::Input &input = m_App->GetInput();
	float mouseX = input.GetMouseX();
	float mouseY = input.GetMouseY();

	if((mouseX < m_posX + Button::Width) && (m_posX < mouseX) && 
	(mouseY < m_posY + Button::Height) && (m_posY < mouseY)) {
		return true;
	}
	else
	  return false;
}

void DropOption::move(float x, float y){

	m_posX = x;
	m_posY = y;

	label.SetPosition(m_posX+30, m_posY+2);
	m_background.SetPosition(m_posX, m_posY);

	icon = sf::Shape::Circle(m_posX+15, m_posY+15, 6, sf::Color(255, 255,255,200), 1, sf::Color(0,0,0));
	icon_on = sf::Shape::Circle(m_posX+15, m_posY+15, 4, sf::Color(0,180,0), 1, sf::Color(0,0,0));
}





/*
*	Button Base
*/

void Button::onMouseClicked(){

	const sf::Input &input = m_App->GetInput();
	float mouseX = input.GetMouseX();
	float mouseY = input.GetMouseY();

	if( isMouseOver() ) {

		if(!active) active = true;
		else active = false;
		updated = true;
	}
}

void Button::highlight(){

	if(isMouseOver()) { m_background.SetColor(m_highlightColor); }
	else { m_background.SetColor(m_bgColor); }
}


void Button::checkEvent(){

	switch(m_eventTracker->eventType){

		case EventType::M_Clickleft:{
			onMouseClicked();
		} break;

		case -1:{
		} break;

	}
}

void Button::setPosition(float x, float y){

	m_posX = x;
	m_posY = y;

	m_label.SetPosition(m_posX+m_paddingL, m_posY+m_paddingT);
	m_background = createBackground(m_posX, m_posY, m_height, m_width);
}

void Button::setLabel( std::string lab ){

	//if( m_font.LoadFromFile("resources/Helvetica.ttf", LabelSize, lab ) ) {
	//	m_label = sf::String(lab, m_font, LabelSize);
	//}
	//else {
		m_label.SetText( lab );
		m_label.SetSize( LabelSize );
	//}
	setPosition( m_posX, m_posY );
	m_label.SetColor( m_textColor );
}

void Button::setSize(float width, float height){

	m_width = width;
	m_height = height;
	m_background = createBackground(m_posX, m_posY, m_height, m_width);
}

float Button::getValue(){ return 0; }
void Button::setValue(float v){}
void Button::setMinMax( float min, float max ){}
void Button::setNewKey(std::string newKey){}
void Button::setDropList( std::vector<std::string> ){}
void Button::setDropSelected( std::string str ){}
void Button::clearSelected(){}
std::string Button::getDropSelected(){ return ""; }



/*
*	Standard Button
*/



StandardButton::StandardButton(std::string lab, EventTracker *e, sf::RenderWindow* app){

	m_App = app;
	m_eventTracker = e;

	//if( m_font.LoadFromFile("resources/Helvetica.ttf", LabelSize, lab ) ) {
	//	m_label = sf::String(lab, m_font, LabelSize);
	//}
	//else {
		m_label.SetText( lab );
		m_label.SetSize( LabelSize );
	//}

	m_label.SetColor(sf::Color(0,0,0));
	type = ButtonType::Standard;
	active = false;
	updated = false;
	m_posX = 0;
	m_posY = 0;

	m_width = Width;
	m_height = Height;
	m_paddingL = 30;
	m_paddingT = 2;

	changeColors(m_borderColor, sf::Color(255, 255, 255, 255), m_highlightColor, m_textColor);
	m_background = createBackground(m_posX, m_posY, m_height, m_width);

}

void StandardButton::draw(){

	m_App->Draw(m_background);
	m_App->Draw(m_label);
}

void StandardButton::setPosition(float x, float y){

	m_posX = x;
	m_posY = y;

	m_label.SetPosition(m_posX+m_paddingL, m_posY+m_paddingT);
	m_background.SetPosition(m_posX, m_posY);
}

void StandardButton::onMouseClicked(){
	if( isMouseOver() ){
		updated = true;
	}
}


/*
*	Radio Button
*/


RadioButton::RadioButton(std::string lab, bool init, EventTracker *e, sf::RenderWindow* app){

	m_App = app;
	m_eventTracker = e;

	//if( m_font.LoadFromFile("resources/Helvetica.ttf", LabelSize, lab ) ) {
	//	m_label = sf::String(lab, m_font, LabelSize);
	//}
	//else {
		m_label.SetText( lab );
		m_label.SetSize( LabelSize );
	//}

	m_label.SetColor(sf::Color(0,0,0));
	type = ButtonType::Radio;
	active = init;
	updated = false;
	m_posX = 0;
	m_posY = 0;

	m_width = Width;
	m_height = Height;
	m_paddingL = 30;
	m_paddingT = 2;

	changeColors(m_borderColor, sf::Color(255, 255, 255, 255), m_highlightColor, m_textColor);
	m_background = createBackground(m_posX, m_posY, m_height, m_width);
}

void RadioButton::draw(){

	m_App->Draw(m_background);
	m_App->Draw(m_label);
	m_App->Draw(icon);
	if(active) m_App->Draw(activeIcon);
}

void RadioButton::setPosition(float x, float y){

	m_posX = x;
	m_posY = y;

	icon = sf::Shape::Circle(m_posX+15, m_posY+15, 6, m_bgColor, 1, sf::Color(0,0,0));
	activeIcon = sf::Shape::Circle(m_posX+15, m_posY+15, 4, sf::Color(0,180,0), 1, sf::Color(0,0,0));
	m_label.SetPosition(m_posX+m_paddingL, m_posY+m_paddingT);
	m_background.SetPosition(m_posX, m_posY);

}

float RadioButton::getValue(){
	return active;
}

void RadioButton::setValue( float v ){

	if( v == 0 ) active = false;
	else if( v == 1) active = true;
	updated = true;
}




/*
*	Value Button
*/




ValueButton::ValueButton(std::string lab, float init, EventTracker *e, sf::RenderWindow* app){

	m_App = app;
	m_eventTracker = e;
	value = init;

	if(value > 0){
		setMinMax( 0, value*2 );
	}
	else if(value == 0){
		setMinMax( -1, 1 );
	}
	else{
		setMinMax( value*2, 0 );
	}

	// Set the label
	//if( m_font.LoadFromFile("resources/Helvetica.ttf", LabelSize, lab ) ) {
	//	m_label = sf::String(lab, m_font, LabelSize);
	//}
	//else {
		m_label.SetText( lab );
		m_label.SetSize( LabelSize );
	//}

	// Set the value text
	val.SetText("0.00");
	val.SetColor(m_textColor);
	val.SetSize(LabelSize);


	m_label.SetColor(m_textColor);
	type = ButtonType::Value;
	active = false;
	updated = false;
	m_posX = 0;
	m_posY = 0;

	m_width = Width;
	m_height = Height;
	m_paddingL = 20;
	m_paddingT = 2;

	changeColors(m_borderColor, sf::Color(255, 255, 255, 255), m_highlightColor, m_textColor);
	m_background = createBackground(m_posX, m_posY, m_height, m_width);

	float lineBegin = m_posX+20;
	float lineEnd = m_posX+m_width-80;

	incIcon = sf::Shape::Rectangle(0, 0, 10, Height-10, m_bgColor, 1, sf::Color(0,0,0));
	incIcon.SetPosition( lineEnd, m_posY+5 );

	decIcon = sf::Shape::Rectangle(0, 0, 10, Height-10, m_bgColor, 1, sf::Color(0,0,0));
	decIcon.SetPosition( lineBegin-10, m_posY+5 );

	decLab.AddPoint( 0, 0, sf::Color::Black, sf::Color::Black);
	decLab.AddPoint( 0, 8, sf::Color::Black, sf::Color::Black);
	decLab.AddPoint( -4, 4, sf::Color::Black, sf::Color::Black);
	decLab.SetOutlineWidth(1);
	decLab.SetPosition( lineBegin-3, m_posY+11 );

	incLab.AddPoint( 0, 0, sf::Color::Black, sf::Color::Black);
	incLab.AddPoint( 0, 8, sf::Color::Black, sf::Color::Black);
	incLab.AddPoint( 4, 4, sf::Color::Black, sf::Color::Black);
	incLab.SetOutlineWidth(1);
	incLab.SetPosition( lineEnd+3, m_posY+11 );

}

void ValueButton::draw(){

	m_App->Draw(m_background);

	if(!active){

		char temp[128];
		if(abs(value) < 10)
			sprintf(temp, "%.2f", value);
		else if(abs(value) < 100)
			sprintf(temp, "%.1f", value);
		else if(abs(value) < 1000)
			sprintf(temp, "%3.0f", value);

		val.SetText(temp);
		if(value < 0){
			val.SetPosition(val.GetPosition().x-5, val.GetPosition().y);
			m_App->Draw(val);
			val.SetPosition(val.GetPosition().x+5, val.GetPosition().y);
		}
		else m_App->Draw(val);

		m_App->Draw(m_label);

	}
	else {

		char temp[128];
		if(abs(value) < 10)
			sprintf(temp, "%.2f", value);
		else if(abs(value) < 100)
			sprintf(temp, "%.1f", value);
		else if(abs(value) < 1000)
			sprintf(temp, "%3.0f", value);

		val.SetText(temp);

		if(value < 0){
			val.SetPosition(val.GetPosition().x-5, val.GetPosition().y);
			m_App->Draw(val);
			val.SetPosition(val.GetPosition().x+5, val.GetPosition().y);
		}
		else m_App->Draw(val);

		float lineBegin = m_posX+20;
		float lineEnd = m_posX+m_width-80;

		m_App->Draw( sf::Shape::Line(lineBegin, m_posY+Height/2, 
			lineEnd, m_posY+Height/2, 4, m_textColor) );

		icon = sf::Shape::Rectangle(sliderPosX, m_posY+5, sliderPosX+10, 
			m_posY+Height-5, m_bgColor, 1, sf::Color(0,0,0));

		m_App->Draw(icon);
		m_App->Draw(incIcon);
		m_App->Draw(decIcon);
		m_App->Draw(decLab);
		m_App->Draw(incLab);
	}
}

void ValueButton::setPosition(float x, float y){

	m_posX = x;
	m_posY = y;

	m_label.SetPosition(m_posX+m_paddingL, m_posY+m_paddingT);
	m_background.SetPosition(m_posX, m_posY);

	val.SetPosition(m_posX+m_width-50, m_posY+2);

	float lineBegin = m_posX+20;
	float lineEnd = m_posX+m_width-80;

	incIcon.SetPosition( lineEnd, m_posY+5 );
	incLab.SetPosition( lineEnd+3, m_posY+11 );
	decIcon.SetPosition( lineBegin-10, m_posY+5 );
	decLab.SetPosition( lineBegin-3, m_posY+11 );

}

void ValueButton::checkEvent(){

	//highlight();

	switch(m_eventTracker->eventType){

		case EventType::M_Clickleft:{
			onMouseClicked();
		} break;

		case EventType::M_Dragleft:{
			onMouseDragged();
		} break;

		case -1:{
		} break;

	}
}

void ValueButton::onMouseClicked(){

	const sf::Input &input = m_App->GetInput();
	float mouseX = input.GetMouseX();
	float mouseY = input.GetMouseY();


	// If the click was over the icon, it was probably meant to be
	// a mouseDragged and registered as a mouseClicked
	if(active && (mouseX < sliderPosX + 10) && (sliderPosX < mouseX) && 
		(mouseY < m_posY+Height-5) && ( m_posY+5 < mouseY)) {
	}
	else if( isMouseOver() ) {

		float decPosX = m_posX+10;
		float decPosY = m_posY+5;
		float incPosX = m_posX+m_width-80;
		float incPosY = m_posY+5;
		float lineBegin = m_posX+20;
		float lineEnd = m_posX+m_width-80;
		updated = true;


		if(active && (mouseX < decPosX + 10) && (decPosX < mouseX) && 
		(mouseY < decPosY+Height-5) && ( decPosY < mouseY)) {
			// Over decrement button

			float change = (maxVal - minVal) * 0.01;
			if( (value - change) > minVal)
				value -= change;
			else
				value = minVal;				
		}
		else if(active && (mouseX < incPosX + 10) && (incPosX < mouseX) && 
		(mouseY < incPosY+Height-5) && ( incPosY < mouseY)) {
			// Over increment button

			float change = (maxVal - minVal) * 0.01;
			if( (value + change) < maxVal)
				value += change;
			else
				value = maxVal;	
		}
		else{
			if(!active) active = true;
			else active = false;
		}

		float offSet = 0;
		if(minVal < 0){
			offSet = minVal;
			minVal -= offSet;
			maxVal -= offSet;
			value -= offSet;
		}

		sliderPosX = (lineEnd-10-lineBegin)*(value/(maxVal-minVal))+lineBegin;

		if(offSet < 0){
			minVal += offSet;
			maxVal += offSet;
			value += offSet;
		}
	}

}

void ValueButton::onMouseDragged(){

	float horz = (m_eventTracker->oldPos.x - m_eventTracker->newPos.x);
	const sf::Input &input = m_App->GetInput();
	float mouseX = m_eventTracker->oldPos.x;
	float mouseY = m_eventTracker->oldPos.y;

	// Check if the mouse over the slider before the drag
	if(active && (mouseX < sliderPosX + 10) && (sliderPosX < mouseX) && 
		(mouseY < m_posY+Height-5) && ( m_posY+5 < mouseY)) {

		updated = true;

		float new_sliderPosX = sliderPosX;
		new_sliderPosX -= horz;
		float lineBegin = m_posX+20;
		float lineEnd = m_posX+m_width-80;

		if( new_sliderPosX+10 < lineEnd && new_sliderPosX > lineBegin ){
			sliderPosX = new_sliderPosX;
			value -= horz * (maxVal - minVal)/(lineEnd - lineBegin - 10);
		}
		else if( new_sliderPosX+10 > lineEnd ){
			setValue( maxVal );
		}
		else if(  new_sliderPosX < lineBegin ){
			setValue( minVal );
		}

	}
}

void ValueButton::setMinMax( float min, float max ){

	minVal = min;
	maxVal = max;
	dragSpeed = fabs( (maxVal-minVal)/40 );
	updated = true;
}

float ValueButton::getValue(){
	return value;
}

void ValueButton::setValue(  float v ){
	value = v;
	updated = true;
}

void ValueButton::setSize(float w, float h){

	m_width = w;
	m_height = h;
	m_background = createBackground(m_posX, m_posY, m_height, m_width);

	float lineBegin = m_posX+20;
	float lineEnd = m_posX+m_width-80;

	incIcon = sf::Shape::Rectangle(0, 0, 10, m_height-10, m_bgColor, 1, sf::Color(0,0,0));
	incIcon.SetPosition( lineEnd, m_posY+5 );

	decIcon = sf::Shape::Rectangle(0, 0, 10, m_height-10, m_bgColor, 1, sf::Color(0,0,0));
	decIcon.SetPosition( lineBegin-10, m_posY+5 );

	decLab.SetPosition( lineBegin-3, m_posY+11 );
	incLab.SetPosition( lineEnd+3, m_posY+11 );
}




/*
*	Key Button
*/




KeyButton::KeyButton(std::string lab, std::string init, EventTracker *e, sf::RenderWindow* app){

	type = ButtonType::Key;
	m_App = app;
	m_eventTracker = e;
	m_width = Width;
	m_height = Height;
	lab += ":";
	active = false;
	updated = false;
	m_posX = 0;
	m_posY = 0;
	m_paddingL = 20;
	m_paddingR = 40;
	m_paddingT = 2;

	// Set the label
	//if( m_font.LoadFromFile("resources/Helvetica.ttf", LabelSize, lab ) ) {
	//	m_label = sf::String(lab, m_font, LabelSize);
	//}
	//else {
		m_label.SetText( lab );
		m_label.SetSize( LabelSize );
	//}
	m_label.SetColor(m_textColor);

	key.SetText( init );
	key.SetColor(m_textColor);
	key.SetSize(LabelSize);

	float keySize = key.GetRect().GetWidth();
	m_paddingR = keySize+20;

	changeColors(m_borderColor, sf::Color(255, 255, 255, 255), m_highlightColor, m_textColor);
	m_background = createBackground(m_posX, m_posY, m_height, m_width);

}

void KeyButton::draw(){

	m_App->Draw(m_background);
	m_App->Draw(m_label);
	sf::String temp = key;
	std::string keyStr = key.GetText();
	temp.SetText("  " + keyStr + "  ");
	if(active){
		temp.SetStyle(sf::String::Underlined);
	}
	m_App->Draw(temp);
}

void KeyButton::setPosition(float x, float y){

	m_posX = x;
	m_posY = y;

	m_label.SetPosition(m_posX+m_paddingL, m_posY+m_paddingT);
	m_background = createBackground(m_posX, m_posY, m_height, m_width);
	key.SetPosition(m_posX+m_width-m_paddingR, m_posY+m_paddingT);

}

void KeyButton::checkEvent(){

	//highlight();

	switch(m_eventTracker->eventType){

		case EventType::M_Clickleft:{
			onMouseClicked();
		} break;

		case EventType::K_Press:{
			if( active ) onKeyPressed();
		} break;

		case -1:{
		} break;
	}

}

float KeyButton::getValue(){

	Keys tempKey;
	std::string keyStr = key.GetText();
	return (int)tempKey.findKey( keyStr );
}

void KeyButton::setValue( float v ){
	//TODO
}

void KeyButton::onKeyPressed(){

	Keys tempKey;
	std::string desiredKey = tempKey.keyToString( m_eventTracker->lastKeyCode );

	if( desiredKey.compare("bad") != 0 ){
		updated = true;
		key.SetText( desiredKey );
	}

	if(!active) active = true;
	else active = false;
}

void KeyButton::setNewKey(std::string newKey){

	key.SetText( newKey );
	float keySize = key.GetRect().GetWidth();
	m_paddingR = keySize+20;
	setPosition( m_posX, m_posY );
}




/*
*	List Button
*/







ListButton::ListButton(std::string lab, EventTracker *e, sf::RenderWindow* app){

	type = ButtonType::List;
	m_App = app;
	m_eventTracker = e;
	m_width = Width;
	m_height = Height;
	active = false;
	updated = false;
	m_posX = 0;
	m_posY = 0;
	m_paddingL = 20;
	m_paddingR = 20;
	m_paddingT = 2;
	listDroppedHeight = Height+5;
	currentListVal = "";

	// Set the label
	//if( m_font.LoadFromFile("resources/Helvetica.ttf", LabelSize, lab ) ) {
	//	m_label = sf::String(lab, m_font, LabelSize);
	//}
	//else {
		m_label.SetText( lab );
		m_label.SetSize( LabelSize );
	//}
	m_label.SetColor(m_textColor);

	changeColors(m_borderColor, sf::Color(255, 255, 255, 255), m_highlightColor, m_textColor);
	m_background = createBackground(m_posX, m_posY, m_height, m_width);
	m_droppedBackground = createBackground(m_posX, m_posY, listDroppedHeight, m_width);

	icon.AddPoint(0, 0, sf::Color::Black, sf::Color::Black);
	icon.AddPoint(8, 0, sf::Color::Black, sf::Color::Black);
	icon.AddPoint(4, 4, sf::Color::Black, sf::Color::Black);
	icon.SetOutlineWidth(1.f);
	icon.SetPosition( m_posX+Width-m_paddingR, m_posY+Height/2 );

	activeIcon.AddPoint(0, 0, sf::Color::Black, sf::Color::Black);
	activeIcon.AddPoint(8, 0, sf::Color::Black, sf::Color::Black);
	activeIcon.AddPoint(4, -4, sf::Color::Black, sf::Color::Black);
	activeIcon.SetOutlineWidth(1.f);
	activeIcon.SetPosition( m_posX+Width-m_paddingR, m_posY+Height/2 );
}

void ListButton::draw(){

	if( !active ){
		m_App->Draw( m_background );
		m_App->Draw(icon);
	}
	else {
		m_App->Draw(m_droppedBackground);
		m_App->Draw(activeIcon);

		for(int i=0; i<dropOptions.size(); i++){
			if(currentListVal.compare(dropOptions.at(i).label.GetText())==0){
				dropOptions.at(i).draw(m_App, true);
			}
			else{
				dropOptions.at(i).draw(m_App, false);
			}
		}
	}

	m_App->Draw(m_label);
}

void ListButton::setPosition(float x, float y){

	m_posX = x;
	m_posY = y;

	m_label.SetPosition(m_posX+m_paddingL, m_posY+m_paddingT);
	m_background.SetPosition(m_posX, m_posY);
	m_droppedBackground.SetPosition(m_posX, m_posY);

	icon.SetPosition( m_posX+Width-m_paddingR, m_posY+Height/2-2 );
	activeIcon.SetPosition( m_posX+Width-m_paddingR, m_posY+Height/2-2 );

	for(int i=0; i<dropOptions.size(); i++){
		dropOptions[i].move(m_posX, m_posY+(Height*(i+1)));
	}
}

void ListButton::checkEvent(){

	//highlight();

	switch(m_eventTracker->eventType){

		case EventType::M_Clickleft:{
			onMouseClicked();
		} break;

		case -1:{
		} break;
	}

}

float ListButton::getValue(){

	return dropOptions.size();
}

void ListButton::setDropList( std::vector<std::string> opts ){

	dropOptions.clear();

	for(int i=0; i<opts.size(); i++){

		DropOption newOption(opts.at(i), m_posX, m_posY+(Height*(i+1)));
		dropOptions.push_back(newOption);
	}

	listDroppedHeight = Height*dropOptions.size()+Height+5;
	m_droppedBackground = createBackground(m_posX, m_posY, listDroppedHeight, m_width);

	//if( opts.size() > 0 ) currentListVal = opts[0];
}

sf::Shape ListButton::createListBackground(float x, float y, float height, float width){

	sf::Shape rrect;
	rrect.SetOutlineWidth(2); 
	float radius = 7;

	float x2 = 0;
	float y2 = 0;

	for(int i=0; i<10; i++) { 
		x2 += radius/10; 
		y2 = sqrt(radius*radius - x2*x2); 
		rrect.AddPoint(x2+x+width-radius, y-y2+radius, m_bgColor, m_borderColor); 
	} 

	y2=0; radius = 0;
	for(int i=0; i<10; i++) { 
		y2 += radius/10; 
		x2 = sqrt(radius*radius - y2*y2); 
		rrect.AddPoint(x+width+x2-radius, y+height-radius+y2, m_bgColor, m_borderColor); 
	} 

	x2=0; radius = 0;
	for(int i=0; i<10; i++) {
		x2 += radius/10; 
		y2 = sqrt(radius*radius - x2*x2); 
		rrect.AddPoint(x+radius-x2, y+height-radius+y2, m_bgColor, m_borderColor); 
	}

	y2=0; radius = 7;
	for(int i=0; i<10; i++) { 
		y2 += radius/10; 
		x2 = sqrt(radius*radius - y2*y2); 
		rrect.AddPoint(x-x2+radius, y+radius-y2, m_bgColor, m_borderColor); 
	}

	return rrect; 
}

void ListButton::highlight(){

	if(isMouseOver()) { m_background.SetColor(m_highlightColor); }
	else { m_background.SetColor(m_bgColor); }

	if( active ){
		for(int i=0; i<dropOptions.size(); i++){
			dropOptions.at(i).highlight(m_App);
		}
	}
}

void ListButton::onMouseClicked(){

	if( isMouseOver() ) {

		updated = true;
		if(!active){ active = true; }
		else { active = false; }
	}
	else if( active ) {

		for(int i=0; i<dropOptions.size(); i++){

			if(dropOptions.at(i).mouseClicked(m_App)){

				updated = true;
				currentListVal = dropOptions.at(i).label.GetText();
				active = false;
			}
		}

	} // end if active
}

void ListButton::setDropSelected( std::string str ){

	for(int i=0; i<dropOptions.size(); i++){
		std::string label = dropOptions.at(i).label.GetText();
		if( label.compare( str )==0 ){
			currentListVal = dropOptions.at(i).label.GetText();
		}
	}
}

void ListButton::clearSelected(){
	currentListVal = "";
}

std::string ListButton::getDropSelected(){

	return currentListVal;
}

float ListButton::getHeight(){
	if( !active ){
		return m_height;
	}
	else{
		return listDroppedHeight;
	}
}

void ListButton::setSize(float width, float height){

	m_width = width;
	m_height = height;
	m_background = createBackground(m_posX, m_posY, m_height, m_width);
}


