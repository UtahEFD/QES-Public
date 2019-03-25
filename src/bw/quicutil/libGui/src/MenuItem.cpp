/* File: MenuItem.cpp
 * Author: Matthew Overby
 */

#include "MenuItem.h"

using namespace SLUI;

MenuItem::MenuItem(){

	type = -1;
}

MenuItem::MenuItem(int posX, int posY, std::string lab, EventTracker *evt, OptionTracker *opt, WindowManager *wnt, sf::RenderWindow *app){

	changeColors(m_borderColor, sf::Color(100, 100, 100, 255), sf::Color(205, 183, 158, 255), sf::Color(255, 255, 255));

	m_eventTracker = evt;
	m_optTracker = opt;
	m_winManager = wnt;
	m_App = app;
	m_posX = posX;
	m_posY = posY;
	m_height = DefaultHeight-4;
	m_width = 0; // set by fitString
	m_paddingT = 2;
	m_paddingB = 2;

	label.SetPosition(m_posX+10, m_posY+2);
	label.SetText(lab);
	label.SetColor(m_textColor);
	label.SetSize(DefaultLabelSize);
	fitString(label);

	m_background = sf::Shape::Rectangle(posX, posY, posX+m_width, posY+m_height, sf::Color(255, 255, 255, 255));
	m_background.SetColor(m_bgColor);

	dropped = false;
	dropIndex.x = posX;
	dropIndex.y = posY+DefaultHeight;
	type = MenuType::Main;
}

MenuItem::MenuItem(EventTracker *evt, sf::RenderWindow *app){

	type = MenuType::Hidden;
	dropped = false;
	dropIndex.x = 0;
	dropIndex.y = 0;
	m_posX = 0;
	m_posY = 0;
	m_height = 0;
	m_width = 0;
	label.SetText("");

	m_App = app;
	m_eventTracker = evt;
}

void MenuItem::draw(){

	m_App->Draw(m_background);
	m_App->Draw(label);
	if(dropped){
		std::map< int, SubItem >::iterator it;
		for(it = m_items.begin(); it != m_items.end(); it++){
			if( it->second.type == MenuType::List ){
				std::string optSelected = it->second.label.GetText();
				if( optSelected.compare( m_optTracker->getListValue(it->second.command) ) == 0 ){
					it->second.setHighlight(true);
				}
			}
			it->second.draw();
		}
	} // end if dropped
}

void MenuItem::addWindowItem(unsigned int val, std::string s){

	SubItem newItem = SubItem(dropIndex.x, dropIndex.y, s, m_App);
	newItem.type = MenuType::Window;
	newItem.window = val;

	dropIndex.y = dropIndex.y + DefaultHeight;
	m_items[m_items.size()] = newItem;

}

void MenuItem::addIndexItem(int val, std::string s){

	SubItem newItem = SubItem(dropIndex.x, dropIndex.y, s, m_App);
	newItem.type = MenuType::Index;
	newItem.index = val;

	dropIndex.y = dropIndex.y + DefaultHeight;
	m_items[m_items.size()] = newItem;

}

void MenuItem::addListItem(std::string command, std::string label){

	SubItem newItem = SubItem(dropIndex.x, dropIndex.y, label, m_App);
	newItem.type = MenuType::List;
	newItem.command = command;

	dropIndex.y = dropIndex.y + DefaultHeight;
	m_items[m_items.size()] = newItem;
}

void MenuItem::addFuncItem(std::string command, std::string label){

	SubItem newItem = SubItem(dropIndex.x, dropIndex.y, label, m_App);
	newItem.type = MenuType::Func;
	newItem.command = command;

	dropIndex.y = dropIndex.y + DefaultHeight;
	m_items[m_items.size()] = newItem;

}

void MenuItem::highlight(){

	const sf::Input &input = m_App->GetInput();

	if(isMouseOver()){
		m_background.SetColor(m_highlightColor);
	}
	else if(!dropped) m_background.SetColor(m_bgColor);

	if(dropped){
		std::map< int, SubItem >::iterator it;
		for(it = m_items.begin(); it != m_items.end(); it++){
			it->second.highlight();
		}
	}
}

void MenuItem::checkEvent(){

	highlight();
	const sf::Input &input = m_App->GetInput();

	switch(m_eventTracker->eventType){

		case EventType::M_Clickleft:{
			mouseClicked();
		} break;

		case EventType::M_Clickright:{
			if(input.IsKeyDown(sf::Key::LShift)){
				mouseClickedRight();
			}
		} break;

		case -1:{
		} break;

	}
}

int MenuItem::getIndexClicked(){

	if( dropped ){
		std::map< int, SubItem >::iterator it;
		for ( it=m_items.begin(); it != m_items.end(); it++ ) {
			if( it->second.isMouseOver() ){
				if( it->second.type == MenuType::Index ){
					return it->second.index;
				}
			} // end sub item clicked
		} // end loop through sub items
	}

	if(isMouseOver()){
		switchDropped();
	}
	else{
		dropped = false;
		m_background.SetColor(m_bgColor);
	}

	return -1;
}

std::string MenuItem::getFuncClicked(){

	std::map< int, SubItem >::iterator it;
	for ( it=m_items.begin(); it != m_items.end(); it++ ) {
		if( it->second.isMouseOver() && it->second.type == MenuType::Func ){
			m_background.SetColor(m_bgColor);
			return it->second.command;
		} // end sub item clicked
	} // end loop through sub items

	return "";
}

void MenuItem::mouseClicked(){

	if( dropped ){
		std::map< int, SubItem >::iterator it;
		for ( it=m_items.begin(); it != m_items.end(); it++ ) {

			if( it->second.isMouseOver() ){

				if( it->second.type == MenuType::Window ){
					int window = it->second.window;
					if(window > 0){
						if(!m_winManager->isOpen(window)) {
							m_winManager->open(window);
						}
						else m_winManager->close(window);
					}
					else if(window == 0){
						m_App->Close();
					}
				}
				else if( it->second.type == MenuType::Func ){
					std::string optClicked = getFuncClicked();
					if( optClicked.size() > 0 ) m_eventTracker->callFunction( optClicked );
				}
				else if( it->second.type == MenuType::List ){

					std::string optClicked = "";
					std::string optLabel = "";
					std::map< int, SubItem >::iterator it;
					for ( it=m_items.begin(); it != m_items.end(); it++ ) {
						if( it->second.isMouseOver() && it->second.type == MenuType::List ){
							m_background.SetColor(m_bgColor);
							optClicked = it->second.command;
							optLabel = it->second.label.GetText();
						} // end sub item clicked
					} // end loop through sub items

					if( optClicked.size() > 0 ) m_optTracker->setListValue( optClicked, optLabel );
				}

			} // end sub item clicked

		} // end loop through sub items
	}

	if(isMouseOver()){
		switchDropped();
	}
	else{
		dropped = false;
		m_background.SetColor(m_bgColor);
	}
}

void MenuItem::mouseClickedRight(){

	if( type == MenuType::Hidden ){
		switchDropped();
		if( dropped ){
			move(m_eventTracker->oldPos.x, m_eventTracker->oldPos.y);
		}
	}
}

void MenuItem::switchDropped(){

	if(dropped == true){
		m_background.SetColor(m_bgColor);
		dropped = false;
	}
	else {
		m_background.SetColor(m_highlightColor);
		dropped = true;
	}
}

void MenuItem::setDropped(bool drop){

	if(drop){
		m_background.SetColor(m_highlightColor);
		dropped = true;
	}
	else{
		m_background.SetColor(m_bgColor);
		dropped = false;
	}
}

void MenuItem::setLabel(std::string newLabel){

	label.SetPosition(m_posX+10, m_posY+2);
	label.SetText(newLabel);
	label.SetColor(m_textColor);
	label.SetSize(DefaultLabelSize);
	fitString(label);
}

void MenuItem::setSubItemLabel(int index, std::string newLabel){

	std::map<int,SubItem>::iterator it = m_items.find( index );
	if( it != m_items.end() ){
		it->second.label.SetText(newLabel);
	}
}

void MenuItem::move(float newX, float newY){

	m_posX = newX;
	m_posY = newY;
	std::map< int, SubItem >::iterator it;
	for ( it=m_items.begin(); it != m_items.end(); it++ ) {
		it->second.move( newX, newY );
		newY += MenuItem::DefaultHeight;
	}
}

void MenuItem::fitString(sf::String str){

	std::string str1 = str.GetText();
	sf::String tempStr;
	tempStr.SetPosition(10, m_posY);
	tempStr.SetSize(str.GetSize());
	tempStr.SetText(str.GetText());

	if(tempStr.GetRect().GetWidth()+m_paddingL*2 > m_width){
		m_width = tempStr.GetRect().GetWidth()+m_paddingL*2;
	}
}

/*
*	SUBITEM FUNCTIONS
*/


SubItem::SubItem(){
	type = -1;
}

SubItem::SubItem(int posX, int posY, std::string lab, sf::RenderWindow *app){
	changeColors(m_borderColor, sf::Color(100, 100, 100, 255), sf::Color(205, 183, 158, 255), sf::Color(255, 255, 255));

	m_App = app;
	m_posX = posX;
	m_posY = posY;
	m_height = MenuItem::DefaultHeight;
	m_width = MenuItem::DefaultWidth;
	m_paddingT = 2;
	m_paddingB = 2;
	label.SetPosition(m_posX+10, m_posY+2);
	label.SetText(lab);
	label.SetColor(m_textColor);
	label.SetSize(MenuItem::DefaultLabelSize);
	type = -1;
	window = -1;
	command = "";

	m_background = sf::Shape::Rectangle(posX, posY, posX+m_width, posY+m_height, sf::Color(255, 255, 255, 255));
	m_background.SetColor(m_bgColor);
}

void SubItem::draw(){

	m_App->Draw(m_background);
	m_App->Draw(label);
}

void SubItem::move( float newX, float newY ){

	float moveY = newY - m_posY;
	m_posX = newX;
	m_posY = newY;
	label.SetPosition(m_posX+10, m_posY+2);
	m_background.SetX(m_posX);
	m_background.Move(0, moveY);
}

void SubItem::highlight(){

	if(isMouseOver()){
		m_background.SetColor(m_highlightColor);
	}
	else{
		m_background.SetColor(m_bgColor);
	}
}


