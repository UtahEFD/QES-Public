/* File: MenuTracker.cpp
 * Author: Matthew Overby
 */

#include "MenuTracker.h"

using namespace SLUI;

MenuTracker::MenuTracker(WindowManager *m_w, EventTracker *m_e, OptionTracker *m_o, sf::RenderWindow *m_a){

	m_winTracker = m_w;
	m_eventTracker = m_e;
	m_optTracker = m_o;
	m_App = m_a;

	curr_posY = 2;
	curr_posX = 2;

	menuBar = sf::Shape::Rectangle(0, 0, m_App->GetWidth(), MenuItem::DefaultHeight, sf::Color(100, 100, 100, 255));

	m_mouseMenu = MenuItem(m_eventTracker, m_App);
}

MenuTracker::~MenuTracker(){

}

void MenuTracker::createMenu(std::string label){

	MenuItem newItem = MenuItem(curr_posX, curr_posY, label, m_eventTracker, m_optTracker, m_winTracker, m_App);
	m_menuItems.insert( std::pair<std::string, MenuItem>( label, newItem ) );
	curr_posX += newItem.getWidth() + 2;
}

void MenuTracker::createMenu( std::string label, std::string command, std::vector<std::string> opts ){

	MenuItem newItem = MenuItem(curr_posX, curr_posY, label, m_eventTracker, m_optTracker, m_winTracker, m_App);
	m_menuItems.insert( std::pair<std::string, MenuItem>( label, newItem ) );
	curr_posX += newItem.getWidth() + 2;

	std::map<std::string, MenuItem>::iterator it = m_menuItems.find(label);
	if( it != m_menuItems.end() ){
		for( int i=0; i<opts.size(); i++ ){
			it->second.addListItem(command, opts[i]);
		}
	}

}

void MenuTracker::addItem(int WINDOW, std::string sub_label, std::string menu_label){

	std::map<std::string, MenuItem>::iterator it = m_menuItems.find(menu_label);

	if( it != m_menuItems.end() ){
		it->second.addWindowItem(WINDOW, sub_label);
	}
}

void MenuTracker::addItem(std::string command, std::string sub_label, std::string menu_label){

	std::map<std::string, MenuItem>::iterator it = m_menuItems.find(menu_label);

	if( it != m_menuItems.end() ){
		it->second.addFuncItem(command, sub_label);
	}
}

void MenuTracker::draw(){

	m_App->Draw(menuBar);

	std::map<std::string, MenuItem>::iterator it;
	for( it = m_menuItems.begin(); it != m_menuItems.end(); it++){
		it->second.draw();
	}

	m_mouseMenu.draw();
}

void MenuTracker::checkEvent(){

	if(!m_optTracker->getActive("hidevisuals")){

		if( m_mouseMenu.dropped ) m_mouseMenu.highlight();
		const sf::Input &input = m_App->GetInput();

		std::map<std::string, MenuItem>::iterator it;
		for( it = m_menuItems.begin(); it != m_menuItems.end(); it++){
			it->second.checkEvent();
			m_mouseMenu.highlight();
		}

		if( m_eventTracker->eventType == EventType::W_Resize ){
			menuBar = sf::Shape::Rectangle(0, 0, m_App->GetWidth(), MenuItem::DefaultHeight, sf::Color(100, 100, 100, 255));
		}

		if( m_eventTracker->eventType == EventType::M_Clickright ){
			if(input.IsKeyDown(sf::Key::LShift)){ 
				m_mouseMenu.mouseClickedRight();
			}
		}

		if( m_eventTracker->eventType == EventType::M_Clickleft ){
			m_mouseMenu.checkEvent();
		}
	}
}

bool MenuTracker::isOpen(){

	std::map<std::string, MenuItem>::iterator it;
	for( it = m_menuItems.begin(); it != m_menuItems.end(); it++){
		if(it->second.dropped){
			return true;
		}
	}

	return false;
}


