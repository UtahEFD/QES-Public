/* File: Console.cpp
 * Author: Matthew Overby
 */

#include "Console.h"

using namespace SLUI;

Message::Message(sf::RenderWindow *app){

	m_App = app;
	text.SetPosition(4, m_App->GetHeight()-26);
	text.SetColor(sf::Color(0,0,0, 255));
	text.SetSize(18);
	text.SetText("lala");

	show = false;
	clock.Reset();
	displayTime = 0;
}

void Message::draw(){

	m_App->Draw(text);
	if(displayTime < clock.GetElapsedTime()) show = false;
}

Console::Console(int newId, WindowController *wnC, OptionTracker *opT, EventTracker *evt, sf::RenderWindow* app){

	int w = app->GetWidth();
	int h = app->GetHeight();
	id = newId;
	isOpen = false;

	m_App = app;
	m_optTracker = opT;
	m_winController = wnC;
	m_eventTracker = evt;
	m_width = w;
	m_height = 200;
	m_posX = 0;
	m_posY = h-m_height;
	logStyle = sf::String("");
	logStyle.SetSize(14);
	logStyle.SetColor(sf::Color::Black);

	input.SetPosition(4, m_posY+4);
	input.SetColor(sf::Color(0,0,0, 255));
	input.SetSize(18);
	clearInput();

	m_background = createBackground(m_posX, m_posY, m_height, m_width);

	inputBar.Resize(m_width-4, 30-4);
	inputBar.SetPosition(2, m_posY+4);
	inputBar.SetColor(sf::Color(102, 102, 102, 100));

	textBox = TextBox(m_width-4, m_height-30, m_posX+2, m_posY+30, m_App);
	textBox.setTextFormat(logStyle);

	m_message = new Message(m_App);
	messageBg = sf::Shape::Rectangle(0, h-30, w, h, sf::Color(255, 255, 255, 50));
}

Console::~Console(){

	delete m_message;
}

void Console::addInput(char newChar){

	if((int)newChar == -1) {
		std::string temp = input.GetText();
		temp.resize(temp.length()-1);
		input.SetText(temp);
	}
	else {
		std::string temp = input.GetText();
		input.SetText(temp + newChar);
	}

}

std::string Console::getInput(){

	std::string temp = input.GetText();
	return temp.substr(3, temp.length()-3);
}

void Console::enterInput(std::string str){

	std::string tempLog = textBox.getText();
	textBox.setText(str+"\n"+tempLog);
}

void Console::clearInput(){

	input.SetText(">  ");
}

void Console::showMessage(std::string str, int t){

	m_message->show = true;
	m_message->displayTime = t;
	m_message->clock.Reset();
	m_message->text.SetText(str);

	enterInput(str);
}

void Console::draw(){

	if(isOpen){
		m_App->Draw(m_background);
		m_App->Draw(inputBar);
		m_App->Draw(input);
		textBox.draw();
	}
	else if(m_message->show){
		m_App->Draw(messageBg);
		m_message->draw();
	}
}

void Console::close(){

	isOpen = false;
	clearInput();
	m_winController->close(id);
}

void Console::open(){

	isOpen = true;
	m_winController->open(id);
}

void Console::resizeEvent(){

	m_width = m_App->GetWidth();
	float h = m_App->GetHeight();
	m_height = 200;
	m_posY = h-m_height;

	input.SetPosition(4, h-m_height+2);
	m_background = createBackground(m_posX, m_posY, m_height, m_width);

	inputBar.Resize(m_width-4, 30-4);
	inputBar.SetPosition(2, h-m_height+2);

	std::string tempText = textBox.getText();
	textBox = TextBox(m_width-4, m_height-30, m_posX+2, m_posY+30, m_App);
	textBox.setTextFormat(logStyle);
	textBox.setText(tempText);

	m_message->text.SetPosition(4, h-26);
	messageBg = sf::Shape::Rectangle(0, h-30, m_width, h, sf::Color(255, 255, 255, 50));
}

void Console::highlight(){

	const sf::Input &input = m_App->GetInput();
	textBox.highlight(&input);
}

void Console::mouseClicked() {

	const sf::Input &input = m_App->GetInput();
	textBox.mouseClicked(&input);
}

void Console::scroll(bool up){

	textBox.scroll(up);
}

void Console::handleKeyPressed(){

	sf::Event *event = m_eventTracker->lastEvent;
	sf::Key::Code keyCode = event->Key.Code;
	const sf::Input& input = m_App->GetInput();

	if ( (	event->Text.Unicode < 0x80 || 
		keyCode == sf::Key::Equal || 
		keyCode == sf::Key::Period ||
		keyCode == sf::Key::Space ||
		keyCode == sf::Key::Dash ||
		( keyCode == sf::Key::SemiColon )
	 ) ) {
		/*
		*	Text Entered
		*/

		// Character
		char newChar = (char)event->Text.Unicode;
		if(keyCode == sf::Key::Period) newChar = '.';
		else if(keyCode == sf::Key::Equal) newChar = '=';
		else if(keyCode == sf::Key::Space) newChar = ' ';
		else if(keyCode == sf::Key::Dash) newChar = '-';
		else if(keyCode == sf::Key::SemiColon) newChar = ':';
		else if(keyCode == sf::Key::LBracket || keyCode == sf::Key::RBracket) newChar = 0x16;

		if(newChar != 0x16 && newChar != 0x14) addInput(newChar);

	} // end Text Entered
	else if( keyCode == sf::Key::Return ){

		/*
		*	Enter Pressed
		*/

		std::string input = getInput();
		if(input.length() > 0){

			std::stringstream temp(input);
			std::string command, option, action;
			temp >> command >> option >> action;

			if(command.compare("bind")==0){
				std::string desiredKey = option;
				if(desiredKey.length() > 0){
					if(!m_optTracker->bind(action, desiredKey)){
						input = "Bind Usage:  bind <key> <command>";
					}
					else{
						m_eventTracker->callFunction( "updatemenus" );
					}
				}
				else{
					input = "Bind Usage:  bind <key> <command>";
				}
			}
			else if(command.compare("set")==0){

				std::string value = action;
				if(option.length() > 1 && value.length() > 1 ) {
					m_optTracker->setString(option, value);
					m_eventTracker->callFunction( "updatemenus" );
				}
				else{
					input = "Set Usage:  set <option> <value>";
				}
			}
			else if(command.compare("quit")==0 || command.compare("exit")==0){
				m_eventTracker->callFunction( "updateconfig" );
				m_App->Close();
			}
			else if(command.compare("help")==0){
				input = "Console Usage:\n> set <option> <value>\n> bind <key> <action>\n> show options\n> show actions\n> <option>\n";
			}
			else if(command.compare("show")==0){

				if(option.compare("options")==0) {
					input = "Option List";
					std::map<std::string, Option*> tempOptions = m_optTracker->getOptions();
					for(std::map<std::string, Option*>::iterator it = tempOptions.begin(); it != tempOptions.end(); it++){
						input += "\n> "+it->first;
					}
					input += "\n";
				}
				else if(option.compare("actions")==0) {
					input = "Action List";
					std::map<std::string, Keys*> tempKeys = m_optTracker->getKeys();
					for(std::map<std::string, Keys*>::iterator it = tempKeys.begin(); it != tempKeys.end(); it++){
						input += "\n> "+it->first;
					}
					input += "\n";
				}

			}
			else {
				std::string val = m_optTracker->getString(command);
				if(val.length() > 0) input = command+"="+val;
			}
		}

		enterInput(input);
		clearInput();

	} // end Enter Pressed
	else if ( keyCode == sf::Key::Back) {

		/*
		*	Backspace Pressed
		*/

		std::string temp = getInput();
		if(temp.length() > 0){	
			char newChar = (char)(-1);
			addInput(newChar);
		}

	} // end Backspace Pressed

}

void Console::checkEvent(){

	switch(m_eventTracker->eventType){

		case EventType::K_Press:{
			handleKeyPressed();
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

