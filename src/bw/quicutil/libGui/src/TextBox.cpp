/* File: TextBox.cpp
 * Author: Matthew Overby
 */

#include "TextBox.h"

using namespace SLUI;

TextBox::TextBox(){}

TextBox::TextBox(int w, int h, int px, int py, sf::RenderWindow* app){

	m_App = app;
	m_height = h;
	m_width = w;
	m_posX = px;
	m_posY = py;
	fullText = "";

	scrollWidth = 15;
	txtPadding = 15;
	lineIndex = linesTotal = 0;

	text = sf::String("");
	text.SetPosition(m_posX+txtPadding, m_posY+txtPadding);
	text.SetColor(sf::Color(0,0,0));
	text.SetSize(18);

	scrollbar = Scrollbar(m_posX+m_width-scrollWidth, m_posY, m_height, m_App);

	linesVisible = m_height/text.GetSize();

	float txtHeight = text.GetRect().GetHeight();
	text = chopText(fullText, m_height-(txtPadding*2));
}

TextBox::~TextBox(){}

void setPosition(int posX, int posY){
	// TODO
}

void setSize(int width, int height){
	// TODO
}


void TextBox::loadText(std::string filename){

	linesTotal = 0;
	std::stringstream result("");
	sf::String tempTxt = text;
	tempTxt.SetText("");
	fullText = "";

	if(checkFile(filename.c_str())){

		std::ifstream textFile;
		textFile.open(filename.c_str());

		if(textFile.is_open()) {

			std::string line = "";
			std::string newWord = "";

			while (getline(textFile, line)) {

				std::stringstream ss(line);
				while(ss.good()){

					newWord = " ";
					ss >> newWord;
					tempTxt.SetText(result.str()+newWord+" ");
					if(tempTxt.GetRect().GetWidth() > m_width-txtPadding*2-scrollWidth){
						result << "\n";
						linesTotal++;
					}
					result << newWord + " ";
				}
				result << "\n";
				linesTotal++;
			}
		}

		textFile.close();
	}
	else{
		result << "Text file \"" << filename << "\" could not be loaded";
	}

	fullText = result.str();
	scrollbar.makeSlider(linesTotal, linesVisible);

	float txtHeight = text.GetRect().GetHeight();
	text = chopText(fullText, m_height-(txtPadding*2));
}

void TextBox::setText(std::string newText){

	linesTotal = 0;
	std::string result = "";
	sf::String tempTxt = text;
	tempTxt.SetText("");
	fullText = "";

	std::string line = "";
	std::string newWord = "";
	std::stringstream newT(newText);

	while (getline(newT, line)) {

		std::stringstream ss(line);
		while(ss.good()){

			newWord = " ";
			ss >> newWord;
			tempTxt.SetText(result+newWord+" ");
			if(tempTxt.GetRect().GetWidth() > m_width-txtPadding*2-scrollWidth){
				result += "\n";
				linesTotal++;
			}
			result += newWord + " ";
		}
		result += "\n";
		linesTotal++;
	}

	fullText = result;
	scrollbar.makeSlider(linesTotal, linesVisible);

	float txtHeight = text.GetRect().GetHeight();
	text = chopText(fullText, m_height-(txtPadding*2));
}

std::string TextBox::getText(){

	return fullText;
}

bool TextBox::mouseOverScrollbar(){

	return scrollbar.isMouseOverBar();
}

void TextBox::setTextFormat(sf::String newText){

	text.SetColor(newText.GetColor());
	text.SetSize(newText.GetSize());

	float txtHeight = text.GetRect().GetHeight();
	text = chopText(fullText, m_height-(txtPadding*2));
}

void TextBox::highlight(const sf::Input *input) {

	scrollbar.highlight(input);
}

void TextBox::mouseClicked(const sf::Input *input) {

	if(scrollbar.isMouseOverUp(input)) {
		scroll(true);
	}
	else if(scrollbar.isMouseOverDown(input)) {
		scroll(false);
	}
}

void TextBox::scroll(bool up){

	if(linesTotal > 0){
		if(up && lineIndex > 0){
			scrollbar.scroll(true);
			lineIndex--;

			float txtHeight = text.GetRect().GetHeight();
			text = chopText(fullText, m_height-(txtPadding*2));
		}
		else if(!up && lineIndex <= linesTotal-linesVisible){
			scrollbar.scroll(false);
			lineIndex++;

			float txtHeight = text.GetRect().GetHeight();
			text = chopText(fullText, m_height-(txtPadding*2));
		}
	}
}

void TextBox::scroll(int diff){

}

void TextBox::draw(){
	
	m_App->Draw(text);
	scrollbar.draw();
}

sf::String TextBox::chopText(std::string newText, int height){

	std::stringstream ss(newText);
	std::string result = "";
	std::string line = "";
	sf::String tempTxt = text;
	tempTxt.SetText("");
	int currentIndex = 1;

	while (getline(ss, line)) {
		if(currentIndex > lineIndex && tempTxt.GetRect().GetHeight() < height){
			result += line + "\n";
			tempTxt.SetText(result);
		}
		currentIndex++;
	}

	return tempTxt;
}

bool TextBox::checkFile(std::string filename){

	FILE *f;
	f = fopen(filename.c_str(), "rb");
	if(f == NULL) { return false; }
	return true;
}

void TextBox::checkEvent(){

	//switch(m_eventTracker->eventType){
	switch(-1){

		case EventType::W_Resize:{
			//resizeEvent();
		} break;

		case EventType::M_Dragleft:{
		} break;

		case EventType::M_Scrollup:{
			//scroll(true);
		} break;

		case EventType::M_Scrolldown:{
			//scroll(false);
		} break;

		case EventType::M_Clickleft:{
			//mouseClicked();
		} break;

		case -1:{
		} break;
	}
}





