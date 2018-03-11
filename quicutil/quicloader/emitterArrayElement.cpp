#include "emitterArrayElement.h"

emitterElement* emitterElement::clone() const
{
	emitterElement* de = new emitterElement(name);
	*(de->attributes) = *attributes;
	de->getValuesFromAttributes();
	return de;
}

std::ostream& emitterElement::operator>>(std::ostream& output) const
{
	output << "emitter[" << *attributes << "]" << std::endl;
	return output;
}

std::istream& emitterElement::operator<<(std::istream& input)
{
	parseName(input);
	parseAttributes(input);
	getValuesFromAttributes();
	return input;
}
		
void emitterElement::registerAttributes()
{
	attributes->add(standardAttribute("type", ""));
	attributes->add(standardAttribute("sx", "0"));
	attributes->add(standardAttribute("sy", "0"));
	attributes->add(standardAttribute("sz", "0"));
	attributes->add(standardAttribute("ex", "0"));
	attributes->add(standardAttribute("ey", "0"));
	attributes->add(standardAttribute("ez", "0"));
}

bool emitterElement::getValuesFromAttributes()
{
	std::string att_val;
	std::stringstream ss(std::stringstream::in | std::stringstream::out);
	attributes->getValue("type", att_val); 	ss.str(att_val); ss >> type;  ss.clear();
	attributes->getValue("sx", att_val); 		ss.str(att_val); ss >> sx; ss.clear();
	attributes->getValue("sy", att_val); 		ss.str(att_val); ss >> sy; ss.clear();
	attributes->getValue("sz", att_val); 		ss.str(att_val); ss >> sz; ss.clear();
	attributes->getValue("ex", att_val); 		ss.str(att_val); ss >> ex; ss.clear();
	attributes->getValue("ey", att_val); 		ss.str(att_val); ss >> ey; ss.clear();
	attributes->getValue("ez", att_val); 		ss.str(att_val); ss >> ez; ss.clear();
	return true;
}






emitterArrayElement* emitterArrayElement::clone() const
{
	emitterArrayElement* de = new emitterArrayElement(name);
	de->mttrs = mttrs;
	return de;	
}

int emitterArrayElement::getNumEmitters() const
{
	// Again, do we want to use the size_type type here for returning?
	return static_cast<int>(mttrs.size());
}

emitterElement emitterArrayElement::getEmitter(unsigned int ndx) const
{
	if(ndx < 0 || mttrs.size() <= ndx) {return emitterElement();}
	
	return *mttrs[ndx];
}

std::ostream& emitterArrayElement::operator>>(std::ostream& output) const
{
	std::cout << "# emitter array" << std::endl;
	for(unsigned int i = 0; i < mttrs.size(); i++)
	{
		output << *mttrs[i] << std::flush;
	}
	return output;
}

std::istream& emitterArrayElement::operator<<(std::istream& input)
{
	emitterElement* e = new emitterElement(name);
	input >> *e;
	mttrs.push_back(e);
	return input;
}

