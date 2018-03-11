#include "datam.h"

datam::datam()
{
  name = std::string("");
  found = false;
}

datam::datam(std::string _name)
{
  name = _name;
  found = false;
}

std::string datam::getName() const {return name;}
void datam::setName(std::string const& _name) {name = _name;}

bool datam::foundQ() const {return found;}
void datam::setFound(bool _f) {found = _f;}

std::ostream& operator<<(std::ostream& output, datam const& dtm)
{
  dtm >> output;
  return output;
}

std::istream& operator>>(std::istream& input, datam & dtm)
{
  dtm << input;
  return input;
}

