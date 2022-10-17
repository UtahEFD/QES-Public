#include "util/ParseInterface.h"

class ignition : public ParseInterface
{
private:
public:
  float xStart, yStart, length, width, baseHeight, height;

  virtual void parseValues()
  {

    parsePrimitive<float>(true, height, "height");
    parsePrimitive<float>(true, baseHeight, "baseHeight");
    parsePrimitive<float>(true, xStart, "xStart");
    parsePrimitive<float>(true, yStart, "yStart");
    parsePrimitive<float>(true, length, "length");
    parsePrimitive<float>(true, width, "width");
  }
};
