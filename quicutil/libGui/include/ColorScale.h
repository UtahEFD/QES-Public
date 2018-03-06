/*
 *  ColorScale.h
 *
 *  Created by mooglwy on 31/10/10.
 *
 *  This software is provided 'as-is', without any express or
 *  implied warranty. In no event will the authors be held
 *  liable for any damages arising from the use of this software.
 *  
 *  Permission is granted to anyone to use this software for any purpose,
 *  including commercial applications, and to alter it and redistribute
 *  it freely, subject to the following restrictions:
 *  
 *  1. The origin of this software must not be misrepresented;
 *     you must not claim that you wrote the original software.
 *     If you use this software in a product, an acknowledgment
 *     in the product documentation would be appreciated but
 *     is not required.
 *  
 *  2. Altered source versions must be plainly marked as such,
 *     and must not be misrepresented as being the original software.
 *  
 *  3. This notice may not be removed or altered from any
 *     source distribution.
 *
 */
 
#ifndef _COLOR_SCALE_
#define _COLOR_SCALE_
#include <map>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics.hpp>
 
namespace InterpolationFunction 
{
	enum InterpolationFunction
	{
		Linear,
		Cosinus
	};
}
 
namespace GradientStyle 
{
	enum GradientStyle 
	{
		Linear,
		Circle,
		Radial,
		Reflex
	};
}
 
class ColorScale : protected std::map<double,sf::Color>
{
public:
 
	ColorScale();
    
    ColorScale(std::string colorScaleName);
    
    static std::vector<std::string> getColorScaleNames();
    
    sf::Color colorAtUnitPosition(double val) const;

	bool insert(double position, sf::Color color);
 
	void fillTab(sf::Color* colorTab, int size, 
		InterpolationFunction::InterpolationFunction function = InterpolationFunction::Cosinus) const;

	void draw(sf::Image&,const sf::Vector2f& start,const sf::Vector2f& end,
		GradientStyle::GradientStyle style=GradientStyle::Linear, int size = 500) const;
    
    static void loadColorScales();
    
    static void resetColorScales();
 
private:
    static std::map<std::string, ColorScale> colorScales;
    static std::vector<std::string> colorScaleNames;
};

 
#endif //end of _COLOR_SCALE_
