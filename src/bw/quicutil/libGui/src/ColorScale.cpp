/*
 *  ColorScale.cpp
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
 
#include "ColorScale.h"
#include <math.h>
#include "pugixml.hpp"
#include "util/ResourcesHandler.h"
#include <iostream>
 
const double PI = 3.1415926535;

std::map<std::string, ColorScale> ColorScale::colorScales;
std::vector<std::string> ColorScale::colorScaleNames;

double linearInterpolation(double v1, double v2, double mu)
{
	return(v1*(1-mu)+v2*mu);
}
 
double interpolateCosinus(double y1, double y2, double mu)
{
	double mu2;
 
	mu2 = (1-cos(mu*PI))/2;
	return (y1*(1-mu2)+y2*mu2);
}
 
sf::Color GradientLinear(sf::Color* colorTab,int size,const sf::Vector2f& start,const sf::Vector2f& end,int x,int y)
{
	sf::Vector2f dir  = end-start;
	sf::Vector2f pix  = sf::Vector2f(x,y)-start; 
	double dotProduct = pix.x*dir.x+pix.y*dir.y;
	dotProduct       *= (size-1)/(dir.x*dir.x+dir.y*dir.y);
 
	if((int)dotProduct < 0.0      ) return colorTab[0];
	if((int)dotProduct > (size-1) ) return colorTab[size-1];
	return colorTab[(int)dotProduct];
}
 
sf::Color GradientCircle(sf::Color* colorTab,int size,const sf::Vector2f& start,const sf::Vector2f& end,int x,int y)
{
	sf::Vector2f v_radius  = end-start;
	double radius          = sqrt(v_radius.x*v_radius.x+v_radius.y*v_radius.y);
	sf::Vector2f pix       = sf::Vector2f(x,y)-start;
	double dist            = sqrt(pix.x*pix.x+pix.y*pix.y);
	dist                  *= (size-1)/radius;
 
	if((int)dist < 0.0      ) return colorTab[0];
	if((int)dist > (size-1) ) return colorTab[size-1];
	return colorTab[(int)dist];
}
 
sf::Color GradientRadial(sf::Color* colorTab,int size,const sf::Vector2f& start,const sf::Vector2f& end,int x,int y)
{
	sf::Vector2f base     = end-start;
	base                 /= (float)sqrt(base.x*base.x+base.y*base.y);
	sf::Vector2f pix      = sf::Vector2f(x,y)-start;
	pix                  /= (float)sqrt(pix.x*pix.x+pix.y*pix.y);
	double angle          = acos(pix.x*base.x+pix.y*base.y);
	double aSin           = pix.x*base.y-pix.y*base.x;
	if( aSin < 0) angle   = 2*PI-angle;
	angle                *= (size-1)/(2*PI);
 
 
	if((int)angle < 0.0      ) return colorTab[0];
	if((int)angle > (size-1) ) return colorTab[size-1];
	return colorTab[(int)angle];
}
 
sf::Color GradientReflex(sf::Color* colorTab,int size,const sf::Vector2f& start,const sf::Vector2f& end,int x,int y)
{
	sf::Vector2f dir  = end-start;
	sf::Vector2f pix  = sf::Vector2f(x,y)-start; 
	double dotProduct = pix.x*dir.x+pix.y*dir.y;
	dotProduct       *= (size-1)/(dir.x*dir.x+dir.y*dir.y);
	dotProduct        = (dotProduct>0)?dotProduct:-dotProduct;
 
	if((int)dotProduct < 0.0      ) return colorTab[0];
	if((int)dotProduct > (size-1) ) return colorTab[size-1];
	return colorTab[(int)dotProduct];
}
 
ColorScale::ColorScale()
{
}

ColorScale::ColorScale(std::string colorScaleName) {
    if (colorScales.count(colorScaleName) != 1) {
        std::cerr << "ColorScale::ColorScale(std::string): Unable to find color scale with name " << colorScaleName << std::endl;
        return;
    }
    const ColorScale& other = colorScales[colorScaleName];
    for (ColorScale::const_iterator it = other.begin(); it != other.end(); ++it) {
        insert(it->first, it->second);
    }
}

std::vector<std::string> ColorScale::getColorScaleNames() {
    if (colorScaleNames.size() == 0) {
        loadColorScales();
    }
    return colorScaleNames;
}

sf::Color ColorScale::colorAtUnitPosition(double val) const {
    if (size() == 0) {
        return sf::Color(255,0,0);
    } if (size() == 1) {
        return begin()->second;
    } if (val <= 0) {
        return begin()->second;
    } if (val >= 1) {
        return (--end())->second;
    }
    double min = begin()->first;
    double max = (--end())->first;
    double remapVal = val * (max-min) + min;
    const_iterator next = begin();

    while (next != end() && next->first < remapVal) {
        ++next;
    }
    
    const_iterator prev = next;
    --prev;
    
    if (prev->first > remapVal || next->first < remapVal) {
        std::cerr << "ColorScale::colorAtUnitPosition(double): error in finding between gradient point" << std::endl;
    }
    
    double blend = (remapVal - prev->first) / (next->first - prev->first);
    
    sf::Color ans;
    ans.r = linearInterpolation(prev->second.r, next->second.r, blend);
    ans.g = linearInterpolation(prev->second.g, next->second.g, blend);
    ans.b = linearInterpolation(prev->second.b, next->second.b, blend);
    ans.a = linearInterpolation(prev->second.a, next->second.a, blend);
    
    return ans;
    
}

bool ColorScale::insert(double position, sf::Color color)
{
	std::pair< ColorScale::iterator,bool > ret = std::map<double,sf::Color>::insert(std::pair<double, sf::Color>(position,color));
	return ret.second;
}
 
 
#define ABS(a) ((a>0)?(a):0)
 
void ColorScale::fillTab(sf::Color* colorTab, int size,InterpolationFunction::InterpolationFunction function) const
{
 
	ColorScale::const_iterator start = std::map<double,sf::Color>::begin();
	ColorScale::const_iterator last  = std::map<double,sf::Color>::end();
	last--;
 
	double pos = 0.0;
	double distance = last->first - start->first;
	ColorScale::const_iterator it =  start;
 
	double(*pFunction)(double,double,double);
 
	switch (function) 
	{
		case InterpolationFunction::Cosinus: pFunction = interpolateCosinus;  break;
		case InterpolationFunction::Linear : pFunction = linearInterpolation; break;
		default: pFunction = interpolateCosinus;  break;
 
	}
	while(it!=last)
	{
		sf::Color startColor = it->second;
		double    startPos   = it->first;
		it++;
		sf::Color endColor   = it->second;
		double    endPos     = it->first;
		double nb_color         = ((endPos-startPos)*(double)size/distance);
 
		for(int i = (int)pos;i<=(int)(pos+nb_color);i++)
		{
			colorTab[i].r = (unsigned char)pFunction(startColor.r,endColor.r,ABS((double)i-pos)/(nb_color-1.0));
			colorTab[i].g = (unsigned char)pFunction(startColor.g,endColor.g,ABS((double)i-pos)/(nb_color-1.0));
			colorTab[i].b = (unsigned char)pFunction(startColor.b,endColor.b,ABS((double)i-pos)/(nb_color-1.0));
			colorTab[i].a = (unsigned char)pFunction(startColor.a,endColor.a,ABS((double)i-pos)/(nb_color-1.0));
            colorTab[i] = colorAtUnitPosition(double(i)/(size - 1));
		}
		pos+=nb_color;
	}
 
}
 
#undef ABS
 
void ColorScale::draw(sf::Image& img,const sf::Vector2f& start,const sf::Vector2f& end,GradientStyle::GradientStyle style, int size) const
{
 
	sf::Color (*pFunction)(sf::Color*,int,const sf::Vector2f&,const sf::Vector2f&,int,int);
 
	sf::Color* tab =new sf::Color[size];
	fillTab(tab,size);
 
	switch (style) 
	{
		case GradientStyle::Linear : pFunction = GradientLinear; break;
		case GradientStyle::Circle : pFunction = GradientCircle; break;
		case GradientStyle::Radial : pFunction = GradientRadial; break;
		case GradientStyle::Reflex : pFunction = GradientReflex; break;
 
		default: pFunction = GradientLinear;  break;
	}
 
	for(int i=0;i<img.GetWidth();i++)
	{
		for(int j=0;j<img.GetHeight();j++)
		{
			img.SetPixel(i,j,pFunction(tab,size,start,end,i,j));
		}		
	}
	delete[] tab;
}

void ColorScale::loadColorScales() {
    const bool PARANOID=true;
    const std::string FUNCTION_NAME = "ColorScale::loadColorScales(std::string): ";
    
    std::string xmlFilename = sivelab::ResourcesHandler::find("ColorMaps.xml");
    if (xmlFilename == "") {
        std::cerr << FUNCTION_NAME << "Unable to find ColorMaps.xml";
    }
    
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(xmlFilename.c_str());
    if (!result) {
        std::cerr << FUNCTION_NAME << "Error loading file " << xmlFilename << std::endl;
    }
    if (PARANOID) {
        // Count to make sure document node (which isn't an actual xml node)
        // only contains one child: "ColorMaps"
        int count = 0;
        for (pugi::xml_node::iterator it = doc.begin(); it != doc.end(); ++it) {
            count++;
        }
        if (count != 1) {
            std::cerr << FUNCTION_NAME << "XML document with " << count << " root nodes instead of 1" << std::endl;
        }
        if (doc.first_child().name() != std::string("ColorMaps")) {
            std::cerr << FUNCTION_NAME << "XML root node has name " << doc.first_child().name() << " instead of ColorMaps";
        }
    } // End paranoid checks
    
    pugi::xml_node colormaps_node = doc.first_child();
    
    for (pugi::xml_node::iterator colormap_it = colormaps_node.begin(); colormap_it != colormaps_node.end(); ++colormap_it) {
        
        ColorScale tempCS;
    
        pugi::xml_node colormap_node = *colormap_it;
        
        if (colormap_node.name() != std::string("ColorMap")) {
            std::cerr << FUNCTION_NAME << "XML node with name " << colormap_node.name() << " instead of ColorMap" << std::endl;
        }
        for (pugi::xml_node::iterator it = colormap_node.begin(); it != colormap_node.end(); ++it) {
            pugi::xml_node pt = *it;
            if (pt.name() != std::string("Point")) {
                continue;
            }
            double r = pt.attribute("r").as_double();
            double g = pt.attribute("g").as_double();
            double b = pt.attribute("b").as_double();
            double x = pt.attribute("x").as_double();
            
            uint8_t ir, ig, ib;
            ir = r * 255;
            ig = g * 255;
            ib = b * 255;
            tempCS.insert(x, sf::Color(ir, ig, ib));
        }
        
        std::string colorScaleName = colormap_node.attribute("name").as_string();
        colorScaleNames.push_back(colorScaleName);
        colorScales.insert(std::make_pair(colorScaleName, tempCS));
    }
    
}

void ColorScale::resetColorScales() {
    colorScales.clear();
    colorScaleNames.clear();
    
}

