#include "ParseInterface.h"
#include "X.h"

	ParseInterface::ParseInterface()
	{
		xVar = 0;
	}

	template <typename T>
	void ParseInterface::parsePrimative(T& val, const std::string tag, const pt::ptree tree)
	{
		boost::optional<T> newVal = tree.get_optional<T>(tag);
		if (newVal)
			val = *newVal;
	}

	template <typename T>
	void ParseInterface::parseMultiPrimatives(std::vector<T>& vals, const std::string tag, const pt::ptree tree)
	{
		pt::ptree::const_iterator end = tree.end();
		for (pt::ptree::const_iterator it = tree.begin(); it != end; ++it)
		{
			if (it->first == tag)
			{
				T newVal;
				newVal = (it->second).get_value<T>();
				vals.push_back(newVal);
			}
		}
	}

	template <typename T>
	void ParseInterface::parseElement(T*& ele, const std::string tag, const pt::ptree tree)
	{
		auto child = tree.get_child_optional(tag);
		
		if (child)
		{
			ele = new T();
			ele->parseValues(*child);
		}
	}

	template <typename T>
	void ParseInterface::parseMultiElements(std::vector<T*>& eles, const std::string tag, const pt::ptree tree)
	{
		pt::ptree::const_iterator end = tree.end();
		for (pt::ptree::const_iterator it = tree.begin(); it != end; ++it)
		{
			if (it->first == tag)
			{
				T* newEle = new T();
				newEle->parseValues(it->second);
				eles.push_back(newEle);
			}
		}
	}


	void ParseInterface::parseValues(const pt::ptree tree) 
	{
		parseElement<X>(xVar, "X", tree);
	}
