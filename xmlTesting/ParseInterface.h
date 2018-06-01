#pragma once

#include <string>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/optional.hpp>


namespace pt = boost::property_tree;

class ParseInterface
{

	private:

		//dummy method for compiling should be pure virtual
		virtual void parseValues(const pt::ptree tree) { int x = 5;}

	public:

		template <typename T>
		void parsePrimative(T& val, std::string tag, const pt::ptree tree)
		{
			boost::optional<T> newVal = tree.get_optional<T>(tag);
			if (newVal)
				val = *newVal;
		}

		template <typename T>
		void parseMultiPrimatives(std::vector<T>& vals, std::string tag, const pt::ptree tree)
		{
			pt::ptree::const_iterator end = tree.end();
			/*for (tree::const_iterator it = tree.begin(); it != end; ++it)
			{
				if (it.first == tag)
				{
					T newEle;
					newEle = it.second.get<T>;
					eles.push_back(newEle);
				}
			}*/
		}

		void parseElement(ParseInterface* ele, std::string tag, const pt::ptree tree)
		{
			auto child = tree.get_child_optional(tag);
			
			if (child)
			{
				ele->parseValues(*child);
			}
		}

		template <typename T>
		void parseMultiElements(std::vector<T*>& eles, std::string tag, const pt::ptree tree)
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
};