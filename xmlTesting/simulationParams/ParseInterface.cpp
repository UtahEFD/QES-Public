#include "ParseInterface.h"
#include "Root.h"

	ParseInterface::ParseInterface()
	{
		root = 0;
		treeParents = "";
	}

	ParseInterface::ParseInterface(pt::ptree t)
	{
		root = new Root();
		tree = t;
		treeParents = "";
	}

	template <typename T>
	void ParseInterface::parsePrimative(bool isReq, T& val, const std::string tag)
	{
		boost::optional<T> newVal = tree.get_optional<T>(tag);
		if (newVal)
			val = *newVal;
		else
			if (isReq)
				throw(ParseException("Could not find " + treeParents + "::" + tag));
	}

	template <typename T>
	void ParseInterface::parseMultiPrimatives(bool isReq, std::vector<T>& vals, const std::string tag)
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
		if (isReq && vals.size() == 0)
			throw(ParseException("Could not find " + treeParents + "::" + tag));
	}

	template <typename T>
	void ParseInterface::parseElement(bool isReq, T*& ele, const std::string tag)
	{
		auto child = tree.get_child_optional(tag);
		if (child)
		{
			ele = new T();
			ele->setTree(*child);
			ele->setParents(treeParents + "::" + tag);
			ele->parseValues();
		}
		else
			if (isReq)
				throw(ParseException("Could not find " + treeParents + "::" + tag));
	}

	template <typename T>
	void ParseInterface::parseMultiElements(bool isReq, std::vector<T*>& eles, const std::string tag)
	{
		pt::ptree::const_iterator end = tree.end();
		for (pt::ptree::const_iterator it = tree.begin(); it != end; ++it)
		{
			if (it->first == tag)
			{
				T* newEle = new T();
				newEle->setTree(it->second);
				newEle->setParents(treeParents + "::" + tag);
				newEle->parseValues();
				eles.push_back(newEle);
			}
		}
		if (isReq && eles.size() == 0)
			throw(ParseException("Could not find " + treeParents + "::" + tag));
	}

	template <typename T, typename X, typename... ARGS>
	void ParseInterface::parsePolymorph(bool isReq, T*& ele, X poly, ARGS... args)
	{
		auto child = tree.get_child_optional(poly.tag);
		if (child)
		{
			poly.setNewType(ele);
			ele->setTree(*child);
			ele->setParents(treeParents + "::" + poly.tag);
			ele->parseValues();
		}
		else
			parsePolymorph(isReq, ele, args...);
	}


	template <typename T, typename X>
	void ParseInterface::parsePolymorph(bool isReq, T*& ele, X poly)
	{
		auto child = tree.get_child_optional(poly.tag);
		if (child)
		{
			poly.setNewType(ele);
			ele->setTree(*child);
			ele->setParents(treeParents + "::" + poly.tag);
			ele->parseValues();
		}
		else
			if (isReq)
				throw(ParseException("Could not find " + treeParents + "::" + poly.tag));
	}


	template <typename T, typename X, typename... ARGS>
	void ParseInterface::parseMultiPolymorphs(bool isReq, std::vector<T*>& eles, X poly, ARGS... args)
	{
		pt::ptree::const_iterator end = tree.end();
		for (pt::ptree::const_iterator it = tree.begin(); it != end; ++it)
		{
			if (it->first == poly.tag)
			{
				T* newEle;
				poly.setNewType(newEle);
				newEle->setTree(it->second);
				newEle->setParents(treeParents + "::" + poly.tag);
				newEle->parseValues();
				eles.push_back(newEle);
			}
		}
		parseMultiPolymorphs(isReq, eles, args...);
	}


	template <typename T, typename X>
	void ParseInterface::parseMultiPolymorphs(bool isReq, std::vector<T*>& eles, X poly)
	{
		pt::ptree::const_iterator end = tree.end();
		for (pt::ptree::const_iterator it = tree.begin(); it != end; ++it)
		{
			if (it->first == poly.tag)
			{
				T* newEle;
				poly.setNewType(newEle);
				newEle->setTree(it->second);
				newEle->setParents(treeParents + "::" + poly.tag);
				newEle->parseValues();
				eles.push_back(newEle);
			}
		}
		if (isReq && eles.size() == 0)
			throw(ParseException("Could not find " + treeParents + "::" + poly.tag));

	}

	template <typename T>
	void ParseInterface::parseTaglessValues(std::vector<T>& eles)
	{
		std::istringstream buf;
		buf.str( tree.get_value<std::string>() );
		T temp;
		while (buf >> temp)
			eles.push_back(temp);
    	buf.clear();
	}


	void ParseInterface::parseValues() 
	{
		root->setTree(tree);
		root->setParents("root");
		root->parseValues();
	}
