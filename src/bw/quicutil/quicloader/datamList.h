#ifndef DATAMLIST
#define DATAMLIST

#include <iostream>
#include <sstream>

#include "datam.h"

class datamList
{
	public:
	
		datamList();
		datamList(datamList const&);
		virtual ~datamList();
	
		bool isEmptyQ() const;
		void clear();
		
		void remove(std::string const&);

		datamList& operator=(datamList const&);

		friend std::ostream& operator<<(std::ostream& output, datamList const&);
		//friend std::istream& operator>>(std::istream& input, datamList&);
		
		void print() const;	
			
	protected:
			
		void unshift(datam*);
		datam* shift();
		
		void push(datam*);
		datam* pop();
			
		datam* search(std::string const&) const;
	
	private:
	
		typedef struct datamnode
		{
			datam* dtm;
			datamnode* nxt;
		} datamnode;
		
		datamnode* head;		
};

std::ostream& operator<<(std::ostream& output, datamList const&);
//std::istream& operator>>(std::istream& input, datamList&);

#endif

