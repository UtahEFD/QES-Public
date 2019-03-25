#include "datamList.h"

datamList::datamList() 
{
	head = NULL; 
}

datamList::datamList(datamList const& dl)
{
	head = NULL;
	datamnode* trvsl = dl.head;
	while(trvsl != NULL)
	{
		this->push(trvsl->dtm->clone());
		trvsl = trvsl->nxt;
	}
}

datamList::~datamList()
{
	datamnode* victim = 0;
	while(head != NULL)
	{
		victim = head;
		head   = head->nxt;
	
		//delete victim->dtm; 
		victim->dtm = 0; victim->nxt = 0;
		delete victim;
	}
}

bool datamList::isEmptyQ() const {return (head == NULL);}

void datamList::clear() 
{
	while(!isEmptyQ())
	{
		shift();
	}
}

void datamList::remove(std::string const& name)
{
	datamnode* prior  = 0;
	datamnode* trvsl  = 0;
	datamnode* victim = 0;

	if(head == NULL) {return;}

	if(name == head->dtm->getName())
	{
		victim 	= head;
		head 	= head->nxt;
	
		//delete victim->dtm; 
		victim->dtm = 0; victim->nxt = 0;
		delete victim;
	}

	if(head == NULL) {return;}

	trvsl = head->nxt;
	prior = head;

	while(trvsl != NULL)
	{
		if(name == trvsl->dtm->getName())
		{
			victim 	   = trvsl;
			prior->nxt = trvsl->nxt;
			
			//delete victim->dtm; 
			victim->dtm = 0; victim->nxt = 0;
			delete victim;
		}
	}
}

datamList& datamList::operator=(datamList const& dl)
{
	if(this == &dl) {return *this;}
	
	this->clear();
	
	head = NULL;
	datamnode* trvsl = dl.head;	
	while(trvsl != NULL)
	{
		this->push(trvsl->dtm->clone());
		trvsl = trvsl->nxt;
	}

	return *this;	
}

void datamList::print() const
{
	datamnode* trvsl = head;
	while(trvsl != NULL) 
	{
		if(trvsl->dtm == NULL) {std::cout << "traversal datam null" << std::endl;}
		std::cout << (*trvsl->dtm) << std::endl;
		trvsl = trvsl->nxt;
	}
}

void datamList::unshift(datam* _d)
{
	// datam's could contain some horrendous amount of memory, which
	// doesn't need to be copied and then why the method is protected.
	
	datamnode* unshiftee = new datamnode;
	unshiftee->dtm = _d;
	unshiftee->nxt = head;
	
	head = unshiftee;
}

datam* datamList::shift()
{
	// datam's could contain some horrendous amount of memory, which
	// doesn't need to be copied and then why the method is protected.
	if(head == NULL) {return NULL;}

	datamnode* shiftee = head;
	head = head->nxt;
	
	datam* top = shiftee->dtm;
	shiftee->dtm = 0;
	shiftee->nxt = 0;
	delete shiftee; shiftee = 0;
	
	return top;
}

void datamList::push(datam* _d)
{
	// datam's could contain some horrendous amount of memory, which
	// doesn't need to be copied and then why the method is protected.
	datamnode* noob = new datamnode;
	noob->dtm = _d;
	noob->nxt = NULL;
	
	if(head == NULL) 
	{
		head = noob; 
		return;
	}

	datamnode* trvsl = head;
	while(trvsl->nxt != NULL) {trvsl = trvsl->nxt;}

	trvsl->nxt = noob;
}

datam* datamList::pop()
{
	
	// datam's could contain some horrendous amount of memory, which
	// doesn't need to be copied and then why the method is protected.
	if(head == NULL) {return NULL;}
	
	datamnode* popee = head;
	datamnode* trvsl = head->nxt;

	while(trvsl != NULL)
	{
		popee = popee->nxt;
		trvsl = trvsl->nxt;
	}
	
	datam* bottom = popee->dtm;
	popee->dtm = 0;
	popee->nxt = 0;
	delete popee; popee = 0;
	
	return bottom;
}

datam* datamList::search(std::string const& name) const
{
	datamnode* trvsl = head;

	while(trvsl != NULL)
	{
		if(name == trvsl->dtm->getName())
		{
			return trvsl->dtm;
		}
		trvsl = trvsl->nxt;
	}
	return NULL;
}

std::ostream& operator<<(std::ostream& output, datamList const& dtm)
{
	datamList::datamnode* trvsl = dtm.head;
	while(trvsl != NULL)
	{
		output << *trvsl->dtm << std::flush;
		trvsl = trvsl->nxt;
	}	
	return output;
}

