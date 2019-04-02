template <class T>
Edge<T>::Edge()
{
	values.push_back( 0 );
	values.push_back( 0 );
}

template <class T>
Edge<T>::Edge(const T a, const T b)
{
	values.push_back(a);
	values.push_back(b);
}

template <class T>
T& Edge<T>::operator[](const int i)
{
	return values[i%2];
}

template <class T>
bool Edge<T>::operator==(const Edge< T > e) const
{
	if (values[0] == e.values[0])
		return values[1] == e.values[1];
	else if (values[0] == e.values[1])
		return values[1] == e.values[0]; 
	else
		return false;

}

template <class T>
int Edge<T>::getIndex(T v)
{
	if (values[0] == v)
		return 0;
	else if (values[1] == v)
		return 1;
	else
		return -1;
}