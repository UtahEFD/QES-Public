/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Common functions.
* Source: Bjarne's reference for C++ 3rd ed.
*/

#ifndef MINMAX
#define MINMAX 1

namespace QUIC
{

template<class T> inline 
T max(T t1, T t2)
{
	return (t1 >= t2) ? t1 : t2 ;
}

template<class T> inline
T min(T t1, T t2)
{
	return (t1 <= t2) ? t1 : t2 ;
}

inline int max(int i, int j) {return max<int>(i,j);}
inline float max(float i, float j) {return max<float>(i,j);}
inline double max(double i, double j) {return max<double>(i,j);}
inline char max(char i, char j) {return max<char>(i,j);}

template<class T> inline
int rnd(T t1)
{
	return int(t1 + .5);
}

inline int rnd(int i) {return rnd<int>(i);}
inline int rnd(float i) {return rnd<float>(i);}
inline int rnd(double i) {return rnd<double>(i);}

}

#endif

