#ifndef DOESFOLDEREXIST_H
#define DOESFOLDEREXIST_H

#include <iostream>
#include <sys/stat.h>


inline bool doesDirExist(std::string pathName)
{
    bool exists = true;

    struct stat info;
    if( stat( pathName.c_str(), &info ) != 0 )
    {
        //printf( "cannot access %s\n", inputString.c_str() );
        exists = false;
    } else if( info.st_mode & S_IFDIR )  // S_ISDIR() doesn't exist on my windows
    {
        //printf( "%s is a directory\n", inputString.c_str() );
        exists = true;
    } else
    {
        //printf( "%s is no directory\n", inputString.c_str() );
        exists = false;
    }

    return exists;
}


#endif // DOESFOLDEREXIST_H