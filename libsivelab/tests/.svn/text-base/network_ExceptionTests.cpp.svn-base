
#include <iostream>
#include "network/SocketException.h"

int main(int argc, char *argv[])
{
  try 
    {
      throw sivelab::SocketException();
    }
  catch (std::exception &e)
    {
      std::cout << e.what() << std::endl;
    }
  
  exit(EXIT_SUCCESS);
}
