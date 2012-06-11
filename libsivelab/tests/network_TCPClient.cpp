#include <iostream>
#include <string>

#include <signal.h>

#include "util/handleNetworkArgs.h"
#include "network/ClientSocket.h"

using namespace sivelab;

static bool shutdownSender = false;

void signalHandler(int sig)
{
  std::cout << "Signal Caught!" << std::endl;
  shutdownSender = true;
}

int main(int argc, char *argv[])
{
  signal(SIGINT, signalHandler);

  NetworkArgs nArgs;
  nArgs.process(argc, argv);

  ClientSocket *cSock = 0;
  try 
    {
      cSock = new ClientSocket(nArgs.hostname, nArgs.port);
    }
  catch (std::exception &e)
    {
      std::cout << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }

  std::cout << "Sending client request..." << std::endl;
  std::string msg = "REQUEST";
  try 
    {
      cSock->send( msg );
    }
  catch (std::exception &e)
    {
      std::cout << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }

  

#if 0
  std::ostringstream d("");
  int count = 0;
  while (shutdownSender != true)
    {
      d.str("");
      d << count << ": " << time(NULL);
      uSock->send(sendTo_Addr, d);
      count++;

      usleep(100000);
    }
  std::cout << "Sent " << count << " messages via TCP." << std::endl;
#endif

  cSock->close();
}
