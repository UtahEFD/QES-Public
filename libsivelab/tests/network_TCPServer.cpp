#include <iostream>
#include <string>

#include "util/handleNetworkArgs.h"
#include "network/ServerSocket.h"

using namespace sivelab;

int main(int argc, char *argv[])
{
  NetworkArgs nArgs;
  nArgs.process(argc, argv);

  // This code shows how to use either the IPv4 or the IPv6 sockets.
  // We setup the code this way to explicitly create a specific type
  // of socket.  Note that IPv6 sockets can received IPv4 traffic via
  // the IPv4-Mapped IPv6 address (:FFFF:IPv4Address)

  // This creates a server that is bound to the port provided in the
  // constructor args.
  ServerSocket *serverSocket = 0;
  try 
    {
      serverSocket = new ServerSocket(nArgs.port);
    } 
  catch (std::exception &e)
    {
      std::cout << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
  
  bool doneProcessing = false;
  do {
    ClientSocket *clientSocket = serverSocket->accept();
    std::cout << "Client connected!" << std::endl;

    // Wait for response from server
    std::string cRequest;
    clientSocket->receive( cRequest );
    std::cout << "Received Client Request: " << cRequest << std::endl;

    // Send server a request for text
    std::string msg = "DATA";
    clientSocket->send( msg );

    delete clientSocket;
    doneProcessing = true;
  }
  while (!doneProcessing);

#if 0
  int count = 0;
  std::ostringstream d;
  while (count < 25)
    {
      tSock->recv( d );
      std::cout << "Got: " << d.str() << std::endl;
      count++;
    }
#endif

  delete serverSocket;
}
