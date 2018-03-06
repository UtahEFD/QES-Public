#ifndef __SOCKET_EXCEPTION_H__
#define __SOCKET_EXCEPTION_H__ 1

#include <string>

namespace sivelab {

  class SocketException : public std::exception
  {
  public:
    SocketException(std::string message="General Socket Exception!") : m_message(message) {}
    ~SocketException() throw() {}
    
    const char* what() const throw() { return m_message.c_str(); }
    
  private:
    std::string m_message;
  };

}

#endif
