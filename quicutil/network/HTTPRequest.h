/*
 *  HTTPRequest.h
 *  NETCODE
 *
 *  Created by Pete Willemsen on 10/03/12.
 *  Copyright 2012 Department of Computer Science, University of Minnesota-Duluth. All rights reserved.
 *
 */

#ifndef __SIVELAB_HTTP_REQUEST_H__
#define __SIVELAB_HTTP_REQUEST_H__

#include <iostream>
#include <sstream>

#include "util/logstream.h"
#include "ProtocolMessage.h"

namespace sivelab {

  class HTTPRequestMessage : public ProtocolMessage
  {
  public:
    HTTPRequestMessage();
    ~HTTPRequestMessage();
    
    bool validateBuffer(const ProtocolMessageBuffer &m) { return false; }
    bool processMessage(ProtocolMessageBuffer &m);

    bool generateResponse(ProtocolMessageBuffer &m);
    
  protected:

  private:
    bool processHeaders();

    bool processAndConsumeEOL();

    bool m_messageReady;
    std::istringstream m_msgStream;
  };
  
}

#endif
