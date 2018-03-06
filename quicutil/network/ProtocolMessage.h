/*
 *  ProtocolMessage.h
 *  NETCODE
 *
 *  Created by Pete Willemsen on 10/03/12.
 *  Copyright 2012 Department of Computer Science, University of Minnesota-Duluth. All rights reserved.
 *
 */

#ifndef __SIVELAB_PROTOCOL_MESSAGE_H__
#define __SIVELAB_PROTOCOL_MESSAGE_H__

#include <cstdio>
#include <sstream>

#include "util/logstream.h"
#include "network/ProtocolMessageBuffer.h"

namespace sivelab {

  class ProtocolMessage
  {
  public:
    ProtocolMessage();
    virtual ~ProtocolMessage();

    virtual bool validateBuffer(const ProtocolMessageBuffer &m) = 0;
    virtual bool processMessage(ProtocolMessageBuffer &m) = 0;

  protected:
  private:
  };
  
}

#endif
