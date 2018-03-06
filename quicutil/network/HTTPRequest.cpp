#include <algorithm>

#include "HTTPRequest.h"

using namespace sivelab;

HTTPRequestMessage::HTTPRequestMessage()
{
  m_msgStream.str("");
}

HTTPRequestMessage::~HTTPRequestMessage()
{
}
    
bool HTTPRequestMessage::processAndConsumeEOL()
{
  // Get the CRLF at the end of the previous line
  char carriageRt = m_msgStream.get();
  char lineFd = m_msgStream.get();
      
  if ((carriageRt == '\r') && (lineFd == '\n')) 
    return true;
  else 
    return false;

#if 0
      // Peek-ahead for CRLF
      if (inputM_MsgStream.peek() == '\r')
	{
	  // consume and check if next is LF
	  carriageRt = inputM_MsgStream.get();
	  if (inputM_MsgStream.peek() == '\n')      
	    {
	      std::cout << "peek was a LF..." << std::endl;
	      lineFd = inputM_MsgStream.get();
	      crlfCount++;
	      std::cout << "crlfCount: " << crlfCount << std::endl;
	    }
	  else
	    syntaxError = true;
	}
#endif
  
}

bool HTTPRequestMessage::processMessage(ProtocolMessageBuffer &m)
{
  std::cout << "Processing HTTP request" << std::endl;
  
  m_msgStream.str( m.byteArray() );
  
  bool syntax_error = false;

  std::string reqType, reqURL, reqVers;
  m_msgStream >> reqType >> reqURL >> reqVers;
  processAndConsumeEOL();

  while ((m_msgStream.peek() != '\r') && !syntax_error)
    {
      // Should have a header to process
      std::string headerName, host;
      m_msgStream >> headerName;
      std::transform(headerName.begin(), headerName.end(), headerName.begin(), ::tolower);
      if (headerName == "host:")
	{
	  m_msgStream >> host;
	  std::cout << "host = " << host << std::endl;
	}

      // Eat up the CRLF
      if (processAndConsumeEOL() == false) syntax_error = true;
    }
  
  processAndConsumeEOL();

  std::cout << "Received valid HTTP Request: " << reqType << ' ' << reqURL << ' ' << reqVers << std::endl;
  return true;
}

void processHeaders()
{
  // From RFC 2616, HTTP
  /* 
       request-header = Accept                   ; Section 14.1
                      | Accept-Charset           ; Section 14.2
                      | Accept-Encoding          ; Section 14.3
                      | Accept-Language          ; Section 14.4
                      | Authorization            ; Section 14.8
                      | Expect                   ; Section 14.20
                      | From                     ; Section 14.22
                      | Host                     ; Section 14.23
                      | If-Match                 ; Section 14.24
                      | If-Modified-Since        ; Section 14.25
                      | If-None-Match            ; Section 14.26
                      | If-Range                 ; Section 14.27
                      | If-Unmodified-Since      ; Section 14.28
                      | Max-Forwards             ; Section 14.31
                      | Proxy-Authorization      ; Section 14.34
                      | Range                    ; Section 14.35
                      | Referer                  ; Section 14.36
                      | TE                       ; Section 14.39
                      | User-Agent               ; Section 14.43
  */
}

bool HTTPRequestMessage::generateResponse(ProtocolMessageBuffer &m)
{
  
}
