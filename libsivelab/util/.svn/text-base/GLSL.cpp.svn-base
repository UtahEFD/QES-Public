#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cstring>
#include <assert.h>
#include "GLSL.h"

// ////////////////////////////////////////////////////////////////////////////////
// 
// GLSLObject - class that encapsulates the OpenGL calls necessary for
// using the OpenGL Shading Language shaders.
//
// Comments or Questions? Contact Pete Willemsen <willemsn@d.umn.edu>
// $Id: GLSL.cpp,v 1.5 2006/12/09 03:39:56 willemsn Exp $
//
// ////////////////////////////////////////////////////////////////////////////////

namespace sivelab
{
  GLSLObject::GLSLObject()
  {
    // no extra output
    m_verbose_level = 0;

    // Perform some checks to see if GLSL vertex shaders and fragment
    // shaders are actually available on the hardware.
    int err = 0;

    // The following four extensions are required to use GLSL Shaders
    if (!GL_ARB_vertex_shader)
      {
        std::cerr << "GL_ARB_vertex_shader is MISSING!" << std::endl;
        err++;
      }

    if (!GL_ARB_fragment_shader)
      {
        std::cerr << "GL_ARB_fragment_shader is MISSING!" << std::endl;
        err++;
      }

    if (!GL_ARB_shader_objects)
      {
        std::cerr << "GL_ARB_shader_objects is MISSING!" << std::endl;
        err++;
      }

    if (!GL_ARB_shading_language_100)
      {
        std::cerr << "GL_ARB_shading_language_100 is MISSING!" << std::endl;
        err++;
      }

    if (err > 0)
      {
        std::cerr << std::endl;
        std::cerr << "The extensions required to use the OpenGL Shading Language are\n";
        std::cerr << "not available on this machine.  The following extensions are required:\n";
        std::cerr << "\tGL_ARB_vertex_shader\n";
        std::cerr << "\tGL_ARB_fragment_shader\n";
        std::cerr << "\tGL_ARB_shader_objects\n";
        std::cerr << "\tGL_ARB_shading_language_100\n" << std::endl;;
        std::cerr << std::endl;
        std::cerr << "Exiting." << std::endl;

        exit(1);
      }
  }

  GLSLObject::~GLSLObject()
  {
  }

  void GLSLObject::setInputandOutput(GLenum input, GLenum output,int n)
  {
    input_type = input;
    output_type = output;
    num_vertices = n;
  }
  //This function is hardcoded for setting the output type to positions!!
  void GLSLObject::setVaryingOutput(int count, int* locations,GLenum buffermode){
    GLint positions = glGetVaryingLocationNV(m_program_object,"gl_Position");
    glTransformFeedbackVaryingsNV(m_program_object,count,&positions,buffermode);
  }

  void GLSLObject::addShader( const std::string& filename, ShaderType shader_type )
  {
    glErrorCheck("glCreateShaderObjectARB1( stype )");

    assert( shader_type == VERTEX_SHADER ||
	    shader_type == FRAGMENT_SHADER ||
	    shader_type == GEOMETRY_SHADER);
    

    GLenum stype;
    if (shader_type == VERTEX_SHADER)
      stype = GL_VERTEX_SHADER_ARB;
    else if (shader_type == FRAGMENT_SHADER)
      stype = GL_FRAGMENT_SHADER_ARB;
    else if (shader_type == GEOMETRY_SHADER)
      stype = GL_GEOMETRY_SHADER_EXT;
    else
      return;
    
    // Create the shader object
    GLhandleARB shader = glCreateShaderObjectARB( stype );
    glErrorCheck("glCreateShaderObjectARB( stype )");
    
    // Read the source from the file and load it into the shader
    loadSourceFromFile( filename, shader );

    //
    // Compile the shader
    //
    glCompileShaderARB( shader );
    glErrorCheck("glCompileShaderARB( shader )");

    //
    // Check compilation status
    //
    GLint result;
    glGetObjectParameterivARB( shader, GL_OBJECT_COMPILE_STATUS_ARB, &result );
    if (result) {
      std::cout << "Shader compilation successful: " << filename << std::endl;
      
      // Add the shader to the appropriate list
      if (shader_type == VERTEX_SHADER) 
        m_vertexshader_objects.push_back( shader );
      else if(shader_type == FRAGMENT_SHADER)
        m_fragmentshader_objects.push_back( shader );
      else
        m_geometryshader_objects.push_back( shader );
    }
    else {
      std::cout << "Shader object compilation failed: " << filename << std::endl;
      GLcharARB error_log[ 512 ];
      glGetInfoLogARB( shader, 512, 0, error_log );
      std::cout << "GLSL ERROR LOG\n--------------" << std::endl;
      std::cout << error_log << std::endl << std::endl;
    }
   
  }

  void GLSLObject::createProgram()
  {
    // Create a program object to link the two shaders together
    m_program_object = glCreateProgramObjectARB();
    glErrorCheck("glCreateProgramObjectARB()");
    
    //
    // Attach the vertex shader and fragment shader to the program object
    //
    std::list< GLhandleARB >::iterator li;

    // Vertex Shaders first
    for (li=m_vertexshader_objects.begin(); li!=m_vertexshader_objects.end(); ++li)
      {
        glAttachObjectARB( m_program_object, *li );
        glErrorCheck("glAttachObjectARB( m_program_object, *li ) - Vertex Shader");
      }
    // Fragment Shaders second
    for (li=m_fragmentshader_objects.begin(); li!=m_fragmentshader_objects.end(); ++li)
      {
        glAttachObjectARB( m_program_object, *li );
        glErrorCheck("glAttachObjectARB( m_program_object, *li ) - Fragment Shader");
      }
    // Geometry Shaders
    for (li=m_geometryshader_objects.begin(); li!=m_geometryshader_objects.end(); ++li)
      {
        glProgramParameteriEXT(m_program_object, GL_GEOMETRY_INPUT_TYPE_EXT, input_type);
        glProgramParameteriEXT(m_program_object, GL_GEOMETRY_OUTPUT_TYPE_EXT, output_type);
        glProgramParameteriEXT(m_program_object, GL_GEOMETRY_VERTICES_OUT_EXT, num_vertices);
        glAttachObjectARB( m_program_object, *li );
        glErrorCheck("glAttachObjectARB( m_program_object, *li ) - Geomtry Shader");
      }
    // Now, link the program
    glLinkProgramARB( m_program_object );
    glErrorCheck("glLinkProgramARB( m_program_object )");
  }

  void GLSLObject::activate()
  {
    glUseProgramObjectARB( m_program_object );
  }

  void GLSLObject::deactivate()
  {
    glUseProgramObjectARB( 0 );
  }

  int GLSLObject::createUniform(const std::string& name)
  {
    return glGetUniformLocationARB(m_program_object, name.c_str());  
  }

  void GLSLObject::setReportingVerbosity(int val)
  {
    if (val <= 0)
      m_verbose_level = 0;
    else
      m_verbose_level = val;
  }

  bool GLSLObject::f_stripComments( char* buf )
  {
    char* comment_ptr = strstr(buf, "//");
    if (comment_ptr == 0)
      return 1;  // comment not located, so do not alter the buffer
    else 
      {
        // comment delimiter located, remove comments
        *comment_ptr = '\0';

        // check the strlen, if less than 1, return false
        if (strlen(buf) == 0)
	  return false;
        else 
	  {
	    for (unsigned int i=0; i<strlen(buf); i++)
	      {
	        if (buf[i] != ' ')
		  return true;
	      }

	    // if we make it here, all the characters in the buffer were
	    // spaces, so return false...
	    return false;
	  }
      }
  }

  void GLSLObject::loadSourceFromFile( const std::string& filename, GLhandleARB& shader_obj )
  {
    if (m_verbose_level)
      std::cout << "GLSLObject::loadSourceFromFile( filename=" << filename << " )" << std::endl;
	
      //
      // open the file and load into string array
      //
      std::ifstream shader_file( filename.c_str() );
      if (shader_file.is_open() == false)
        {
	  std::cerr << "Error opening file \"" << filename << "\".  Exiting." << std::endl;
	  exit(1);
        }

      int linemax = 1024;
      char *linebuf = new char[ linemax ];
      std::list< std::string > string_list;
      while (shader_file.good() && !shader_file.eof())
      {
	  // read a line off the file and append to string list
	  shader_file.getline( linebuf, linemax );

	  // strip out the comments, following "//"
	  if (f_stripComments( linebuf )) 
	      string_list.push_back( linebuf );
      }
      shader_file.close();
      delete [] linebuf;

      //
      // allocate the memory, load the source strings, deallocate memory,
      // and return.
      //
      GLcharARB **shader_string = new GLcharARB*[string_list.size()];
    
      unsigned int idx, j;
      std::list< std::string >::const_iterator li;
      for (li=string_list.begin(), idx=0; li!=string_list.end(); ++li, idx++)
      {
	  shader_string[idx] = new GLcharARB[li->length()+1];
	  for (j=0; j<li->length(); j++)
	      shader_string[idx][j] = (*li)[j];
	  shader_string[idx][j] = '\0';  // null terminate the string

	  if (m_verbose_level)
	    std::cout << "Added \"" << shader_string[idx] << "\", length=" << li->length() << std::endl;
      }
    
      //
      // Load the shader source into the shader object 
      //
      glShaderSourceARB( shader_obj, string_list.size(), (const GLcharARB**)shader_string, 0 );

      // Clean up allocated memory
      for (idx=0; idx<string_list.size(); idx++)
	  delete [] shader_string[idx];
      delete [] shader_string;
  }

  void GLSLObject::glErrorCheck( const char* msg )
  {
  #ifndef NDEBUG
    GLenum err_code;
    err_code = glGetError();
    while (err_code != GL_NO_ERROR) {
      fprintf(stderr, "OpenGL Error: %s, Context[%s]\n", gluErrorString(err_code), msg);
      err_code = glGetError();
    }
  #endif
  }
}

