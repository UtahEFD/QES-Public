#include "urbViewer.h"

// const GLenum textureTarget = GL_TEXTURE_2D;
const GLenum textureTarget = GL_TEXTURE_RECTANGLE_ARB;

const GLenum textureFormat = GL_RGBA;

namespace QUIC 
{

	urbViewer::urbViewer() : viewer(), urbModule(), lagrange_shader(), euler_shader()
	{
		std::cout << "urbViewer constructor" << std::endl;
	
		nnx = nny = ngx = ngy = 0.;

		seperation = .5;
		shift = 0.;
		show_slice = -1;

		flat = view_euler = false;

		alpha = old_alpha = .2;

		l_pbos = l_texs = NULL;
		e_pbos = e_texs = NULL;

		t_p1 = NULL;

		t_vels = NULL;
		t_u = t_v = t_w = NULL;
			
		x_axis = y_axis = z_axis = NULL;

		uniform_lagrange_tex = 0;
		uniform_lagrange_alpha = 0.;
			
		uniform_euler_u_tex = 0;
		uniform_euler_alpha = 0.;
	}

	urbViewer::~urbViewer() 
	{
		this->cleanUpOpenGL();	
		cudaFree(t_p1);
		cudaFree(t_vels);
		delete l_pbos; l_pbos = 0;
		delete l_texs; l_texs = 0;	
		delete e_pbos; e_pbos = 0;
		delete e_texs; e_texs = 0;
		delete x_axis; x_axis = 0;
		delete y_axis; y_axis = 0;
		delete z_axis; z_axis = 0;
	}

	void urbViewer::initialize()
	{
		if(urbModule::nx == 0)
		{
			std::cerr << "Unable to initialize urbViewer. urbModule::nx = 0." << std::endl;
			std::cerr << "urbModule was probably not setup correctly." << std::endl;
		}
	
		nnx = 1.*urbModule::nx / urbModule::nx;
		nny = 1.*urbModule::ny / urbModule::nx;
		ngx = 1.*urbModule::gx / urbModule::gx;
		ngy = 1.*urbModule::gy / urbModule::gx;

		l_pbos = new GLuint[1];
		l_texs = new GLuint[urbModule::nz];
		e_pbos = new GLuint[1];
		e_texs = new GLuint[urbModule::gz];

		this->initOpenGL();

		point axises_loc = point(5.*nnx, 0., 0.);
		
		normal left    = normal(-1., 0., 0.);
		normal forward = normal( 0., 1., 0.);
		normal up      = normal( 0., 0., 1.);

		x_axis = new axis(0., 10.*nnx,             nx - 1., axises_loc, left);
		y_axis = new axis(0., seperation*(nz - 1), nz - 1., axises_loc,	forward);
		z_axis = new axis(0., 10.*nny,             ny - 1., axises_loc, up);

		cudaMalloc((void**) &t_p1, urbModule::domain_size*sizeof(float));
		cudaZero(t_p1, urbModule::domain_size, 0.);

		cudaMalloc((void**) &t_vels, urbModule::grid_size*4*sizeof(float));
		cudaZero(t_vels, urbModule::grid_size*4, 0.);

		t_u = (float*) &t_vels[0 * urbModule::grid_size];
		t_v = (float*) &t_vels[1 * urbModule::grid_size];
		t_w = (float*) &t_vels[2 * urbModule::grid_size];
	}

	void urbViewer::loadShaders(std::string directory)
	{
		if(directory.at(directory.length() - 1) != '/') 
		{
			directory = directory.append("/");
		}

		std::cout << "Loading shaders from " << directory << " ... " << std::endl;

		// Relative to where the program is run.		
		lagrange_shader.addShader(directory + "lagrange_shader_vp.glsl", GLSLObject::VERTEX_SHADER);
 		lagrange_shader.addShader(directory + "lagrange_shader_fp.glsl", GLSLObject::FRAGMENT_SHADER);
 		lagrange_shader.createProgram();
 		
		uniform_lagrange_tex   = lagrange_shader.createUniform("lagrange_tex");
		uniform_lagrange_alpha = lagrange_shader.createUniform("lagrange_alpha");

		// Relative to where the program is run.		
		euler_shader.addShader(directory + "euler_shader_vp.glsl", GLSLObject::VERTEX_SHADER);
 		euler_shader.addShader(directory + "euler_shader_fp.glsl", GLSLObject::FRAGMENT_SHADER);
 		euler_shader.createProgram();

		uniform_euler_u_tex = euler_shader.createUniform("euler_u_tex");
		uniform_euler_alpha = euler_shader.createUniform("euler_alpha");
		
		//std::cout << "Done loading shaders." << std::endl;
	}
	
	void urbViewer::nextIteration() 
	{
		if(!urbModule::converged) 
		{
			if(urbModule::getIteration() == 0) 	
			{
				QUIC::urbCUDA::firstIteration(this);
			}
			else 
			{
				QUIC::urbCUDA::iterate(this, 1);				
				QUIC::urbCUDA::checkForConvergence(this);
				
				QUIC::urbCUDA::calcVelocities(this);
			}
		}

		if(view_euler) {this->loadEulerTextures();}
		else		   		 {this->loadLagragianTextures();}
	}

	void urbViewer::glDrawData()
	{
		glPushMatrix();
		if(view_euler)	{this->glDrawEuler();}
		else			{this->glDrawLagragian();}
		if(!flat) {this->glDrawAxises();}
		glPopMatrix();
	}

	void urbViewer::glDrawAxises() const
	{
		x_axis->glDraw();
		y_axis->setLength(seperation * (nz - 1.)); y_axis->glDraw();
		z_axis->glDraw();
	}

	void urbViewer::setSeperation(float const& new_s)
	{
		seperation = (new_s < 0. || 5. < new_s) ? seperation : new_s ;
		flat       = (new_s < 0.)               ? true       : false ;
	}
	float urbViewer::getSeperation() const {return seperation;}

	void urbViewer::setShift(float const& new_s) 
	{
		shift = (new_s < -5. || 5. < new_s) ? shift : new_s ;
	}
	float urbViewer::getShift() const {return shift;}

	void urbViewer::setAlpha(float const& _alpha) 
	{
		alpha = (.04 < _alpha && _alpha < .96) ? _alpha : alpha ;
	}
		
	float urbViewer::getAlpha() const {return alpha;}

	void urbViewer::showPrevSlice() 
	{
		show_slice--;
		if(!view_euler && show_slice < -1) {show_slice = urbModule::nz - 1;}
		if( view_euler && show_slice < -1) {show_slice = urbModule::gz - 1;}
	}

	void urbViewer::showNextSlice() 
	{
		show_slice++;
		if(!view_euler && show_slice >= (int) urbModule::nz) {show_slice = -1;}
		if( view_euler && show_slice >= (int) urbModule::gz) {show_slice = -1;}
	}
	int urbViewer::getSliceToShow() const {return show_slice;}

	void urbViewer::setFlat(bool const& f)
	{
		flat = f;
		if(flat)
		{
			old_alpha = alpha;
			alpha = 1.;
		}
		else
		{
			alpha = old_alpha;
		}
	}
	
	bool urbViewer::isFlat() const {return flat;}

	void urbViewer::setEuler(bool const& e) 
	{
		if(e == true) {this->loadEulerTextures();}
		else 
		{
			this->loadLagragianTextures();
		}
		view_euler = e;
	}
	bool urbViewer::isEuler() const {return view_euler;}

	float urbViewer::getIntervalBound() const {return M;}

	void urbViewer::dumpIteration(std::string directory) const
	{
		if(directory.at(directory.length() - 1) != '/') 
		{
			directory = directory.append("/");
		}
		
		std::string prefix = "v_";

    QUIC::urbParser::outputLagrangians(this, directory, prefix);
    QUIC::urbParser::outputErrors     (this, directory, prefix);
    QUIC::urbParser::outputVelocities (this, directory, prefix);
	}

////////////////////
//Private Methods //
////////////////////
	
	void urbViewer::loadLagragianTextures() 
	{
		float* cuda_bff_ptr = 0;

		float* d_min; cudaMalloc((void**) &d_min, sizeof(float));
		float* d_max; cudaMalloc((void**) &d_max, sizeof(float));

		cudaMin(urbModule::d_p1, urbModule::domain_size, d_min);
		cudaMax(urbModule::d_p1, urbModule::domain_size, d_max);

		float l_min; cudaMemcpy(&l_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
		float l_max; cudaMemcpy(&l_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

		//std::cout << "l_min = " << l_min << "\tl_max = " << l_max << std::endl;

		cudaMemcpy
		(
			t_p1, 
			urbModule::d_p1, 
			urbModule::domain_size*sizeof(float), 
			cudaMemcpyDeviceToDevice
		);

		M = max(fabs(l_min), fabs(l_max));
		l_min = -M;
		l_max =  M;
		cudaMemcpy(d_min, &l_min, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_max, &l_max, sizeof(float), cudaMemcpyHostToDevice);

		cudaMapMatrixToIntervalWithDeviceLimits
		(
			t_p1, 
			urbModule::domain_size, 
			0., 1., 
			d_min, d_max
		);

		cudaFree(d_min);
		cudaFree(d_max);

		int slice_size = urbModule::nx * urbModule::ny;
		for(unsigned i = 0; i < urbModule::nz; i++) 
		{
			// map PBO to get CUDA device pointer
			cudaGLMapBufferObject((void**)&cuda_bff_ptr, l_pbos[0]);

			cudaPack
			(
				cuda_bff_ptr,
 				&t_p1[slice_size * i], 
				&t_p1[slice_size * i],
				&t_p1[slice_size * i],
				&t_p1[slice_size * i],
				slice_size
			);

			cudaGLUnmapBufferObject(l_pbos[0]);

			// download image from PBO to OpenGL texture
			glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, l_pbos[0]);
			glBindTexture(textureTarget, l_texs[i]);
			glTexSubImage2D
			(
				textureTarget, 
				0, 0, 0, 
				urbModule::nx, urbModule::ny, 
				textureFormat, GL_FLOAT, 
				NULL
			);
			glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		}
		glBindTexture(textureTarget, 0);
	}

	void urbViewer::loadEulerTextures() 
	{
		int ttl_bts = urbModule::grid_size*sizeof(float);
	
		cudaMemcpy(t_u, urbModule::d_vels.u, ttl_bts, cudaMemcpyDeviceToDevice);
		cudaMemcpy(t_v, urbModule::d_vels.v, ttl_bts, cudaMemcpyDeviceToDevice);
		cudaMemcpy(t_w, urbModule::d_vels.w, ttl_bts, cudaMemcpyDeviceToDevice);
		// Last section left as zeros //

		float* d_min; cudaMalloc((void**) &d_min, sizeof(float));
		float* d_max; cudaMalloc((void**) &d_max, sizeof(float));

		cudaMin(t_vels, urbModule::grid_size*3, d_min);
		cudaMax(t_vels, urbModule::grid_size*3, d_max);

		float v_min; cudaMemcpy(&v_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
		float v_max; cudaMemcpy(&v_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

		//std::cout << "e_v_min = " << v_min << "\te_v_max = " << v_max << std::endl;

		M = max(fabs(v_min), fabs(v_max));
		v_min = -M;
		v_max =  M;
		cudaMemcpy(d_min, &v_min, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_max, &v_max, sizeof(float), cudaMemcpyHostToDevice);

		cudaMapMatrixToIntervalWithDeviceLimits
		(
			t_vels, 
			urbModule::grid_size*3, 
			0., 1., 
			d_min, d_max
		);

		cudaFree(d_min);
		cudaFree(d_max);

		float* cuda_bff_ptr = 0;
		int slice_size = urbModule::gx * urbModule::gy;

		for(unsigned i = 0; i < urbModule::gz; i++) 
		{			
			// map PBO to get CUDA device pointer
			cudaGLMapBufferObject((void**)&cuda_bff_ptr, e_pbos[0]);
			cudaPack
			(
				cuda_bff_ptr,
				&t_u[i * slice_size],
				&t_v[i * slice_size],
				&t_w[i * slice_size],
				&t_vels[3 * urbModule::grid_size], // Pad alhpa with 0's
				slice_size
			);
			cudaGLUnmapBufferObject(e_pbos[0]);

			// download image from PBO to OpenGL texture				
			glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, e_pbos[0]);
			glBindTexture(textureTarget, e_texs[i]);
			glTexSubImage2D
			(
				textureTarget, 
				0, 0, 0, 
				urbModule::gx, urbModule::gy, 
				textureFormat, GL_FLOAT, 
				NULL
			);
			glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		}
		glBindTexture(textureTarget, 0);
	}

	void urbViewer::getFlatCorners
	(
		int const& slice, 
		float& left, float& bottom, 
		float& right, float& top
	) const
	{
		int cols = (int) sqrt(urbModule::nz * nny) + 1;
		int rows = (int) cols / nny + 1;

		//What am I doing here? Mapping multiple slices to a flat place.
		left   = int(slice % cols) * 1. / cols;
		bottom = int(slice / cols) * 1. / rows;
		right  =   left + 1. / cols;
		top    = bottom + 1. / rows;
	}
	
	void urbViewer::glDrawLagragian()
	{
		
		float left, bottom, right, top;

		for(int i = 0; i < (int) urbModule::nz; i++) 
		{    			
			glBindTexture(textureTarget, l_texs[i]);
			
			lagrange_shader.activate();
			glUniform1iARB(uniform_lagrange_tex, 0); 
			glUniform1fARB(uniform_lagrange_alpha, alpha);

			glEnable(textureTarget);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);	

			if(flat) 
			{				
				this->getFlatCorners(i, left, bottom, right, top);

				glBegin(GL_QUADS);

				if (textureTarget == GL_TEXTURE_2D)
				{
				    glTexCoord2f(0, 1); glVertex2f( left, top); 
				    glTexCoord2f(1, 1); glVertex2f(right, top); 
				    glTexCoord2f(1, 0); glVertex2f(right, bottom); 
				    glTexCoord2f(0, 0); glVertex2f( left, bottom); 
				}
				else
				{
				    // Have to use texture width and height for texcoord
				    glTexCoord2f(            0,	urbModule::ny); glVertex2f( left, top); 
				    glTexCoord2f(urbModule::nx, urbModule::ny); glVertex2f(right, top); 
				    glTexCoord2f(urbModule::nx,             0); glVertex2f(right, bottom); 
				    glTexCoord2f(            0,	            0); glVertex2f( left, bottom); 
				}
				glEnd();		
			} 
			else 
			{				
				if(show_slice == i || show_slice == -1) 
				{
					glBegin(GL_QUADS);

					if (textureTarget == GL_TEXTURE_2D)
					{
					    glTexCoord2f(0, 1); 
					    glVertex3f( 5.*nnx, seperation*i, 10.*nny + shift*i);
					    
					    glTexCoord2f(1, 1); 
					    glVertex3f(-5.*nnx, seperation*i, 10.*nny + shift*i);
					    
					    glTexCoord2f(1, 0); 
					    glVertex3f(-5.*nnx, seperation*i,  0.*nny + shift*i);
					    
					    glTexCoord2f(0, 0); 
					    glVertex3f( 5.*nnx, seperation*i,  0.*nny + shift*i);
					}
					else
					{
					    glTexCoord2f(            0, urbModule::ny); 
					    glVertex3f( 5.*nnx, seperation*i, 10.*nny + shift*i);
					    
					    glTexCoord2f(urbModule::nx, urbModule::ny); 
					    glVertex3f(-5.*nnx, seperation*i, 10.*nny + shift*i);
					    
					    glTexCoord2f(urbModule::nx,	            0); 
					    glVertex3f(-5.*nnx, seperation*i,  0.*nny + shift*i);
					    
					    glTexCoord2f(            0,	            0); 
					    glVertex3f( 5.*nnx, seperation*i,  0.*nny + shift*i);
					}
					glEnd();
				}
			}
			lagrange_shader.deactivate();

			glBindTexture(textureTarget, 0);
			glDisable(textureTarget);
		}
	}

	void urbViewer::glDrawEuler()
	{		
		float left, bottom, right, top;

		for(int i = 0; i < (int) urbModule::gz; i++) 
		{		
			glBindTexture(textureTarget, e_texs[i]);
			
			euler_shader.activate();
			glUniform1iARB(uniform_euler_u_tex, 0); 
			glUniform1fARB(uniform_euler_alpha, alpha);	

			glEnable(textureTarget);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);	

			if(flat) 
			{
				this->getFlatCorners(i, left, bottom, right, top);

				glBegin(GL_QUADS);

				if (textureTarget == GL_TEXTURE_2D)
				{
					glTexCoord2f(0, 1); glVertex2f( left, top); 
				  glTexCoord2f(1, 1); glVertex2f(right, top); 
				  glTexCoord2f(1, 0); glVertex2f(right, bottom); 
				  glTexCoord2f(0, 0); glVertex2f( left, bottom); 
				}
				else
				{
				    // Have to use texture width and height for texcoord
					glTexCoord2f(            0, urbModule::gy); glVertex2f( left, top);
				  glTexCoord2f(urbModule::gx, urbModule::gy); glVertex2f(right, top);
				  glTexCoord2f(urbModule::gx,             0); glVertex2f(right, bottom);
				  glTexCoord2f(            0,             0); glVertex2f( left, bottom);
				}
				glEnd();
			}
			else
			{
				if(show_slice == i || show_slice == -1)
				{
					glBegin(GL_QUADS);

					if (textureTarget == GL_TEXTURE_2D)
					{
				    glTexCoord2f(0, 1); 
				    glVertex3f( 5.*ngx, seperation*i, 10.*ngy + shift*i);
					    
				    glTexCoord2f(1, 1); 
				    glVertex3f(-5.*ngx, seperation*i, 10.*ngy + shift*i);
					    
				    glTexCoord2f(1, 0); 
				    glVertex3f(-5.*ngx, seperation*i,  0.*ngy + shift*i);
					    
				    glTexCoord2f(0, 0); 
				    glVertex3f( 5.*ngx, seperation*i,  0.*ngy + shift*i);
					}
					else
					{
				    glTexCoord2f(            0, urbModule::gy); 
				    glVertex3f( 5.*ngx, seperation*i, 10.*ngy + shift*i);
				    
				    glTexCoord2f(urbModule::gx, urbModule::gy); 
				    glVertex3f(-5.*ngx, seperation*i, 10.*ngy + shift*i);
				    
				    glTexCoord2f(urbModule::gx,             0); 
				    glVertex3f(-5.*ngx, seperation*i,  0.*ngy + shift*i);
				    
				    glTexCoord2f(            0,             0); 
				    glVertex3f( 5.*ngx, seperation*i,  0.*ngy + shift*i);
					}
					glEnd();
				}
			}
			euler_shader.deactivate();

			glBindTexture(textureTarget, 0);
			glDisable(textureTarget);
		}
	}
	
	void urbViewer::initOpenGL() 
	{		
		glewInit();


		// Lagrangians //
		glGenBuffers (1, l_pbos);
		glGenTextures(urbModule::nz, l_texs);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		// Create pixel buffer object for display.
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, l_pbos[0]);
		glBufferData
		(
			GL_PIXEL_UNPACK_BUFFER,  
			urbModule::nx * urbModule::ny * 4 * sizeof(GLfloat), 
			0, 
			GL_DYNAMIC_COPY
		);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		cudaGLRegisterBufferObject(l_pbos[0]);

		for(int i = 0; i < (int) urbModule::nz; i++) 
		{
			// Set texture parameters for display.
			glBindTexture(textureTarget, l_texs[i]);
			glTexParameteri(textureTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(textureTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexImage2D
			(
				textureTarget, 0, GL_RGBA32F_ARB, 
				urbModule::nx, urbModule::ny, 
				0, textureFormat, GL_FLOAT, NULL
			);
			glBindTexture(textureTarget, 0);
		}

		// Euler - Velocities //
		glGenBuffers (1, e_pbos);
		glGenTextures(urbModule::gz, e_texs);

		// Create pixel buffer object for display.
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, e_pbos[0]);
		glBufferData
		(
			GL_PIXEL_UNPACK_BUFFER, 
			urbModule::gx * urbModule::gy * 4 * sizeof(GLfloat), 
			0, 
			GL_DYNAMIC_COPY
		);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		cudaGLRegisterBufferObject(e_pbos[0]);

		for(unsigned i = 0; i < urbModule::gz; i++) 
		{
			// Set texture parameters for display.
			glBindTexture(textureTarget, e_texs[i]);
			glTexParameteri(textureTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(textureTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexImage2D
			(
				textureTarget, 0, GL_RGBA32F_ARB, 
				urbModule::gx, urbModule::gy, 
				0, textureFormat, GL_FLOAT, NULL
			);
			glBindTexture(textureTarget, 0);
		}
	}
			
	void urbViewer::cleanUpOpenGL() 
	{
		cudaGLUnmapBufferObject(l_pbos[0]);
    	cudaGLUnregisterBufferObject(l_pbos[0]);
		
		glDeleteTextures(urbModule::nz, l_texs);
		glDeleteBuffers (1, l_pbos);

		cudaGLUnmapBufferObject(e_pbos[0]);
   		cudaGLUnregisterBufferObject(e_pbos[0]);

		glDeleteTextures(urbModule::gz, e_texs);
		glDeleteBuffers (1, e_pbos);
	}
}

