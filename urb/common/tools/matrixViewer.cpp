#include "matrixViewer.h"

namespace QUIC
{

	matrixViewer::matrixViewer(float* M_1, float* M_2, int _nx, int _ny, int _nz) : viewer() 
	{
		nx = _nx;
		ny = _ny;
		nz = _nz;
		
		nnx = nx / (nx * 1.0f);
		nny = ny / (nx * 1.0f);

		std::cout << "nnx = " << nnx << std::endl;
		std::cout << "nny = " << nny << std::endl;

		pbos = new GLuint[nz*4];
		texs = new GLuint[nz*4];
		
		smoothTex = false;
		threshold = 0.001f;

		seperation = .5f;
		shift = 0.0f;
		show_slice = -1;
		flat = false;
		show_abs_dif = !(show_rel_dif = true);
		
		alpha = old_alpha = 0.2f;

		int data_size = nx * ny * nz * sizeof(float);

		//Setup Matrix device pointers
			std::cout << "Allocating device memory..." << std::flush;
		//Allocate memory
		cudaMalloc((void**) &d_M1, data_size);
		cudaMalloc((void**) &d_M2, data_size);
		cudaMalloc((void**) &d_abs_dif, data_size);
		cudaMalloc((void**) &d_rel_dif, data_size);
			std::cout << "done." << std::endl;

			//Copy matrices
			std::cout << "Transferring matrices to device..." << std::flush;
		cudaMemcpy(d_M1, M_1, data_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_M2, M_2, data_size, cudaMemcpyHostToDevice);
			std::cout << "done." << std::endl;

		//Find the difference
			std::cout << "Finding absolute difference..." << std::flush;
		cudaAbsDiff(d_abs_dif, d_M1, d_M2, nx*ny*nz);
			std::cout << "done." << std::endl;

		//Find the percentage error
			std::cout << "Finding the relative difference..." << std::flush;
		cudaRelDiff(d_rel_dif, d_M1, d_M2, nx*ny*nz, pow(10.,-6.));
			std::cout << "done." << std::endl;

		float* d_mins; cudaMalloc((void**) &d_mins, 2 * sizeof(float));
		float* d_maxs; cudaMalloc((void**) &d_maxs, 2 * sizeof(float));

		cudaMin(d_M1, nx*ny*nz, &d_mins[0]);
		cudaMin(d_M2, nx*ny*nz, &d_mins[1]);
		cudaMin(d_mins, 2, &d_mins[0]);

		cudaMax(d_M1, nx*ny*nz, &d_maxs[0]);
		cudaMax(d_M2, nx*ny*nz, &d_maxs[1]);
		cudaMax(d_maxs, 2, &d_maxs[0]);

		float t_min = 0; cudaMemcpy(&t_min, d_mins, sizeof(float), cudaMemcpyDeviceToHost);
		float t_max = 0; cudaMemcpy(&t_max, d_maxs, sizeof(float), cudaMemcpyDeviceToHost);
/*
		if(fabs(t_min) > fabs(t_max)) 
		{
			t_max = -t_min;
			cudaMemcpy(d_maxs, &t_max, sizeof(float), cudaMemcpyHostToDevice);
		}
		else 
		{
			t_min = -t_max;
			cudaMemcpy(d_mins, &t_min, sizeof(float), cudaMemcpyHostToDevice);
		}
*/
		// Looks like I normalized to a [-M, M] interval, 
		// where M = max(abs(min), abs(max)). 
		
		// Grab the min and max for display (and eventually cool value-bar-thingy).
		// \\todo Josh, add a cool value-bar-thingy.
		M = max(fabs(t_min), fabs(t_max));
		t_min = -M;
		t_max =  M;
		cudaMemcpy(d_mins, &t_min, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_maxs, &t_max, sizeof(float), cudaMemcpyHostToDevice);
				
		cudaMapMatrixToIntervalWithDeviceLimits
		(
			d_M1, nx*ny*nz, 
			0.0, 1.0, &d_mins[0], &d_maxs[0]
		); 
		cudaMapMatrixToIntervalWithDeviceLimits
		(
			d_M2, nx*ny*nz, 
			0.0, 1.0, &d_mins[0], &d_maxs[0]
		); 

		cudaFree(d_mins);
		cudaFree(d_maxs);

		//Now that the difference is found, setup the display
			std::cout << "Initializing buffers..." << std::flush;
		this->initOpenGL();
			std::cout << "done." << std::endl;

		//Setup Shaders
		difference_shader.addShader("../src/shaders/difference_shader_vp.glsl", GLSLObject::VERTEX_SHADER);
		difference_shader.addShader("../src/shaders/difference_shader_fp.glsl", GLSLObject::FRAGMENT_SHADER);
 		difference_shader.createProgram();
		uniform_difference_tex       = difference_shader.createUniform("difference_tex");
		uniform_difference_threshold = difference_shader.createUniform("difference_threshold");
		uniform_difference_alpha     = difference_shader.createUniform("difference_alpha");

		unit_interval_shader.addShader("../src/shaders/unit_interval_shader_vp.glsl", GLSLObject::VERTEX_SHADER);
		unit_interval_shader.addShader("../src/shaders/unit_interval_shader_fp.glsl", GLSLObject::FRAGMENT_SHADER);
 		unit_interval_shader.createProgram();
		uniform_unitinterval_tex   = unit_interval_shader.createUniform("unitinterval_tex");
		uniform_unitinterval_alpha = unit_interval_shader.createUniform("unitinterval_alpha");
	}

	matrixViewer::~matrixViewer() 
	{
		this->cleanUpOpenGL();	
		cudaFree(d_M1); d_M1 = 0;
		cudaFree(d_M2); d_M2 = 0;
		cudaFree(d_abs_dif); d_abs_dif = 0;
		cudaFree(d_rel_dif); d_rel_dif = 0;
	}

	void matrixViewer::glDrawMatrixView() 
	{
		if(show_abs_dif || show_rel_dif) {this->glDrawDiff();}
		else			 				 {this->glDrawSeperate();}
	}

	void matrixViewer::setSeperation(const float new_s) 
	{
		if(new_s < 0.0f || 5.0f < new_s) {}
		else							 {seperation = new_s;}

		if(new_s < 0.0f) {flat = true;}
		else			 			 {flat = false;}
	}
	float matrixViewer::getSeperation() {return seperation;}

	void matrixViewer::setShift(const float new_s) 
	{
		if(new_s < -5.0f || 5.0f < new_s) {}
		else							  {shift = new_s;}
	}
	float matrixViewer::getShift() {return shift;}

	void matrixViewer::setAlpha(float const& _alpha) 
	{
		alpha = (0.04 < _alpha && _alpha < .96) ? _alpha : alpha ;
	}
	float matrixViewer::getAlpha() const {return alpha;}

	void matrixViewer::showNextSlice() 
	{
		show_slice++;
		if(show_slice >= nz) {show_slice = -1;}
	}

	void matrixViewer::showPreviousSlice() 
	{
		show_slice--;
		if(show_slice < 0) {show_slice = nz - 1;}
	}

	int matrixViewer::getSliceToShow() {return show_slice;}

	void matrixViewer::setFlat(bool f) 
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
	bool matrixViewer::isFlat() 	   	{return flat;}

	void matrixViewer::showRelDiff(bool rd) {show_abs_dif = !(show_rel_dif = rd);}
	bool matrixViewer::isRelDiff() {return show_rel_dif;}

	void matrixViewer::showDiff() {show_abs_dif = !(show_rel_dif);}
	void matrixViewer::showSeperate() {show_abs_dif = show_rel_dif = false;}
	bool matrixViewer::areSeperate() {return !(show_abs_dif || show_rel_dif);}

	void matrixViewer::toggleSmoothTex() {smoothTex = !(smoothTex);}

	void matrixViewer::setThreshold(float t) 
	{
			 if(t > 1.0f) {threshold = 1.0f;}
		else if(t < 0.0f) {threshold = 0.0f;}
		else 			 {threshold = t;}
	}
	float matrixViewer::getThreshold() {return threshold;}
	
	float matrixViewer::getIntervalBound() const {return M;}

//Private Methods
	void matrixViewer::initOpenGL() 
	{		
		glewInit();

		glGenBuffers(nz*4, pbos);
		glGenTextures(nz*4, texs);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		float* cuda_bff_ptr = 0;
		float* cuda_slice_ptr = 0;

		for(int i = 0; i < nz*4; i++) // A set for m_1, m_2 and their differences
		{
			// create pixel buffer object for display
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbos[i]);
			glBufferData(GL_PIXEL_UNPACK_BUFFER, nx * ny * sizeof(GLfloat), 0, GL_STREAM_COPY);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

			cudaGLRegisterBufferObject(pbos[i]);

			// set texture parameters for display
			glBindTexture  (GL_TEXTURE_2D, texs[i]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glBindTexture  (GL_TEXTURE_2D, 0);

			// point to the current slice.
			int s = i % nz;
			if(i < nz) 			    {cuda_slice_ptr = &d_abs_dif[nx * ny * s];}	// abs_dif
			else if(i < nz * 2) {cuda_slice_ptr = &d_M1[nx * ny * s];}		  // M_1
			else if(i < nz * 3) {cuda_slice_ptr = &d_M2[nx * ny * s];}		  // M_2
			else 				        {cuda_slice_ptr = &d_rel_dif[nx * ny * s];}	// rel_dif

			// map PBO to get CUDA device pointer
			cudaGLMapBufferObject((void**) &cuda_bff_ptr, pbos[i]);
			cudaMemcpy(cuda_bff_ptr, cuda_slice_ptr, nx * ny * sizeof(float), cudaMemcpyDeviceToDevice);
			cudaGLUnmapBufferObject(pbos[i]);	
		}

		glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
	}

	void matrixViewer::cleanUpOpenGL() 
	{
		for(int i = 0; i < nz * 4; i++) 
		{
			cudaGLUnmapBufferObject(pbos[i]);
    		cudaGLUnregisterBufferObject(pbos[i]);
		}
		glDeleteTextures(nz*4,texs);
		glDeleteBuffers(nz*4,pbos);

		delete pbos; pbos = 0;
		delete texs; pbos = 0;	
	}

	void matrixViewer::glDrawDiff() 
	{
		glPushMatrix();

		int d = (show_abs_dif) ? 0 : 3;

		for(int i = 0; i < nz; i++) 
		{    			
			// download image from PBO to OpenGL texture
			glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbos[d*nz + i]);
			glBindTexture(GL_TEXTURE_2D, texs[d*nz + i]);
			this->glTexFilter();
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, nx, ny, 0, GL_LUMINANCE, GL_FLOAT, NULL);

			difference_shader.activate();
			glUniform1iARB(uniform_difference_tex, 0); 
			glUniform1fARB(uniform_difference_threshold, this->getThreshold());
			glUniform1fARB(uniform_difference_alpha, alpha);

			glEnable(GL_TEXTURE_2D);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);		

			if(flat) 
			{
				int cols = (int) sqrt(nz * nny) + 1;
				int rows = (int) cols / nny + 1;

				//What am I doing here? Mapping multiple slices to a flat place.
				float left   = int(i % cols) * 1. / cols;
				float bottom = int(i / cols) * 1. / rows;
				float right  =   left + 1. / cols;
				float top    = bottom + 1. / rows;

				glBegin(GL_QUADS);
				glTexCoord2f(0, 1); glVertex2f( left, top); 
				glTexCoord2f(1, 1); glVertex2f(right, top); 
				glTexCoord2f(1, 0); glVertex2f(right, bottom); 
				glTexCoord2f(0, 0); glVertex2f( left, bottom); 
				glEnd();	
			} 
			else 
			{
				if(show_slice == i || show_slice == -1) 
				{
					glBegin(GL_QUADS);
					glTexCoord2f(0, 1); glVertex3f( 5.0f * nnx, seperation * i, 10.0f * nny + shift * i);
					glTexCoord2f(1, 1); glVertex3f(-5.0f * nnx, seperation * i, 10.0f * nny + shift * i);
					glTexCoord2f(1, 0); glVertex3f(-5.0f * nnx, seperation * i,  0.0f * nny + shift * i);
					glTexCoord2f(0, 0); glVertex3f( 5.0f * nnx, seperation * i,  0.0f * nny + shift * i);
					glEnd();
				}
			}

			difference_shader.deactivate();

			glDisable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, 0);
			glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		}

		glPopMatrix();
	}

	void matrixViewer::glDrawSeperate() 
	{
		for(int i = 0; i < nz; i++) 
		{				
			// download image from PBO to OpenGL texture
			// Do matrix 1 & 2
			for(int m = 1; m <= 2; m++) 
			{
				glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbos[m*nz + i]);
				glBindTexture(GL_TEXTURE_2D, texs[m*nz + i]);
				this->glTexFilter();
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, nx, ny, 0, GL_LUMINANCE, GL_FLOAT, NULL);

				unit_interval_shader.activate();
				glUniform1iARB(uniform_unitinterval_tex, 0);
				glUniform1fARB(uniform_unitinterval_alpha, alpha);

				glEnable(GL_TEXTURE_2D);
				glEnable(GL_BLEND);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);		


				if(flat) 
				{
					glPushMatrix();				

					float s = 0.0;
					if(m == 1) {s = 0.0;}
					else	   {s = 0.5;}

					int cols = (int) sqrt(nz * nny) + 1;
					int rows = (int) cols / nny + 1;

					//What am I doing here? Mapping multiple slices to a flat place.
					float left   = int(i % cols) * 1. / cols / 2. + s;
					float bottom = int(i / cols) * 1. / rows / 2.;
					float right  =   left + 1. / cols / 2.;
					float top    = bottom + 1. / rows / 2.;

					glBegin(GL_QUADS);
					glTexCoord2f(0, 1); glVertex2f( left, top); 
					glTexCoord2f(1, 1); glVertex2f(right, top); 
					glTexCoord2f(1, 0); glVertex2f(right, bottom); 
					glTexCoord2f(0, 0); glVertex2f( left, bottom); 
					glEnd();

					glPopMatrix();			
				} 

				else 
				{
					glPushMatrix();
					if(m == 1) glTranslatef( 6.0f, 0.0f, 0.0f);
					else	   glTranslatef(-6.0f, 0.0f, 0.0f);

					if(show_slice == i || show_slice == -1) 
					{
						glBegin(GL_QUADS);
						glTexCoord2f(0, 1); glVertex3f( 5.0f * nnx, seperation * i, 10.0f * nny + shift * i);
						glTexCoord2f(1, 1); glVertex3f(-5.0f * nnx, seperation * i, 10.0f * nny + shift * i);
						glTexCoord2f(1, 0); glVertex3f(-5.0f * nnx, seperation * i,  0.0f * nny + shift * i);
						glTexCoord2f(0, 0); glVertex3f( 5.0f * nnx, seperation * i,  0.0f * nny + shift * i);
						glEnd();
					}

					glPopMatrix();
				}

				unit_interval_shader.deactivate();

				glDisable(GL_TEXTURE_2D);
				glBindTexture(GL_TEXTURE_2D, 0);
				glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			}
		}
	}

	void matrixViewer::glTexFilter() 
	{
		if(smoothTex) 
		{
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		}
		else 
		{
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		}
	}
}
