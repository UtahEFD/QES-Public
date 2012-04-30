#ifndef MATRIXVIEWER_H
#define MATRIXVIEWER_H

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "../util/matrixIO.h"

#include "../util/minmax.h"
#include "../util/GLSL.h"
#include "../util/viewer.h"


extern "C" 
{
	void cudaAbsDiff
	(
		float* d_err, 
		float* d_p1, float* d_p2, 
		unsigned int size
	);

	void cudaRelDiff
	(
		float* d_rltv_dff, float* d_p1, float* d_p2, 
		int size, float tol
	);

	void cudaMapMatrixToIntervalWithDeviceLimits
	(
		float* d_mat, int size, 
		float a, float b, 
		float* d_min, float* d_max
	);

	void cudaMax(float* d_mat, int size, float* d_max);
	void cudaMin(float* d_mat, int size, float* d_min);
}

namespace QUIC 
{

	class matrixViewer : public viewer 
	{
		
		public:
			matrixViewer(float* M_1, float* M_2, int _nx, int _ny, int _nz);
			~matrixViewer();

			void glDrawMatrixView();
			
			void setSeperation(const float);
			float getSeperation();

			void setShift(const float);
			float getShift();

			void setAlpha(float const&);
			float getAlpha() const;

			void showNextSlice();
			void showPreviousSlice();
			int getSliceToShow();

			void setFlat(bool);
			bool isFlat();

			void showRelDiff(bool);
			bool isRelDiff();

			void showDiff();
			void showSeperate();
			bool areSeperate();
			
			void toggleSmoothTex();

			void setThreshold(float);
			float getThreshold();
			
			float getIntervalBound() const;
			
		private:
			int nx;
			int ny;
			int nz;

			float nnx;
			float nny;

			float* d_M1;
			float* d_M2;
			float* d_abs_dif;
			float* d_rel_dif;

			float seperation;
			float shift;
			int show_slice;

			bool flat;
			bool show_abs_dif;
			bool show_rel_dif;
			bool show_euler;
			
			float alpha;
			float old_alpha;
			
			float M;

			GLuint* pbos;
			GLuint* texs;

			bool smoothTex;
			float threshold;

			GLSLObject difference_shader;
			GLint uniform_difference_tex;
			GLfloat uniform_difference_threshold;
			GLfloat uniform_difference_alpha;

			GLSLObject unit_interval_shader;
			GLint uniform_unitinterval_tex;
			GLfloat uniform_unitinterval_alpha;

			void initOpenGL();
			void cleanUpOpenGL();

			void glDrawDiff();
			void glDrawSeperate();

			void glTexFilter();
	};
}

#endif
