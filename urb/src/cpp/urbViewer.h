/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Debugging / Visualizing the state of the iteration scheme as it
*         proceeds through iterations.
*/

#ifndef URBVIEWER_H
#define URBVIEWER_H

#include "urbCUDA.h"
#include "urbSetup.h"
#include "urbParser.h"

#include "../util/viewer.h"
#include "util/GLSL.h"

extern "C" 
{
	void cudaMapMatrixToInterval(float* d_mat, int size, float a, float b);

	void cudaMapMatrixToIntervalWithDeviceLimits
	(
		float* d_mat, int size, 
		float a, float b, 
		float* d_min, float* d_max
	);

	void cudaMax(float* d_array, int size, float* d_max);

	void cudaMin(float* d_array, int size, float* d_min);

	void cudaPack
	(
		float* d_packee, 
		float* d_per1, float* d_per2, float* d_per3, float* d_per4, 
		int size
	);

	void showError(char const* loc);
}

namespace QUIC 
{

	class urbViewer : public viewer, public urbModule
	{
		public:
			urbViewer();
			virtual ~urbViewer();
			
			// Requires that the urbModule portion of the object has be setup with
			// urbSetup
			void initialize();
			void loadShaders(std::string directory = "../Common/Shaders/");
			
			void nextIteration();
			void glDrawData();

			void setSeperation(float const&);
			float getSeperation() const;

			void setShift(float const&);
			float getShift() const;

			void setAlpha(float const&);
			float getAlpha() const;

			void showNextSlice();
			void showPrevSlice();
			int getSliceToShow() const;

			void setFlat(bool const&);
			bool isFlat() const;

			void setEuler(bool const&);
			bool isEuler() const;
			
			float getIntervalBound() const;
			
			void dumpIteration(std::string directory) const;

		private:

			float nnx;
			float nny;
			float ngx;
			float ngy;

			float seperation;
			float shift;
			int show_slice;

			bool flat;
			bool view_euler;

			float alpha;
			float old_alpha;

			float M;

			GLuint* l_pbos;
			GLuint* l_texs;

			GLuint* e_pbos;
			GLuint* e_texs;

			float* t_p1;

			float* t_vels;
			float* t_u;
			float* t_v;
			float* t_w;
			
			GLSLObject lagrange_shader;
			GLint uniform_lagrange_tex;
			GLfloat uniform_lagrange_alpha;
			
			GLSLObject euler_shader;
			GLint uniform_euler_u_tex;
			GLfloat uniform_euler_alpha;

			void loadLagragianTextures();
			void loadEulerTextures();

			void getFlatCorners(int const&, float&, float&, float&, float&) const;

			void glDrawLagragian();
			void glDrawEuler();
			void glDrawLabels() const;

			void initOpenGL();
			void cleanUpOpenGL();
	};
}

#endif
