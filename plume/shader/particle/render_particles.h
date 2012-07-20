 /*
* render_particles.h
* This file is part of CUDAPLUME
*
* Copyright (C) 2012 - Alex 
*
*
* CUDAPLUME is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with CUDAPLUME. If not, see <http://www.gnu.org/licenses/>.
*/
 
 
#ifndef __RENDER_PARTICLES__
#define __RENDER_PARTICLES__

class ParticleRenderer
{
public:
    ParticleRenderer();
    ~ParticleRenderer();

    void setPositions(float *pos, int numParticles);
    void setVertexBuffer(unsigned int vbo, int numParticles);
    void setColorBuffer(unsigned int vbo) { m_colorVBO = vbo; }

    enum DisplayMode
    {
        PARTICLE_POINTS,
        PARTICLE_SPHERES,
        PARTICLE_NUM_MODES
    };

    void display(DisplayMode mode = PARTICLE_POINTS);
    void displayGrid();

    void setPointSize(float size)  { m_pointSize = size; }
    void setParticleRadius(float r) { m_particleRadius = r; }
    void setFOV(float fov) { m_fov = fov; }
    void setWindowSize(int w, int h) { m_window_w = w; m_window_h = h; }

protected: // methods
    void _initGL();
    void _drawPoints();
    GLuint _compileProgram(const char *vsource, const char *fsource);

protected: // data
    float *m_pos;
    int m_numParticles;

    float m_pointSize;
    float m_particleRadius;
    float m_fov;
    int m_window_w, m_window_h;

    GLuint m_program;

    GLuint m_vbo;
    GLuint m_colorVBO;
};

#endif //__ RENDER_PARTICLES__
