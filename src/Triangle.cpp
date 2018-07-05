#include "Triangle.h"


#define G 0.0f
#define H 0.0f
#define I 1.0f

float Triangle::getHeightTo(float x, float y)
{
  float t, beta, gamma, M;
  float A,B,C,D,E,F,J,K,L;

  A = (*a)[0] - (*b)[0];   D = (*a)[0] - (*c)[0];   J = (*a)[0] - x;
  B = (*a)[1] - (*b)[1];   E = (*a)[1] - (*c)[1];   K = (*a)[1] - y;
  C = (*a)[2] - (*b)[2];   F = (*a)[2] - (*c)[2];   L = (*a)[2];
  
  float EIHF = (E * I - H * F);
  float GFDI = (G * F - D * I);
  float DHEG = (D * H - E * G);
  M = A * EIHF + B * GFDI + C * DHEG;

  float AKJB = (A * K - J * B);
  float JCAL = (J * C - A * L);
  float BLKC = (B * L - K * C);

  t = -1 * ( F * AKJB + E * JCAL + D * BLKC) / M;

  if (t < 0.0f)
    return -1.0f;

  gamma = ( I * AKJB + H * JCAL + G * BLKC) / M;

  if (gamma < 0 || gamma > 1)
    return -1.0f;

  beta = (J * EIHF + K * GFDI + L * DHEG) / M;

  if ( beta < 0  || beta > 1 - gamma)
    return -1.0;
  
  return t;
  
}

void Triangle::getBoundaries(float& xmin, float& xmax, float& ymin, float& ymax, float& zmin, float& zmax)
{
    xmin = LOWEST_OF_THREE((*a)[0], (*b)[0], (*c)[0]);
    xmax = HIGHEST_OF_THREE((*a)[0], (*b)[0], (*c)[0]);
    ymin = LOWEST_OF_THREE((*a)[1], (*b)[1], (*c)[1]);
    ymax = HIGHEST_OF_THREE((*a)[1], (*b)[1], (*c)[1]);
    zmin = LOWEST_OF_THREE((*a)[2], (*b)[2], (*c)[2]);
    zmax = HIGHEST_OF_THREE((*a)[2], (*b)[2], (*c)[2]);
}

void Triangle::parseValues()
{
	parseElement< Vector3<float> >(true, a, "a");
	parseElement< Vector3<float> >(true, b, "b");
	parseElement< Vector3<float> >(true, c, "c");
}