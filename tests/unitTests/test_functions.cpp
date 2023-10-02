#include "test_functions.h"
test_functions::test_functions(WINDSGeneralData *WGD, TURBGeneralData *TGD, const std::string &function_type)
{
  std::cout << "[Test Functions]\t setting test functions" << std::endl;
  if (function_type == "linear") {
    u_test_function = new test_function_linearY(WGD);
    v_test_function = new test_function_linearZ(WGD);
    w_test_function = new test_function_linearX(WGD);
    c_test_function = new test_function_linearZ(WGD);
  } else if (function_type == "trig") {
    u_test_function = new test_function_trig(WGD);
    v_test_function = new test_function_trig(WGD);
    w_test_function = new test_function_trig(WGD);
    c_test_function = new test_function_trig(WGD);
  } else {
  }
  setTestValues(WGD, TGD);
  std::cout << "[Test Functions]\t done" << std::endl;
}

void test_functions::setTestValues(WINDSGeneralData *WGD, TURBGeneralData *TGD)
{
  // uv on vertical face -> k=0...nz-2
  for (int k = 0; k < WGD->nz - 1; k++) {
    for (int j = 0; j < WGD->ny - 1; j++) {
      for (int i = 0; i < WGD->nx - 1; i++) {
        int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u[faceID] = u_test_function->val(i * WGD->dx, WGD->y[j], WGD->z[k]);
      }
    }
    for (int j = 0; j < WGD->ny - 1; j++) {
      for (int i = 0; i < WGD->nx - 1; i++) {
        int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        // WGD->v[faceID] = cos(a * WGD->x[i]) + cos(b * j * WGD->dy) + sin(c * WGD->z[k]);
        // WGD->v[faceID] = a * WGD->x[i] + b * j * WGD->dy + c * WGD->z[k];
        WGD->v[faceID] = v_test_function->val(WGD->x[i], j * WGD->dy, WGD->z[k]);
      }
    }
  }

  // w on horizontal face -> k=0...nz-1
  for (int k = 0; k < WGD->nz - 1; k++) {
    for (int j = 0; j < WGD->ny - 1; j++) {
      for (int i = 0; i < WGD->nx - 1; i++) {
        int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->w[faceID] = w_test_function->val(WGD->x[i], WGD->y[j], WGD->z_face[k]);
      }
    }
  }

  // cell center-> k=0...nz-2
  for (int k = 0; k < WGD->nz - 2; k++) {
    for (int j = 0; j < WGD->ny - 1; j++) {
      for (int i = 0; i < WGD->nx - 1; i++) {
        int cellID = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
        TGD->txx[cellID] = c_test_function->val(WGD->x[i], WGD->y[j], WGD->z[k]);
        TGD->txy[cellID] = c_test_function->val(WGD->x[i], WGD->y[j], WGD->z[k]);
        TGD->txz[cellID] = c_test_function->val(WGD->x[i], WGD->y[j], WGD->z[k]);
        TGD->tyy[cellID] = c_test_function->val(WGD->x[i], WGD->y[j], WGD->z[k]);
        TGD->tyz[cellID] = c_test_function->val(WGD->x[i], WGD->y[j], WGD->z[k]);
        TGD->tzz[cellID] = c_test_function->val(WGD->x[i], WGD->y[j], WGD->z[k]);

        TGD->div_tau_x[cellID] = c_test_function->val(WGD->x[i], WGD->y[j], WGD->z[k]);
        TGD->div_tau_y[cellID] = c_test_function->val(WGD->x[i], WGD->y[j], WGD->z[k]);
        TGD->div_tau_z[cellID] = c_test_function->val(WGD->x[i], WGD->y[j], WGD->z[k]);

        TGD->CoEps[cellID] = c_test_function->val(WGD->x[i], WGD->y[j], WGD->z[k]);
      }
    }
  }
}
