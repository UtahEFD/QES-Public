#include "CanopyROC.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"
#include "TURBGeneralData.h"


void CanopyROC::orthog_vec(float d, float P[2], float Lx[2], float Ly[2], float row_ortho[2])
{

  float alph = 1 + pow((Ly[1] - Ly[0]) / (Lx[1] - Lx[0]), 2);
  row_ortho[1] = sqrt(pow(d, 2) / alph);
  row_ortho[0] = sqrt(pow(d, 2) - pow(row_ortho[1], 2));
  float dist = 100000;
  bool orthCase = 0;
  float orth_test[2];
  float o_signed[2] = { 0, 0 };

  orth_test[0] = P[0] + row_ortho[0];
  orth_test[1] = P[1] + row_ortho[1];
  if (P2L(orth_test, Lx, Ly) < dist) {
    o_signed[0] = row_ortho[0];
    o_signed[1] = row_ortho[1];
    dist = P2L(orth_test, Lx, Ly);
    orthCase = 1;
  }
  orth_test[0] = P[0] - row_ortho[0];
  orth_test[1] = P[1] + row_ortho[1];
  if (P2L(orth_test, Lx, Ly) < dist) {
    o_signed[0] = -row_ortho[0];
    o_signed[1] = row_ortho[1];
    dist = P2L(orth_test, Lx, Ly);
    orthCase = 1;
  }

  orth_test[0] = P[0] + row_ortho[0];
  orth_test[1] = P[1] - row_ortho[1];
  if (P2L(orth_test, Lx, Ly) < dist) {
    o_signed[0] = row_ortho[0];
    o_signed[1] = -row_ortho[1];
    dist = P2L(orth_test, Lx, Ly);
    orthCase = 1;
  }

  orth_test[0] = P[0] - row_ortho[0];
  orth_test[1] = P[1] - row_ortho[1];
  if (P2L(orth_test, Lx, Ly) < dist) {
    o_signed[0] = -row_ortho[0];
    o_signed[1] = -row_ortho[1];
    dist = P2L(orth_test, Lx, Ly);
    orthCase = 1;
  }

  if (orthCase == 0) {
    std::cerr << "ERROR unable to find valid row-orthogonal vector in ROC model" << std::endl;
    exit(EXIT_FAILURE);
  }

  row_ortho[0] = o_signed[0];
  row_ortho[1] = o_signed[1];
}


void CanopyROC::setCellFlags(const WINDSInputData *WID, WINDSGeneralData *WGD, int building_id)
{
  // When THIS canopy calls this function, we need to do the
  // following:
  // readCanopy(nx, ny, nz, landuse_flag, num_canopies, lu_canopy_flag,
  // canopy_atten, canopy_top);

  // this function need to be called to defined the boundary of the canopy and the icellflags
  // setCanopyGrid(WGD, building_id);
  float ray_intersect;
  unsigned int num_crossing, vert_id, start_poly;

  // Find out which cells are going to be inside the polygone
  // Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
  // Check the center of each cell, if it's inside, set that cell to building
  for (auto j = j_start; j < j_end; j++) {
    // Center of cell y coordinate
    float y_cent = (j + 0.5) * WGD->dy;
    for (auto i = i_start; i < i_end; i++) {
      float x_cent = (i + 0.5) * WGD->dx;
      // Node index
      vert_id = 0;
      start_poly = vert_id;
      num_crossing = 0;
      while (vert_id < polygonVertices.size() - 1) {
        if ((polygonVertices[vert_id].y_poly <= y_cent && polygonVertices[vert_id + 1].y_poly > y_cent)
            || (polygonVertices[vert_id].y_poly > y_cent && polygonVertices[vert_id + 1].y_poly <= y_cent)) {
          ray_intersect = (y_cent - polygonVertices[vert_id].y_poly) / (polygonVertices[vert_id + 1].y_poly - polygonVertices[vert_id].y_poly);
          if (x_cent < (polygonVertices[vert_id].x_poly + ray_intersect * (polygonVertices[vert_id + 1].x_poly - polygonVertices[vert_id].x_poly))) {
            num_crossing += 1;
          }
        }
        vert_id += 1;
        if (polygonVertices[vert_id].x_poly == polygonVertices[start_poly].x_poly
            && polygonVertices[vert_id].y_poly == polygonVertices[start_poly].y_poly) {
          vert_id += 1;
          start_poly = vert_id;
        }
      }

      // if num_crossing is odd = cell is oustside of the polygon
      // if num_crossing is even = cell is inside of the polygon
      if ((num_crossing % 2) != 0) {
        int icell_2d = i + j * (WGD->nx - 1);

        if (WGD->icellflag_footprint[icell_2d] == 0) {
          // a  building exist here -> skip
        } else {
          // save the (x,y) location of the canopy
          canopy_cell2D.push_back(icell_2d);
          // set the footprint array for canopy
          // WGD->icellflag_footprint[icell_2d] = getCellFlagCanopy();

          // Define start index of the canopy in z-direction
          for (size_t k = 1u; k < WGD->z.size(); k++) {
            if (WGD->terrain[icell_2d] + base_height <= WGD->z[k]) {
              WGD->canopy->canopy_bot_index[icell_2d] = k;
              WGD->canopy->canopy_bot[icell_2d] = WGD->terrain[icell_2d] + base_height;
              WGD->canopy->canopy_base[icell_2d] = WGD->z_face[k - 1];
              break;
            }
          }


          // Define end index of the canopy in z-direction
          for (size_t k = 0u; k < WGD->z.size(); k++) {
            if (WGD->terrain[icell_2d] + H < WGD->z[k + 1]) {
              WGD->canopy->canopy_top_index[icell_2d] = k + 1;
              WGD->canopy->canopy_top[icell_2d] = WGD->terrain[icell_2d] + H;
              break;
            }
          }

          // Define the height of the canopy
          WGD->canopy->canopy_height[icell_2d] = WGD->canopy->canopy_top[icell_2d] - WGD->canopy->canopy_bot[icell_2d];

          // define icellflag @ (x,y) for all z(k) in [k_start...k_end]
          for (auto k = WGD->canopy->canopy_bot_index[icell_2d]; k < WGD->canopy->canopy_top_index[icell_2d]; k++) {
            int icell_3d = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
            if (WGD->icellflag[icell_3d] != 0 && WGD->icellflag[icell_3d] != 2) {
              // Canopy cell
              // WGD->icellflag[icell_3d] = getCellFlagCanopy();
              // WGD->canopy->canopy_atten_coeff[icell_3d] = attenuationCoeff;
              WGD->canopy->icanopy_flag[icell_3d] = building_id;
              canopy_cell3D.push_back(icell_3d);
            }
          }
        }// end define icellflag!
      }
    }
  }

  // check if the canopy is well defined
  if (canopy_cell2D.size() == 0) {
    k_start = 0;
    k_end = 0;
  } else {
    k_start = WGD->nz - 1;
    k_end = 0;
    for (size_t k = 0u; k < canopy_cell2D.size(); k++) {
      if (WGD->canopy->canopy_bot_index[canopy_cell2D[k]] < k_start)
        k_start = WGD->canopy->canopy_bot_index[canopy_cell2D[k]];
      if (WGD->canopy->canopy_top_index[canopy_cell2D[k]] > k_end)
        k_end = WGD->canopy->canopy_top_index[canopy_cell2D[k]];
    }
  }

  // check of illegal definition.
  if (k_start > k_end) {
    std::cerr << "ERROR in tree definition (k_start > k end)" << std::endl;
    exit(EXIT_FAILURE);
  }


  if (ceil(1.5 * k_end) > WGD->nz - 1) {
    std::cerr << "ERROR domain too short for tree method" << std::endl;
    exit(EXIT_FAILURE);
  }

  nx_canopy = (i_end - i_start - 1) + 2;
  ny_canopy = (j_end - j_start - 1) + 2;

  // Find nearest sensor (so we use the correct 1/L in the vo parameterization)
  float sensor_distance = 99999;
  float curr_dist;
  int num_sites = WID->metParams->sensors.size();
  int nearest_sensor_i = 0;// index of the sensor nearest to ROC block centroid
  for (auto i = 0; i < num_sites; i++) {
    curr_dist = sqrt(pow((building_cent_x - WID->metParams->sensors[i]->site_xcoord), 2) + pow((building_cent_y - WID->metParams->sensors[i]->site_ycoord), 2));
    if (curr_dist < sensor_distance) {
      sensor_distance = curr_dist;
      nearest_sensor_i = i;
    }
  }

  rL = WID->metParams->sensors[nearest_sensor_i]->TS[0]->site_one_overL;// uses first time step value only - figure out how to get current time step inside this function

  z0_site = WID->metParams->sensors[nearest_sensor_i]->TS[0]->site_z0;

  a_obf = pow(beta, 0.4);

  return;
}


void CanopyROC::canopyVegetation(WINDSGeneralData *WGD, int building_id)
{

  std::vector<float> u0_modified, v0_modified;
  std::vector<int> u0_mod_id, v0_mod_id;

  int icell_face = i_building_cent + j_building_cent * WGD->nx + (WGD->nz - 5) * WGD->nx * WGD->ny;
  float u0_uw = WGD->u0[icell_face];// u velocity at the centroid, 5 nodes from the top of domain (avoids effect of nearby wakes)
  float v0_uw = WGD->v0[icell_face];// v velocity at the centroid, 5 nodes from the top of domain

  upwind_dir_unit = atan2(v0_uw, u0_uw) * 180. / M_PI + 180.;// degrees on the unit circle


  // Create unit wind vector
  float M0_uw, u0n_uw, v0n_uw;
  M0_uw = sqrt(pow(u0_uw, 2) + pow(v0_uw, 2));
  u0n_uw = u0_uw / M0_uw;
  v0n_uw = v0_uw / M0_uw;

  // Convert rowAngle (compass rose) to unit circle degrees
  float rowAngle_u = -(rowAngle - 90);
  cosA = cos(rowAngle * M_PI / 180);
  sinA = sin(rowAngle * M_PI / 180);

  float rd[2] = { static_cast<float>(cos(rowAngle_u * M_PI / 180.)),
                  static_cast<float>(sin(rowAngle_u * M_PI / 180.)) };

  float Rx[2] = { building_cent_x, building_cent_x + rd[0] };
  float Ry[2] = { building_cent_y, building_cent_y + rd[1] };

  float uwdV = -1000000;// a low number
  float dwdV = 1000000;// a high number
  float curr_V[2];
  float d2V;
  float wwdV_current;// windward distance to current vertex
  int idS = 0;
  int idE = polygonVertices.size() - 2;
  for (int id = 0; id < polygonVertices.size() - 1; id++) {
    curr_V[0] = polygonVertices[id].x_poly;
    curr_V[1] = polygonVertices[id].y_poly;
    d2V = P2L(curr_V, Rx, Ry);

    // Find orthogonal vector (of length d2V, starting at the current vertex curr_V and ending on line Rx,Ry
    float orthog[2];
    orthog_vec(d2V, curr_V, Rx, Ry, orthog);

    // Find windward distance of current vertex
    wwdV_current = u0n_uw * orthog[0] + v0n_uw * orthog[1];

    if (wwdV_current > uwdV) {
      uwdV = wwdV_current;
      idS = id;// index of vertex where rows "start" from (i.e. the most upwind vertex)
    }
    if (wwdV_current < dwdV) {
      dwdV = wwdV_current;
      idE = id;// index of vertex where rows "end" at (i.e. the most downwind vertex)
    }
  }

  // Row-parallel line intersecting the upwind-est vertex
  Rx_o[0] = polygonVertices[idS].x_poly;
  Rx_o[1] = polygonVertices[idS].x_poly + rd[0];// x-components of upwind-est row vector
  Ry_o[0] = polygonVertices[idS].y_poly;
  Ry_o[1] = polygonVertices[idS].y_poly + rd[1];// y-components "   "   "

  // Find streamwise distance between rows (d_dw):
  betaAngle = acos(rd[0] * u0n_uw + rd[1] * v0n_uw) * 180 / M_PI;// acute angle between wd and row vector, in degrees

  if (betaAngle > 90) {
    betaAngle = 180 - betaAngle;
  }
  if (abs(betaAngle) < 0.5) {
    betaAngle = 0.5;// because we have sin(betaAngle) in a couple denominators
  }
  d_dw = (rowSpacing - rowWidth) / sin(betaAngle * M_PI / 180);

  // Find appropriate reference node for finding U_h (at the centroid of the ROC block)
  // Ref velocity for top shear zone
  int k_top = 0;
  while (WGD->z_face[k_top] < (H + WGD->terrain[i_building_cent + j_building_cent * (WGD->nx - 1)])) {
    k_top += 1;
  }
  icell_face = i_building_cent + j_building_cent * WGD->nx + k_top * WGD->nx * WGD->ny;
  float u0_h = WGD->u0[icell_face];
  float v0_h = WGD->v0[icell_face];
  float rd_o[2] = { rd[1], -rd[0] };// row-orthogonal unit vector
  float M0_h = abs(u0_h * rd_o[0] + v0_h * rd_o[1]);// the component of the wind vector in the row-orthogonal direction <-- should this just be total wind speed, not row-ortho speed?

  // Ref velocity for bottom shear zone
  int k_bot = 0;
  while (WGD->z_face[k_bot] < (understory_height + WGD->terrain[i_building_cent + j_building_cent * (WGD->nx - 1)])) {
    k_bot += 1;
  }

  icell_face = i_building_cent + j_building_cent * WGD->nx + k_bot * WGD->nx * WGD->ny;
  float u0_uh = WGD->u0[icell_face];
  float v0_uh = WGD->v0[icell_face];
  float M0_uh = abs(u0_uh * rd_o[0] + v0_uh * rd_o[1]);


  // Entrance length and number of rows in entry region
  float Cd = 0.5;// Bailey 2013
  float Cd_ud = 1.98;// flat plate in perpendicular flow
  // float Cd = 2 * pow(uustar / M0_h, 2);// Heilman 1996 eq. 5
  // float Cd = 0.2;// Chahine 2014
  // float LAI_row = 5.57;
  // float LAI_eff = LAI_row * (rowWidth / rowSpacing);
  // float LAD_eff = LAI_eff / H;

  // float LAD_eff = 0.6667;
  // float LAD_avg = 4.3138;
  // float fenceThickness = H * (8. / 72.);

  // Which aerodynamic porosity model to use
  if (thinFence == 1) {
    a_obf = beta;
    float A_frontal = (1 - beta) * pow(H, 2);
    LAD_eff = (1 - a_obf) / rowSpacing;
  } else {
    a_obf = pow(beta, 0.4);
  }

  float L_c = pow(Cd * LAD_eff, -1);

  float l_e = L_c * log((M0_h / uustar) * (H / L_c));
  float N_e_float = ceil(l_e / rowSpacing);
  N_e = (int)N_e_float;
  N_e += 2;// tuned to Torkelson PIV

  // Spread rate calculations
  float udelt = (1 - pow(a_obf, N_e + 1));
  float uave = (1 + pow(a_obf, N_e + 1)) / 2;

  float spreadclassicmix = 0.14 * udelt / uave;

  float spreadupstream_top = 2 * stdw / M0_h;
  float spreadupstream_bot = 2 * stdw / M0_uh;


  // Avoid divide-by-zero for the no-understory case (where M0_uh will be 0)
  if (abs(M0_uh) < 0.000001) {
    spreadupstream_bot = spreadupstream_top;
  }

  spreadrate_top = sqrt(pow(spreadclassicmix, 2) + pow(spreadupstream_top, 2));
  spreadrate_bot = sqrt(pow(spreadclassicmix, 2) + pow(spreadupstream_bot, 2));

  // Shear zone origin (initialized to H but parameterized later)
  szo_top = H;
  szo_bot = understory_height;

  // Upwind displacement zone parameters (for no understory)
  float l_ud = 0.5 * H;// horizontal length of UD zone (distance from the vine that it extends to in the windward direction)
  float z_ud;// upper bound of UD zone at current i,j location, if no understory
  float H_ud = 0.6 * H;// max height of UD (occurs where it attaches at the windward side of a vine)
  float br = 0;// blend rate, used in the blending function that smoothes the UD zone into the wake
  float x_ud;// horizontal distance in the UD zone. Measured orthogonally from the "upwindest" side of the vine
  float u_def = 0.0;// velocity deficit at a point in the UD zone
  // UD zone variables (if understory)
  float z_udTOP;// upper bound of UD zone, if there's an understory
  float z_udBOT;// lower bound of UD zone, if there's an understory
  float brTOP = 0;// blend rate for top half of UD zone, if there's an understory
  float brBOT = 0;// blend rate for bottom half of UD zone, if there's an understory
  float a_ud;// a constant


  // General geometry parameters
  float dv_c;// distance of current node to upwind-est vertex
  float dv_c_dw;// distance of current node to downwind-est vertex
  float N;// number of rows upwind of current point
  float ld;// "local distance", the orthogonal distance from the current i,j point to the upwind edge of the closest row in the upwind direction
  int icell_2d, icell_2d_canopy;
  float a_exp;// a blending constant that smooths the transition from wake to upwind flow
  float a_uwv[N_e + 1];// the attenuation due to one row (to be raised to N_e, where N_e is number of rows in entry region)
  float tkeFacU_uwv[N_e + 1];


  int nx = WGD->nx;
  int ny = WGD->ny;
  int nz = WGD->nz;
  int np_cc_v = (nz - 1) * (ny - 1) * (nx - 1);
  int np_fc_v = nz * ny;
  tkeFac.resize(np_cc_v, 0);
  vineLm.resize(np_cc_v, 0);
  float a_uw = 1;


  // DEBUG VARIABLES
  bool BF_flag = 1;
  int UD_zone_flag = 1;

  // BEGIN MAIN PARAMETERIZATION LOOPS
  for (auto j = 0; j < ny_canopy; j++) {
    for (auto i = 0; i < nx_canopy; i++) {
      // icell_2d = i + j * nx_canopy;
      icell_2d = (i + i_start) + (j + j_start) * (WGD->nx - 1);

      // base of the canopy
      z_b = WGD->canopy->canopy_base[icell_2d];


      // Row-orthogonal distance between current point and the upwind-est row, dv_c
      float cxy[2] = { (i - 1 + i_start) * WGD->dx, (j - 1 + j_start) * WGD->dy };// current x position
      dv_c = abs(P2L(cxy, Rx_o, Ry_o));

      // Row-orthogonal distance between current point and the downwind-est row
      float Rx_o_dw[2] = { polygonVertices[idE].x_poly, polygonVertices[idE].x_poly + rd[0] };// x-components of downwind-est row vector
      float Ry_o_dw[2] = { polygonVertices[idE].y_poly, polygonVertices[idE].y_poly + rd[1] };// y-components "   "   "
      dv_c_dw = abs(P2L(cxy, Rx_o_dw, Ry_o_dw));

      // calculate number of rows upwind of this point, N (not including nearest upwind row)
      N = floor(dv_c / rowSpacing);

      // calculate "local row-orthogonal distance" from nearest upwind row
      ld = dv_c - N * rowSpacing;

      // calculate "local streamwise distance" from nearest row
      float d_dw_local = (ld - rowWidth) / sin(betaAngle * M_PI / 180.0f);
      // find blending factor "a_exp" at the current i,j location (a function of d_dw_local)
      a_exp = exp((log(0.01f) / (7.5f * H)) * d_dw_local);

      // find k-node of mid-canopy at current i,j location
      k_mid = 0;
      for (auto k = WGD->canopy->canopy_bot_index[icell_2d]; k < WGD->canopy->canopy_top_index[icell_2d]; k++) {
        if (WGD->z[k] > (z_b + understory_height + (H - understory_height) / 2.0f)) {
          break;
        }
        k_mid = k;
      }

      // Find shear zone origin based on Greg Torkelson's slopes from PIV (S_omega1, Table 2)
      // Data are fit to a modified Michaelis-Menten curve: fit_a*x / (fit_b * x) + fit_c, where x is rowSpacing/H
      float fit_a, fit_b, fit_c;

      // fit_a = 0.7583;//with first data point set at 1
      // fit_b = 1.3333;
      // fit_c = -0.7550;
      fit_a = 1.3945;// with first data point set at 2.5/2.16=1.1574
      fit_b = 0.4209;
      fit_c = -1.4524;

      szo_slope = fit_a * (rowSpacing / H) / ((rowSpacing / H) + fit_b) + fit_c;
      // float szo_slope = -0.43;
      szo_top = std::max(szo_slope * (ld - rowWidth) + H, 0.0f);
      szo_top_uw = std::max(szo_slope * (rowSpacing - rowWidth) + H, 0.0f);// origin right before the next row, e.g. at a distance of "rowSpacing" orthogonally from nearest upwind row

      z_mid = (H - understory_height) / 2 + understory_height;


      // Calculate geometry quantities for UD zone
      x_ud = rowSpacing - ld;// right now I'm using the row-orthogonal distance, not streamwise distance (possibly change in future)
      if (understory_height == 0. && x_ud <= l_ud) {
        z_ud = H_ud * sqrt(abs(1 - x_ud / l_ud));
        br = log(0.01) / (z_ud);// reaches 0.01 at ground
      } else if (understory_height > 0. && x_ud <= l_ud) {
        a_ud = -l_ud * pow(0.5 * (H - understory_height), -2);
        z_udBOT = (a_ud * (H + understory_height) + pow(pow(a_ud, 2) * pow(H + understory_height, 2) - 4 * a_ud * (a_ud * H * understory_height - x_ud), 0.5)) / (2 * a_ud);
        z_udTOP = (a_ud * (H + understory_height) - pow(pow(a_ud, 2) * pow(H + understory_height, 2) - 4 * a_ud * (a_ud * H * understory_height - x_ud), 0.5)) / (2 * a_ud);
        brTOP = log(0.01) / (z_udTOP - z_mid);
        brBOT = log(0.01) / (z_mid - z_udBOT);
      }


      // An arbitrary ref height for v_c parameterization
      zref = 10.;// arbitrarily choose 10m
      zref_k = 0;
      for (auto k = WGD->canopy->canopy_bot_index[icell_2d]; k < WGD->nz; k++) {
        if (WGD->z[k] > (zref + z_b)) {
          break;
        }
        zref_k = k;
      }

      // BEGIN PARAMETERIZATIONS

      if (WGD->canopy->canopy_bot_index[icell_2d] < WGD->canopy->canopy_top_index[icell_2d]) {// if i'm inside the canopy (ROC block) polygon
        icell_face_ref = (i - 1 + i_start) + (j - 1 + j_start) * WGD->nx + zref_k * WGD->nx * WGD->ny;

        // Calculate the stability correction quantities that aren't height dependent (for vo parameterization)
        if (rL >= 0.) {
          psiH = -5.0 * (H - d_v) * rL;
          dpsiH = -5.0 * rL;

          psiRef = -5.0 * (zref - d_v) * rL;
          dpsiRef = -5.0 * rL;


        } else {
          exH = pow(1. - 15. * (H - d_v) * rL, .25);
          psiH = log((0.5 * (1. + pow(exH, 2))) * pow(0.5 * (1. + exH), 2)) - 2. * atan(exH) + M_PI * 0.5;

          dexH = -15. / 4. * rL * pow(1. - 15. * (H - d_v) * rL, -0.75);

          dpsiH = dexH * (((exH * (0.5 * (1 + exH)) + 0.5 * (1 + pow(exH, 2))) / (0.5 * (1 + pow(exH, 2)) * (0.5 * (1 + exH)))) - 2. / (1. + pow(exH, 2.)));

          exRef = pow(1. - 15. * (zref - d_v) * rL, .25);
          psiRef = log((0.5 * (1. + pow(exRef, 2))) * pow(0.5 * (1. + exRef), 2)) - 2. * atan(exRef) + M_PI * 0.5;

          dexRef = -15. / 4. * rL * pow(1. - 15. * (zref - d_v) * rL, -0.75);
          dpsiRef = dexRef * (((exRef * (0.5 * (1 + exRef)) + 0.5 * (1 + pow(exRef, 2))) / (0.5 * (1 + pow(exRef, 2)) * (0.5 * (1 + exRef)))) - 2. / (1. + pow(exRef, 2.)));
        }


        vref_c = sinA * WGD->u0[icell_face_ref] + cosA * WGD->v0[icell_face_ref];// The sign of vref_c (the rotated reference velocity) is preserved (rather than abs() ) so that rotation of the final parameterized v0_c is correct. It determines the sign of ustar_v_c which determines sign of vH_c which determines sign of v_c
        ustar_v_c = WGD->vk * vref_c / (log((zref - d_v) / z0_site) - psiRef);// d_v (displacement height from Nate's data). This is the ustar_v_c that should cause the v profile to be vref_c at zref. it's zref-d_v and not just zref like in Pardyjak 2008 because the v profile that will eventually be prescribed in the canopy is displaced.
        vH_c = ustar_v_c / WGD->vk * (log((H - d_v) / z0_site) - psiH);
        a_v = (H / abs(vH_c + 0.00001)) * (abs(ustar_v_c) / WGD->vk) * (1 / (H - d_v) - dpsiH);// abs() here because the attenuation coefficient should be always positive. add 0.00001 to avoid divide by zero


        // MAIN Z-LOOP
        for (auto k = WGD->canopy->canopy_bot_index[icell_2d]; k < (WGD->nz - 1); k++) {
          z_rel = WGD->z_face[k - 1] - z_b;
          int icell_face = (i - 1 + i_start) + (j - 1 + j_start) * WGD->nx + k * WGD->nx * WGD->ny;
          int icell_cent = (i - 1 + i_start) + (j - 1 + j_start) * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);

          // Rotate u0 and v0 into row-aligned coords
          u_c0 = cosA * WGD->u0[icell_face] - sinA * WGD->v0[icell_face];
          v_c0 = sinA * WGD->u0[icell_face] + cosA * WGD->v0[icell_face];


          // Initialize u_c and v_c so that in the understory, where no parameterization takes place, u_c and v_c have the correct (i.e. unaltered) value
          u_c = u_c0;
          v_c = v_c0;


          // CALCULATE SHELTERING

          // Sparse understory, below mid-canopy
          if (understory_height > 0 && k <= k_mid) {
            if (N <= N_e) {// entry region
              szt_local = spreadrate_bot * (d_dw * N + d_dw_local) + 0.00001;
            } else {// eq region
              szt_local = spreadrate_bot * (d_dw * N_e + d_dw_local) + 0.00001;
            }
            a_local = 1 * (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (szo_bot - z_rel) / szt_local) + 0.5 * (1 + a_obf)) + (1 - a_exp));// there should be a "fac" where the 1* is (matching fac isn't working right now)
            tkeFacU_local = 1 * (a_exp * (0.5 * (1 - pow(a_obf, 1)) * tanh(1.5 * (szo_bot - z_rel) / szt_local) + 0.5 * (1 + pow(a_obf, 1))) + (1 - a_exp));// should be a "fac" where the 1* is
            a_uwv[0] = 1;
            tkeFacU_uwv[0] = 1;
            for (int n = 1; n < N_e + 1; n++) {// create vector for sheltering from all rows in entry region
              szt_uw = spreadrate_bot * (d_dw * n) + 0.00001;
              a_uwv[n] = 1 * (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (szo_bot - z_rel) / szt_uw) + 0.5 * (1 + a_obf)) + (1 - a_exp));// should be a "fac" where the 1* is
              tkeFacU_uwv[n] = 1 * (a_exp * (0.5 * (1 - pow(a_obf, 1)) * tanh(1.5 * (szo_bot - z_rel) / szt_uw) + 0.5 * (1 + pow(a_obf, 1))) + (1 - a_exp));// should be a "fac" where the 1* is
            }

            // Sparse understory, above mid-canopy
          } else if (understory_height > 0 && k > k_mid) {
            if (N <= N_e) {
              szt_local = spreadrate_top * (d_dw * N + d_dw_local) + 0.00001;
            } else {
              szt_local = spreadrate_top * (d_dw * N_e + d_dw_local) + 0.00001;
            }
            a_local = (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (z_rel - szo_top) / szt_local) + 0.5 * (1 + a_obf)) + (1 - a_exp));
            tkeFacU_local = (a_exp * (0.5 * (1 - pow(a_obf, 1)) * tanh(1.5 * (z_rel - szo_top) / szt_local) + 0.5 * (1 + pow(a_obf, 1))) + (1 - a_exp));
            a_uwv[0] = 1;
            tkeFacU_uwv[0] = 1;
            for (int n = 1; n < N_e + 1; n++) {
              szt_uw = spreadrate_top * (d_dw * n) + 0.00001;
              a_uwv[n] = (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (z_rel - szo_top_uw) / szt_uw) + 0.5 * (1 + a_obf)) + (1 - a_exp));
              tkeFacU_uwv[n] = (a_exp * (0.5 * (1 - pow(a_obf, 1)) * tanh(1.5 * (z_rel - szo_top_uw) / szt_uw) + 0.5 * (1 + pow(a_obf, 1))) + (1 - a_exp));
            }

            // No understory space, all heights
          } else if (understory_height == 0) {
            if (N <= N_e) {
              szt_local = spreadrate_top * (d_dw * N + d_dw_local) + 0.00001;
            } else {
              szt_local = spreadrate_top * (d_dw * N_e + d_dw_local) + 0.00001;
            }
            a_local = (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (z_rel - szo_top) / szt_local) + 0.5 * (1 + a_obf)) + (1 - a_exp));
            tkeFacU_local = (a_exp * (0.5 * (1 - pow(a_obf, 1)) * tanh(1.5 * (z_rel - szo_top) / szt_local) + 0.5 * (1 + pow(a_obf, 1))) + (1 - a_exp));
            a_uwv[0] = 1;
            tkeFacU_uwv[0] = 1;
            for (int n = 1; n < N_e + 1; n++) {
              szt_uw = spreadrate_top * (d_dw * n) + 0.00001;
              a_uwv[n] = (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (z_rel - szo_top_uw) / szt_uw) + 0.5 * (1 + a_obf)) + (1 - a_exp));
              tkeFacU_uwv[n] = (a_exp * (0.5 * (1 - pow(a_obf, 1)) * tanh(1.5 * (z_rel - szo_top_uw) / szt_uw) + 0.5 * (1 + pow(a_obf, 1))) + (1 - a_exp));
            }
          }

          if (i + i_start == 130 && j + j_start == 150) {
            // std::cout << "ld: " << ld << " d_dw: " << d_dw << " d_dw_local: " << d_dw_local << " N: " << N << "spreadrate_top: " << spreadrate_top << " szt_local: " << szt_local << " szt_uw: " << szt_uw << " N_e: " << N_e << std::endl;
          }

          // APPLY BLEED FLOW PARAMETERIZATION INSIDE THE ACTUAL VEGETATION
          if (BF_flag == 1 && dv_c > rowSpacing && ld < rowWidth) {// if my x-y position indicates i'm on a vine
            tkeFacU = 1;// tke is tkeMax above the fence (will be overwritten inside the fence)
            if ((z_rel >= understory_height) && (z_rel <= H)) {// if my z position indicates i'm inside the actual vegetation

              // Am I in entry region or /quilibrated region?
              if (N <= N_e) {// i'm in entry region

                WGD->icellflag[icell_cent] = 28;

                u_c = u_c0 * a_obf;
                tkeFacU = pow(a_obf, 1);
                for (int n = 0; n <= N; n++) {
                  u_c *= a_uwv[n];
                  tkeFacU *= tkeFacU_uwv[n];
                }

              }

              else {// if i'm in equilibrated region

                WGD->icellflag[icell_cent] = 28;
                u_c = u_c0 * a_obf;
                tkeFacU = pow(a_obf, 1);
                for (int n = 0; n <= N_e; n++) {
                  u_c *= a_uwv[n];
                  tkeFacU *= tkeFacU_uwv[n];
                }
              }
            }// end bleed flow z if
          }// end bleed flow x-y if


          // APPLY WAKE PARAMETERIZATION
          else {
            WGD->icellflag[icell_cent] = 30;

            // Am I in entry region or equilibrated region?
            if (N <= N_e && dv_c > rowSpacing) {// i'm in entry region

              u_c = u_c0 * a_local;
              tkeFacU = tkeFacU_local;
              for (int n = 0; n <= N; n++) {
                u_c *= a_uwv[n];
                tkeFacU *= tkeFacU_uwv[n];
              }

              szt_Lm = spreadrate_top * (d_dw_local) + 0.00001;
              if (z_rel > szo_top - 0.5 * szt_Lm && z_rel < szo_top + 0.5 * szt_Lm) {
                vineLm[icell_cent] = szt_Lm;
                // vineLm[icell_cent] = spreadrate_top * (d_dw * N + d_dw_local) + 0.00001;
              }

            }

            else if (N > N_e && dv_c > rowSpacing) {// if i'm in equilibrated region

              u_c = u_c0 * a_local;
              tkeFacU = tkeFacU_local;
              for (int n = 0; n <= N_e; n++) {
                u_c *= a_uwv[n];
                tkeFacU *= tkeFacU_uwv[n];
              }

              szt_Lm = spreadrate_top * (d_dw_local) + 0.00001;
              if (z_rel > szo_top - 0.5 * szt_Lm && z_rel < szo_top + 0.5 * szt_Lm) {
                vineLm[icell_cent] = szt_Lm;
                // vineLm[icell_cent] = spreadrate_top * (d_dw * N_e + d_dw_local) + 0.00001;
              }
            }

            // APPLY UD ZONE PARAMETERIZATION
            // Define velocity deficit based on thin (fence) or thick (vegetative) row
            if (thinFence == 1) {
              u_def = -l_ud / rowWidth * a_obf * u_c * Cd_ud * (1 - beta);
            } else {
              u_def = -Cd * LAD_avg * a_obf * u_c * l_ud;
            }

            if (understory_height == 0) {// if no understory space
              if (UD_zone_flag && dv_c_dw > rowSpacing && ((rowSpacing - ld) < l_ud) && (z_rel <= z_ud)) {// if i'm ALSO in the UD zone
                WGD->icellflag[icell_cent] = 31;

                u_c = u_c + u_def * (1 - exp(br * (z_ud - z_rel)));
              }
            } else {// else if there's an understory

              if (UD_zone_flag && dv_c_dw > rowSpacing && ((rowSpacing - ld) < l_ud) && (z_rel <= z_udTOP) && (z_rel >= z_udBOT)) {// if i'm ALSO in the UD zone
                WGD->icellflag[icell_cent] = 31;

                if (z_rel > z_mid) {// upper half of UD zone

                  u_c = u_c + u_def * (1 - exp(brTOP * (z_udTOP - z_rel)));

                } else {// lower half of UD zone

                  u_c = u_c + u_def * (1 - exp(brBOT * (z_rel - z_udBOT)));
                }


              }// end UD zone if, understory
            }// end understory/no understory if for UD zone

            // APPLY V_C PARAMETERIZATION

            if (z_rel <= H) {
              v_c = vH_c * exp(a_v * (z_rel / H - 1));
            } else {
              // Calculate height-dependent stability correction quantities
              if (rL >= 0.) {
                psi = -5.0 * (z_rel - d_v) * rL;
              } else {
                ex = pow((1. - 15. * (z_rel - d_v) * rL), 0.25);
                psi = log((0.5 * (1. + pow(ex, 2))) * pow(0.5 * (1. + ex), 2)) - 2. * atan(ex) + M_PI * 0.5;
              }
              v_c = ustar_v_c / WGD->vk * (log((z_rel - d_v) / z0_site) - psi);
            }

            tkeFacV = fabs(v_c) / fabs(v_c0);
          }// end wake else


          // Total TKE attenuation (attenuation of the velocity magnitude)
          tkeFac[icell_cent] = pow(pow(tkeFacU * u_c0, 2) + pow(tkeFacV * v_c0, 2), 0.5) / pow(pow(u_c0, 2) + pow(v_c0, 2), 0.5);

          // Rotate back into QES-grid coordinates
          u0_mod_id.push_back(icell_face);
          u0_modified.push_back(cosA * u_c + sinA * v_c);
          v0_mod_id.push_back(icell_face);
          v0_modified.push_back(-sinA * u_c + cosA * v_c);

        }// end k-loop
      }// end canopy_bottom_index < canopy_top_index if


    }// end i-loop
  }// end j-loop

  // apply the parameterization (only once per cell/face!)
  for (auto x_id = 0u; x_id < u0_mod_id.size(); x_id++) {
    WGD->u0[u0_mod_id[x_id]] = u0_modified[x_id];
  }
  for (auto y_id = 0u; y_id < v0_mod_id.size(); y_id++) {
    WGD->v0[v0_mod_id[y_id]] = v0_modified[y_id];
  }

  // clear memory
  u0_mod_id.clear();
  v0_mod_id.clear();
  u0_modified.clear();
  v0_modified.clear();

  return;
}
/*
void CanopyROC::canopyWake(WINDSGeneralData *WGD, int building_id)
{
}
*/
void CanopyROC::canopyWake(WINDSGeneralData *WGD, int building_id)
{

  std::vector<float> u0_modified, v0_modified;
  std::vector<int> u0_mod_id, v0_mod_id;

  // Test to see which sides have wake lines trailing off.

  // Make streamwise vector trailing off each midpoint
  int icell_face = i_building_cent + j_building_cent * WGD->nx + (WGD->nz - 5) * WGD->nx * WGD->ny;
  float u0_uw = WGD->u0[icell_face];// u velocity at the centroid, 5 nodes from the top of domain (avoids effect of nearby wakes)
  float v0_uw = WGD->v0[icell_face];// v velocity at the centroid, 5 nodes from the top of domain
  float M0_uw, dwX, dwY;
  M0_uw = sqrt(pow(u0_uw, 2) + pow(v0_uw, 2));
  float a_uwv[N_e + 1];// the attenuation due to one row (to be raised to N_e, where N_e is number of rows in entry region)
  float tkeFacU_uwv[N_e + 1];

  // Find midpoint of each edge
  std::vector<bool> wakeEdges;
  for (int vtx = 0; vtx < polygonVertices.size() - 1; vtx++) {

    double midX = (polygonVertices[vtx].x_poly + polygonVertices[vtx + 1].x_poly) / 2.0;
    double midY = (polygonVertices[vtx].y_poly + polygonVertices[vtx + 1].y_poly) / 2.0;

    dwX = midX + u0_uw / M0_uw;
    dwY = midY + v0_uw / M0_uw;
    // See if endpoint of vector is inside or outside polygon
    float ray_intersect;
    unsigned int num_crossing, vert_id, start_poly;

    // Find out which cells are going to be inside the polygone
    // Based on Wm. Randolph Franklin, "PNPOLY - Point Inclusion in Polygon Test"
    // Check the center of each cell, if it's inside, set that cell to building
    // Center of cell y coordinate
    // Node index
    vert_id = 0;
    start_poly = vert_id;
    num_crossing = 0;
    while (vert_id < polygonVertices.size() - 1) {
      if ((polygonVertices[vert_id].y_poly <= dwY && polygonVertices[vert_id + 1].y_poly > dwY)
          || (polygonVertices[vert_id].y_poly > dwY && polygonVertices[vert_id + 1].y_poly <= dwY)) {
        ray_intersect = (dwY - polygonVertices[vert_id].y_poly) / (polygonVertices[vert_id + 1].y_poly - polygonVertices[vert_id].y_poly);
        if (dwX < (polygonVertices[vert_id].x_poly + ray_intersect * (polygonVertices[vert_id + 1].x_poly - polygonVertices[vert_id].x_poly))) {
          num_crossing += 1;
        }
      }
      vert_id += 1;
      if (polygonVertices[vert_id].x_poly == polygonVertices[start_poly].x_poly
          && polygonVertices[vert_id].y_poly == polygonVertices[start_poly].y_poly) {
        vert_id += 1;
        start_poly = vert_id;
      }
    }

    // Mark this edge as "wake" or "no wake"
    if ((num_crossing % 2) != 0) {
      wakeEdges.push_back(0);// no wake
      // std::cout << "if 1" << std::endl;
      // wakeEdges[vtx] = 0;
    } else {
      wakeEdges.push_back(1);// wake
      // std::cout << "if 2" << std::endl;
      // wakeEdges[vtx] = 1;
    }
    std::cout << "wakeEdges: " << wakeEdges.at(vtx) << " size: " << wakeEdges.size() << " vtx: " << vtx << " num_crossing: " << num_crossing << " x: " << polygonVertices[vtx].x_poly << " y: " << polygonVertices[vtx].y_poly << " midX: " << midX << " midY: " << midY << std::endl;
  }

  double uRefHat = u0_uw / M0_uw;
  double vRefHat = v0_uw / M0_uw;

  double stepLength = WGD->dxy / 2.0;
  for (int edgeNum = 0; edgeNum < polygonVertices.size() - 1; edgeNum++) {

    if (wakeEdges.at(edgeNum) == 1) {

      double edgeLength = pow(pow(polygonVertices[edgeNum + 1].x_poly - polygonVertices[edgeNum].x_poly, 2.0) + pow(polygonVertices[edgeNum + 1].y_poly - polygonVertices[edgeNum].y_poly, 2.0), 0.5);

      int eSteps = floor(edgeLength / stepLength);// number of steps along edge
      // steps in x and y direction (QES coords) along edge. Sign preserved
      double exStep = (polygonVertices[edgeNum + 1].x_poly - polygonVertices[edgeNum].x_poly) / eSteps;
      double eyStep = (polygonVertices[edgeNum + 1].y_poly - polygonVertices[edgeNum].y_poly) / eSteps;
      double dE = 0.0;// distance along edge

      // step in x and y direction (QES coords) along wake line. Sign preserved from ref vel.
      double wxStep = uRefHat * stepLength;
      double wyStep = vRefHat * stepLength;


      // x and y values of current position along edge
      double xE = polygonVertices[edgeNum].x_poly;
      double yE = polygonVertices[edgeNum].y_poly;

      while (dE <= edgeLength) {


        double dW = 0.0;// distance along wake line
        double xW = xE + 0.0;// x position of current place along wake. Starts from xE (current x-pos on edge)
        double yW = yE + 0.0;// y position "   "   "

        float wakeOrigin[2] = { static_cast<float>(xE), static_cast<float>(yE) };

        float dv_wo = abs(P2L(wakeOrigin, Rx_o, Ry_o));// distance from wake line origin (which is on wake edge) to upwindest vertex
        float N_wo = floor(dv_wo / rowSpacing);// number of rows upwind of wake line origin
        float ld_wo = dv_wo - N_wo * rowSpacing;// local row-orthogonal distance
        float d_dw_local_wo = (ld_wo - rowWidth) / sin(betaAngle * M_PI / 180);// downwind distance behind last low row in the block before the edge (to be added to local downwind distance from (xE,yE) to get total local distance from row


        int i = 0;
        int j = 0;
        while (dW <= 7.5 * H && i < WGD->nx && j < WGD->ny) {

          // WAKE MODEL HERE

          i = ceil(xW / WGD->dx) - 1;
          j = ceil(yW / WGD->dy) - 1;

          int icell_2d = i + j * (WGD->nx - 1);
          float wakeXY[2] = { static_cast<float>(xW), static_cast<float>(yW) };// current x-y location in wake in QES coords
          float dv_w = abs(P2L(wakeXY, Rx_o, Ry_o));// orthogonal distance from current x-y position in wake to upwindest vertex
          float d_dw_local_w = d_dw_local_wo + (dv_w - dv_wo) / sin(betaAngle * M_PI / 180);// total streamwise distance from current point in wake to back side of last row in block

          // find k-node of mid-canopy at current i,j location
          k_mid = 0;
          for (auto k = WGD->terrain_id[icell_2d]; k < WGD->nz; k++) {
            if (WGD->z[k] > (WGD->terrain[icell_2d] + understory_height + (H - understory_height) / 2)) {
              break;
            }
            k_mid = k;
          }

          float a_exp_w = exp((log(0.01) / (7.5 * H)) * d_dw_local_w);

          float ld_w = (dv_w - dv_wo) + ld_wo;// total orthogonal distance between current point in wake and last row in block
          szo_top = std::max(szo_slope * ld_w + H, 0.0f);

          for (int k = WGD->terrain_id[icell_2d]; k < WGD->nz; k++) {
            icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
            WGD->icellflag[icell_cent] = 33;
          }
          z_b = WGD->canopy->canopy_base[icell_2d];

          // MAIN Z-LOOP
          for (auto k = WGD->terrain_id[icell_2d]; k < (WGD->nz - 1); k++) {
            z_rel = WGD->z_face[k - 1] - WGD->terrain[icell_2d];
            int icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
            int icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;

            // Rotate u0 and v0 into row-aligned coords
            u_c0 = cosA * WGD->u0[icell_face] - sinA * WGD->v0[icell_face];
            v_c0 = sinA * WGD->u0[icell_face] + cosA * WGD->v0[icell_face];


            // Initialize u_c and v_c so that in the understory, where no parameterization takes place, u_c and v_c have the correct (i.e. unaltered) value
            u_c = u_c0;
            v_c = v_c0;


            // CALCULATE SHELTERING

            // Sparse understory, below mid-canopy
            if (understory_height > 0 && k <= k_mid) {
              if (N_wo <= N_e) {// entry region
                szt_local = spreadrate_bot * (d_dw * N_wo + d_dw_local_w) + 0.00001;
              } else {// eq region
                szt_local = spreadrate_bot * (d_dw * N_e + d_dw_local_w) + 0.00001;
              }

              a_local = 1 * (a_exp_w * (0.5 * (1 - a_obf) * tanh(1.5 * (szo_bot - z_rel) / szt_local) + 0.5 * (1 + a_obf)) + (1 - a_exp_w));// there should be a "fac" where the 1* is (matching fac isn't working right now)
              tkeFacU_local = 1 * (a_exp_w * (0.5 * (1 - pow(a_obf, 1)) * tanh(1.5 * (szo_bot - z_rel) / szt_local) + 0.5 * (1 + pow(a_obf, 1))) + (1 - a_exp_w));// should be a "fac" where the 1* is
              a_uwv[0] = 1;
              tkeFacU_uwv[0] = 1;
              for (int n = 1; n < N_e + 1; n++) {// create vector for sheltering from all rows in entry region
                szt_uw = spreadrate_bot * (d_dw * n) + 0.00001;
                a_uwv[n] = 1 * (a_exp_w * (0.5 * (1 - a_obf) * tanh(1.5 * (szo_bot - z_rel) / szt_uw) + 0.5 * (1 + a_obf)) + (1 - a_exp_w));// should be a "fac" where the 1* is
                tkeFacU_uwv[n] = 1 * (a_exp_w * (0.5 * (1 - pow(a_obf, 1)) * tanh(1.5 * (szo_bot - z_rel) / szt_uw) + 0.5 * (1 + pow(a_obf, 1))) + (1 - a_exp_w));// should be a "fac" where the 1* is
              }

              // Sparse understory, above mid-canopy
            } else if (understory_height > 0 && k > k_mid) {
              if (N_wo <= N_e) {
                szt_local = spreadrate_top * (d_dw * N_wo + d_dw_local_w) + 0.00001;
              } else {
                szt_local = spreadrate_top * (d_dw * N_e + d_dw_local_w) + 0.00001;
              }


              a_local = (a_exp_w * (0.5 * (1 - a_obf) * tanh(1.5 * (z_rel - szo_top) / szt_local) + 0.5 * (1 + a_obf)) + (1 - a_exp_w));
              tkeFacU_local = (a_exp_w * (0.5 * (1 - pow(a_obf, 1)) * tanh(1.5 * (z_rel - szo_top) / szt_local) + 0.5 * (1 + pow(a_obf, 1))) + (1 - a_exp_w));
              a_uwv[0] = 1;
              tkeFacU_uwv[0] = 1;
              for (int n = 1; n < N_e + 1; n++) {
                szt_uw = spreadrate_top * (d_dw * n) + 0.00001;
                a_uwv[n] = (a_exp_w * (0.5 * (1 - a_obf) * tanh(1.5 * (z_rel - szo_top_uw) / szt_uw) + 0.5 * (1 + a_obf)) + (1 - a_exp_w));
                tkeFacU_uwv[n] = (a_exp_w * (0.5 * (1 - pow(a_obf, 1)) * tanh(1.5 * (z_rel - szo_top_uw) / szt_uw) + 0.5 * (1 + pow(a_obf, 1))) + (1 - a_exp_w));
              }

              // No understory space, all heights
            } else if (understory_height == 0) {
              if (N_wo <= N_e) {
                szt_local = spreadrate_top * (d_dw * N_wo + d_dw_local_w) + 0.00001;
              } else {
                szt_local = spreadrate_top * (d_dw * N_e + d_dw_local_w) + 0.00001;
              }


              a_local = (a_exp_w * (0.5 * (1 - a_obf) * tanh(1.5 * (z_rel - szo_top) / szt_local) + 0.5 * (1 + a_obf)) + (1 - a_exp_w));
              tkeFacU_local = (a_exp_w * (0.5 * (1 - pow(a_obf, 1)) * tanh(1.5 * (z_rel - szo_top) / szt_local) + 0.5 * (1 + pow(a_obf, 1))) + (1 - a_exp_w));
              a_uwv[0] = 1;
              tkeFacU_uwv[0] = 1;
              for (int n = 1; n < N_e + 1; n++) {
                szt_uw = spreadrate_top * (d_dw * n) + 0.00001;
                a_uwv[n] = (a_exp_w * (0.5 * (1 - a_obf) * tanh(1.5 * (z_rel - szo_top_uw) / szt_uw) + 0.5 * (1 + a_obf)) + (1 - a_exp_w));
                tkeFacU_uwv[n] = (a_exp_w * (0.5 * (1 - pow(a_obf, 1)) * tanh(1.5 * (z_rel - szo_top_uw) / szt_uw) + 0.5 * (1 + pow(a_obf, 1))) + (1 - a_exp_w));
              }
            }


            // APPLY WAKE PARAMETERIZATION
            // WGD->icellflag[icell_cent] = 30;
            u_c = u_c0 * a_local;
            tkeFacU = tkeFacU_local;
            for (int n = 0; n <= N_e; n++) {
              u_c *= a_uwv[n];
              tkeFacU *= tkeFacU_uwv[n];
            }

            // APPLY V_C PARAMETERIZATION

            if (z_rel <= H) {
              v_c = vH_c * exp(a_v * (z_rel / H - 1));
            } else {
              // Calculate height-dependent stability correction quantities
              if (rL >= 0.) {
                psi = -5.0 * (z_rel - d_v) * rL;
              } else {
                ex = pow((1. - 15. * (z_rel - d_v) * rL), 0.25);
                psi = log((0.5 * (1. + pow(ex, 2))) * pow(0.5 * (1. + ex), 2)) - 2. * atan(ex) + M_PI * 0.5;
              }
              v_c = ustar_v_c / WGD->vk * (log((z_rel - d_v) / z0_site) - psi);
            }

            tkeFacV = fabs(v_c) / fabs(v_c0);


            // Total TKE attenuation (attenuation of the velocity magnitude)
            tkeFac[icell_cent] = pow(pow(tkeFacU * u_c0, 2) + pow(tkeFacV * v_c0, 2), 0.5) / pow(pow(u_c0, 2) + pow(v_c0, 2), 0.5);

            // Rotate back into QES-grid coordinates
            u0_mod_id.push_back(icell_face);
            u0_modified.push_back(cosA * u_c + sinA * v_c);
            v0_mod_id.push_back(icell_face);
            v0_modified.push_back(-sinA * u_c + cosA * v_c);

          }// end k-loop


          // END WAKE MODEL


          xW += wxStep;
          yW += wyStep;
          dW += stepLength;
        }

        xE += exStep;
        yE += eyStep;
        dE += stepLength;
      }
    }
  }

  // apply the parameterization (only once per cell/face!)
  for (auto x_id = 0u; x_id < u0_mod_id.size(); x_id++) {
    WGD->u0[u0_mod_id[x_id]] = u0_modified[x_id];
  }
  for (auto y_id = 0u; y_id < v0_mod_id.size(); y_id++) {
    WGD->v0[v0_mod_id[y_id]] = v0_modified[y_id];
  }

  // clear memory
  u0_mod_id.clear();
  v0_mod_id.clear();
  u0_modified.clear();
  v0_modified.clear();


  // Strafe along each "wake" edge with step dxy
  // For each point along the wake edge, step in wind direction with step dxy
  // For each point along wake edge, record the number of upwind rows and thus the "nonlocal" part of the distance from the upwindest edge. Retain this as you step down the wake line. Can calculate alpha_upwind at this point.
  // For each point along wake line, snap to grid.
  // At this location, calculate
  // Record cell id of modified velocity
  // Need local distance and total alpha upwind. Then can calculate alpha_local and multiply it by alpha_upwind_total

  return;
}

void CanopyROC::canopyTurbulenceWake(WINDSGeneralData *WGD, TURBGeneralData *TGD, int building_id)
{

  // float tkeMax = 0.5;// To be modeled/assimilated from NWP output later
  // for (auto i = 0; i < WGD->nx - 1; i++) {
  //   for (auto j = 0; j < WGD->ny - 1; j++) {
  for (auto i = 0; i < WGD->nx; i++) {
    for (auto j = 0; j < WGD->ny; j++) {
      // Get mean gradients at canopy top
      int k_top = 0;
      while (WGD->z_face[k_top] < (H + WGD->terrain[i + j * (WGD->nx - 1)])) {
        k_top += 1;
      }
      int icell_faceKTOP = i + j * WGD->nx + k_top * WGD->nx * WGD->ny;

      // Central difference at z=H
      float M_KTOPp1 = pow(pow(WGD->u[icell_faceKTOP + WGD->nx * WGD->ny], 2.0) + pow(WGD->v[icell_faceKTOP + WGD->nx * WGD->ny], 2.0), 0.5);
      float M_KTOPm1 = pow(pow(WGD->u[icell_faceKTOP - WGD->nx * WGD->ny], 2.0) + pow(WGD->v[icell_faceKTOP - WGD->nx * WGD->ny], 2.0), 0.5);


      float dMdz = (M_KTOPp1 - M_KTOPm1) / (WGD->z_face[k_top + 1] - WGD->z_face[k_top - 1]);
      float Ls = M_KTOPm1 / dMdz;// shear length scale at current point

      for (auto k = 0; k < WGD->nz - 2; k++) {
        // int icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
        int icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
        // int icell_face = (i - 1 + i_start) + (j - 1 + j_start) * WGD->nx + k * WGD->nx * WGD->ny;

        if (fabs(tkeFac[icell_cent]) > 0.) {
          /*
        // Katul et al. 2003
        float delta = upwind_dir_unit - 180.0;// assumes all winds are SW quadrants and rows are North-South
        float displacement_height = H * (0.00179 * delta + 0.675);
        float alpha = 0.4 / H * (H - displacement_height);
        float z_match_low = alpha * H / 0.4;
        int k_match_low = 0;
        while (WGD->z_face[k_match_low] < (z_match_low + WGD->terrain[(i - 1 + i_start) + (j - 1 + j_start) * (WGD->nx - 1)])) {
          k_match_low += 1;
        }
        if (k <= k_match_low) {
          TGD->Lm[icell_cent] = 0.4 * (WGD->z_face[k] - WGD->terrain[(i - 1 + i_start) + (j - 1 + j_start) * (WGD->nx - 1)]);
        } else if (k > k_match_low && k <= k_top) {
          TGD->Lm[icell_cent] = alpha * H;
        } else {
          TGD->Lm[icell_cent] = 0.4 * (WGD->z_face[k] - WGD->terrain[(i - 1 + i_start) + (j - 1 + j_start) * (WGD->nx - 1)] - displacement_height);
        }
*/

          // Low: kz
          // Canopy: constant
          // Above: k(z-d)
          // Matched above and below wherever they meet the constant canopy value
          // float canopy_lm = 0.5 * Ls;// 0.5 from PIV scaling
          float Cd = 0.5;// Bailey 2013
          float canopy_lm = 2 * pow(0.4, 3) / (Cd * LAD_eff);// Watanabe 1990, Eq. 9
          float z_match_low = canopy_lm / 0.4;
          float z_match_high = canopy_lm / 0.4 + d_v;
          int k_match_low = 0;
          while (WGD->z_face[k_match_low] < (z_match_low + WGD->terrain[i + j * (WGD->nx - 1)])) {
            k_match_low += 1;
          }
          int k_match_high = 0;
          while (WGD->z_face[k_match_high] < (z_match_high + WGD->terrain[i + j * (WGD->nx - 1)])) {
            k_match_high += 1;
          }

          if (k <= k_match_low) {
            TGD->Lm[icell_cent] = 0.4 * (WGD->z_face[k + 1] - WGD->terrain[i + j * (WGD->nx - 1)]);
          } else if (k > k_match_low && k <= k_match_high) {
            TGD->Lm[icell_cent] = canopy_lm;
          } else {
            TGD->Lm[icell_cent] = 0.4 * (WGD->z_face[k + 1] - WGD->terrain[i + j * (WGD->nx - 1)] - d_v);
          }

          // if (fabs(vineLm[icell_cent]) > 0.) {
          // TGD->Lm[icell_cent] = vineLm[icell_cent];
          // TGD->Lm[icell_cent] = vineLm[icell_cent] * fabs(cos(upwind_dir_unit * M_PI / 180.)) + TGD->Lm[icell_cent] * (1 - abs(cos(upwind_dir_unit * M_PI / 180.)));
          // }

          TGD->tke[icell_cent] = pow(tkeFac[icell_cent], 2) * tkeMax;
          TGD->nuT[icell_cent] = 0.55 * sqrt(TGD->tke[icell_cent]) * TGD->Lm[icell_cent];
        }
      }
    }
  }
  //  tkeFac.clear();
  // vineLm.clear();
}

void CanopyROC::canopyStress(WINDSGeneralData *WGD, TURBGeneralData *TGD, int building_id)
{
  // float tkeMax = 0.5;// To be modeled/assimilated from NWP output later
  // for (auto i = 0; i < WGD->nx - 1; i++) {
  //   for (auto j = 0; j < WGD->ny - 1; j++) {
  for (auto i = 0; i < WGD->nx; i++) {
    for (auto j = 0; j < WGD->ny; j++) {

      // Get mean gradients at canopy top
      int k_top = 0;
      while (WGD->z_face[k_top] < (H + WGD->terrain[i + j * (WGD->nx - 1)])) {
        k_top += 1;
      }
      int k_uh = 0;
      while (WGD->z_face[k_uh] < (understory_height + WGD->terrain[i + j * (WGD->nx - 1)])) {
        k_uh += 1;
      }


      int icell_faceKTOP = i + j * WGD->nx + (k_top + 0) * WGD->nx * WGD->ny;
      int icell_faceKUH = i + j * WGD->nx + k_uh * WGD->nx * WGD->ny;

      // Central difference at z=H
      // float dUdz = (WGD->u[icell_faceKTOP + WGD->nx * WGD->ny] - WGD->u[icell_faceKTOP - WGD->nx * WGD->ny]) / (WGD->z_face[k_top + 1] - WGD->z_face[k_top - 1]);
      // float dVdz = (WGD->v[icell_faceKTOP + WGD->nx * WGD->ny] - WGD->v[icell_faceKTOP - WGD->nx * WGD->ny]) / (WGD->z_face[k_top + 1] - WGD->z_face[k_top - 1]);

      // As in TGD
      float dUdz = ((WGD->u[icell_faceKTOP + WGD->nx * WGD->ny] + WGD->u[icell_faceKTOP + 1 + WGD->nx * WGD->ny])
                    - (WGD->u[icell_faceKTOP - WGD->nx * WGD->ny] + WGD->u[icell_faceKTOP + 1 - WGD->nx * WGD->ny]))
                   / (4.0 * (WGD->z_face[k_top + 1] - WGD->z_face[k_top - 1]));
      float dWdx = ((WGD->w[icell_faceKTOP + 1] + WGD->w[icell_faceKTOP + 1 + WGD->nx * WGD->ny])
                    - (WGD->w[icell_faceKTOP - 1] + WGD->w[icell_faceKTOP - 1 + WGD->nx * WGD->ny]))
                   / (4.0 * WGD->dx);
      float dVdz = ((WGD->v[icell_faceKTOP + WGD->nx * WGD->ny] + WGD->v[icell_faceKTOP + WGD->nx + WGD->nx * WGD->ny])
                    - (WGD->v[icell_faceKTOP - WGD->nx * WGD->ny] + WGD->v[icell_faceKTOP + WGD->nx - WGD->nx * WGD->ny]))
                   / (4.0 * (WGD->z_face[k_top + 1] - WGD->z_face[k_top - 1]));
      float dWdy = ((WGD->w[icell_faceKTOP + WGD->nx] + WGD->w[icell_faceKTOP + WGD->nx + WGD->nx * WGD->ny])
                    - (WGD->w[icell_faceKTOP - WGD->nx] + WGD->w[icell_faceKTOP - WGD->nx + WGD->nx * WGD->ny]))
                   / (4.0 * WGD->dy);

      float dUdy = ((WGD->u[icell_faceKTOP + WGD->nx] + WGD->u[icell_faceKTOP + 1 + WGD->nx])
                    - (WGD->u[icell_faceKTOP - WGD->nx] + WGD->u[icell_faceKTOP + 1 - WGD->nx]))
                   / (4.0 * WGD->dy);

      float dVdx = ((WGD->v[icell_faceKTOP + 1] + WGD->v[icell_faceKTOP + 1 + WGD->nx])
                    - (WGD->v[icell_faceKTOP - 1] + WGD->v[icell_faceKTOP - 1 + WGD->nx]))
                   / (4.0 * WGD->dx);


      float dUdx = ((WGD->u[icell_faceKTOP + 1] + WGD->u[icell_faceKTOP + 2])
                    - (WGD->u[icell_faceKTOP] + WGD->u[icell_faceKTOP - 1]))
                   / (4.0 * WGD->dx);
      float dVdy = ((WGD->v[icell_faceKTOP + WGD->nx] + WGD->v[icell_faceKTOP + 2 * WGD->nx])
                    - (WGD->v[icell_faceKTOP] + WGD->v[icell_faceKTOP - WGD->nx]))
                   / (4.0 * WGD->dy);
      float dWdz = ((WGD->w[icell_faceKTOP + WGD->nx * WGD->ny] + WGD->w[icell_faceKTOP + 2 * WGD->nx * WGD->ny])
                    - (WGD->w[icell_faceKTOP] + WGD->w[icell_faceKTOP - WGD->nx * WGD->ny]))
                   / (4.0 * (WGD->z_face[k_top + 1] - WGD->z_face[k_top - 1]));
      float Sxz_H = 0.5 * (dUdz + dWdx);
      float Syz_H = 0.5 * (dVdz + dWdy);
      float Sxy_H = 0.5 * (dUdy + dVdx);
      float Sxx_H = dUdx;
      float Syy_H = dVdy;
      float Szz_H = dWdz;

      /*
      float dUdz = (WGD->u[icell_faceKTOP] - WGD->u[icell_faceKUH]) / (H - understory_height);
      float dVdz = (WGD->v[icell_faceKTOP] - WGD->v[icell_faceKUH]) / (H - understory_height);
*/
      for (auto k = 0; k < k_top; k++) {
        // int icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
        int icell_cent = i + j * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);
        // int icell_face = i + j * WGD->nx + k * WGD->nx * WGD->ny;

        if (fabs(vineLm[icell_cent]) > 0.) {
          // TGD->Lm[icell_cent] = vineLm[icell_cent];
          // TGD->Lm[icell_cent] = vineLm[icell_cent] * fabs(cos(upwind_dir_unit * M_PI / 180.)) + TGD->Lm[icell_cent] * (1 - abs(cos(upwind_dir_unit * M_PI / 180.)));
        }
        if (fabs(tkeFac[icell_cent]) > 0.) {

          TGD->txz[icell_cent] = -2.0 * (TGD->nuT[icell_cent] * Sxz_H);
          TGD->tyz[icell_cent] = -2.0 * (TGD->nuT[icell_cent] * Syz_H);
          TGD->txy[icell_cent] = -2.0 * (TGD->nuT[icell_cent] * Sxy_H);
          TGD->txx[icell_cent] = (2.0 / 3.0) * TGD->tke[icell_cent] - 2.0 * (TGD->nuT[icell_cent] * Sxx_H);
          TGD->tyy[icell_cent] = (2.0 / 3.0) * TGD->tke[icell_cent] - 2.0 * (TGD->nuT[icell_cent] * Syy_H);
          TGD->tzz[icell_cent] = (2.0 / 3.0) * TGD->tke[icell_cent] - 2.0 * (TGD->nuT[icell_cent] * Szz_H);


          /*
          // ROTATE INTO SENSOR-ALIGNED
          double R11, R12, R13, R21, R22, R23, R31, R32, R33;// rotation matrix
          double I11, I12, I13, I21, I22, I23, I31, I32, I33;// inverse of rotation matrix
          double P11, P12, P13, P21, P22, P23, P31, P32, P33;// P = tau*inv(R)

          //float sensorDir = WID->metParams->sensors[i]->TS[0]->site_wind_dir[0];

          int k_ref = 0;
          float refHeight = 10.0;// reference height for wind direction is arbitrarily set to 15m above terrain
          while (WGD->z_face[k_ref] < (refHeight + WGD->terrain[i + j * (WGD->nx - 1)])) {
            k_ref += 1;
          }

          int localRef = i + j * WGD->nx + k_ref * WGD->nx * WGD->ny;
          float dirRot = atan2(WGD->v[localRef], WGD->u[localRef]);// radians on the unit circle

          //Rotation matrix
          R11 = cos(dirRot);
          R12 = -sin(dirRot);
          R13 = 0.0;
          R21 = sin(dirRot);
          R22 = cos(dirRot);
          R23 = 0.0;
          R31 = 0.0;
          R32 = 0.0;
          R33 = 1.0;

          I11 = R11;
          I12 = R12;
          I13 = R13;
          I21 = R21;
          I22 = R22;
          I23 = R23;
          I31 = R31;
          I32 = R32;
          I33 = R33;

          double txx_temp = TGD->txx[icell_cent];
          double txy_temp = TGD->txy[icell_cent];
          double txz_temp = TGD->txz[icell_cent];
          double tyy_temp = TGD->tyy[icell_cent];
          double tyz_temp = TGD->tyz[icell_cent];
          double tzz_temp = TGD->tzz[icell_cent];

          //Invert rotation matrix
          TGD->invert3(I11, I12, I13, I21, I22, I23, I31, I32, I33);

          TGD->matMult(txx_temp, txy_temp, txz_temp, txy_temp, tyy_temp, tyz_temp, txz_temp, tyz_temp, tzz_temp, I11, I12, I13, I21, I22, I23, I31, I32, I33, P11, P12, P13, P21, P22, P23, P31, P32, P33);

          TGD->matMult(R11, R12, R13, R21, R22, R23, R31, R32, R33, P11, P12, P13, P21, P22, P23, P31, P32, P33, txx_temp, txy_temp, txz_temp, txy_temp, tyy_temp, tyz_temp, txz_temp, tyz_temp, tzz_temp);

          txx_temp = fabs(TGD->sigUConst * txx_temp);
          tyy_temp = fabs(TGD->sigVConst * tyy_temp);
          tzz_temp = fabs(TGD->sigWConst * tzz_temp);


          // DEROTATE
          //Rotation matrix
          dirRot = -dirRot;
          R11 = cos(dirRot);
          R12 = -sin(dirRot);
          R13 = 0.0;
          R21 = sin(dirRot);
          R22 = cos(dirRot);
          R23 = 0;
          R31 = 0;
          R32 = 0;
          R33 = 1;

          I11 = R11;
          I12 = R12;
          I13 = R13;
          I21 = R21;
          I22 = R22;
          I23 = R23;
          I31 = R31;
          I32 = R32;
          I33 = R33;

          //Invert rotation matrix
          TGD->invert3(I11, I12, I13, I21, I22, I23, I31, I32, I33);

          TGD->matMult(txx_temp, txy_temp, txz_temp, txy_temp, tyy_temp, tyz_temp, txz_temp, tyz_temp, tzz_temp, I11, I12, I13, I21, I22, I23, I31, I32, I33, P11, P12, P13, P21, P22, P23, P31, P32, P33);

          TGD->matMult(R11, R12, R13, R21, R22, R23, R31, R32, R33, P11, P12, P13, P21, P22, P23, P31, P32, P33, txx_temp, txy_temp, txz_temp, txy_temp, tyy_temp, tyz_temp, txz_temp, tyz_temp, tzz_temp);

          TGD->txx[icell_cent] = txx_temp;
          TGD->txy[icell_cent] = txy_temp;
          TGD->txz[icell_cent] = txz_temp;
          TGD->tyy[icell_cent] = tyy_temp;
          TGD->tyz[icell_cent] = tyz_temp;
          TGD->tzz[icell_cent] = tzz_temp;
        */
        }
      }
    }
  }
  tkeFac.clear();
  vineLm.clear();
}
