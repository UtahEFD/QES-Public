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

  float orth_test[2];
  float o_signed[2];

  orth_test[0] = P[0] + row_ortho[0];
  orth_test[1] = P[1] + row_ortho[1];
  if (P2L(orth_test, Lx, Ly) < dist) {
    o_signed[0] = row_ortho[0];
    o_signed[1] = row_ortho[1];
    dist = P2L(orth_test, Lx, Ly);
  }
  orth_test[0] = P[0] - row_ortho[0];
  orth_test[1] = P[1] + row_ortho[1];
  if (P2L(orth_test, Lx, Ly) < dist) {
    o_signed[0] = -row_ortho[0];
    o_signed[1] = row_ortho[1];
    dist = P2L(orth_test, Lx, Ly);
  }

  orth_test[0] = P[0] + row_ortho[0];
  orth_test[1] = P[1] - row_ortho[1];
  if (P2L(orth_test, Lx, Ly) < dist) {
    o_signed[0] = row_ortho[0];
    o_signed[1] = -row_ortho[1];
    dist = P2L(orth_test, Lx, Ly);
  }

  orth_test[0] = P[0] - row_ortho[0];
  orth_test[1] = P[1] - row_ortho[1];
  if (P2L(orth_test, Lx, Ly) < dist) {
    o_signed[0] = -row_ortho[0];
    o_signed[1] = -row_ortho[1];
    dist = P2L(orth_test, Lx, Ly);
  }

  row_ortho[0] = o_signed[0];
  row_ortho[1] = o_signed[1];
}
/*
float CanopyROC::UV2compass(float u, float v){

  if (u>0 && v>0){
    
  }


  float unitDir = -(compDir- 90);

  return compDir;
}
*/


void CanopyROC::setCellFlags(const WINDSInputData *WID, WINDSGeneralData *WGD, int building_id)
{
  // When THIS canopy calls this function, we need to do the
  // following:
  //readCanopy(nx, ny, nz, landuse_flag, num_canopies, lu_canopy_flag,
  //canopy_atten, canopy_top);

  // this function need to be called to defined the boundary of the canopy and the icellflags
  //setCanopyGrid(WGD, building_id);
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
          //WGD->icellflag_footprint[icell_2d] = getCellFlagCanopy();

          // Define start index of the canopy in z-direction
          for (size_t k = 1u; k < WGD->z.size(); k++) {
            if (WGD->terrain[icell_2d] + base_height <= WGD->z[k]) {
              WGD->canopy->canopy_bot_index[icell_2d] = k;
              WGD->canopy->canopy_bot[icell_2d] = WGD->terrain[icell_2d] + base_height;
              WGD->canopy->canopy_base[icell_2d] = WGD->z_face[k - 1];
              //std::cout << "canopy_bot[" << icell_2d <<"] = " << WGD->canopy->canopy_bot[icell_2d] << "canopy_bot_index[" << icell_2d <<"] = " << WGD->canopy->canopy_bot_index[icell_2d] << std::endl;
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
              //WGD->icellflag[icell_3d] = getCellFlagCanopy();
              //WGD->canopy->canopy_atten_coeff[icell_3d] = attenuationCoeff;
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
  int nearest_sensor_i;// index of the sensor nearest to ROC block centroid
  for (auto i = 0; i < num_sites; i++) {
    curr_dist = sqrt(pow((building_cent_x - WID->metParams->sensors[i]->site_xcoord), 2) + pow((building_cent_y - WID->metParams->sensors[i]->site_ycoord), 2));
    if (curr_dist < sensor_distance) {
      sensor_distance = curr_dist;
      nearest_sensor_i = i;
    }
  }

  rL = WID->metParams->sensors[nearest_sensor_i]->TS[0]->site_one_overL;// uses first time step value only - figure out how to get current time step inside this function

  z0_site = WID->metParams->sensors[nearest_sensor_i]->TS[0]->site_z0;
  //std::cout << "z0_site = " << z0_site << "\n";
  //std::cout << "rL = " << rL << "\n";
  //std::cout << "nearest_sensor_i = " << nearest_sensor_i << "\n";

  a_obf = pow(beta, 0.4);

  return;
}


void CanopyROC::canopyVegetation(WINDSGeneralData *WGD, int building_id)
{

  std::vector<float> u0_modified, v0_modified;
  std::vector<int> u0_mod_id, v0_mod_id;
  //std::cout << "checkpoint 1 \n";

  //std::cout << "VINEYARD PARAMETERIZATION STARTING \n";

  int icell_face = i_building_cent + j_building_cent * WGD->nx + (WGD->nz - 5) * WGD->nx * WGD->ny;
  float u0_uw = WGD->u0[icell_face];// u velocity at the centroid, 5 nodes from the top of domain (avoids effect of nearby wakes)
  float v0_uw = WGD->v0[icell_face];// v velocity at the centroid, 5 nodes from the top of domain
  //std::cout << "is u0_uw a nan? " << isnan(u0_uw) << "  is v0_uw a nan? " << isnan(v0_uw) << "\n";

  upwind_dir_unit = atan2(v0_uw, u0_uw) * 180. / M_PI + 180.;// degrees on the unit circle

  //std::cout << "v0_uw = " << v0_uw << "u0_uw = " << u0_uw << "upwind_dir_unit = " << upwind_dir_unit << std::endl;

  // Create unit wind vector
  float M0_uw, u0n_uw, v0n_uw;
  M0_uw = sqrt(pow(u0_uw, 2) + pow(v0_uw, 2));
  u0n_uw = u0_uw / M0_uw;
  v0n_uw = v0_uw / M0_uw;

  // Convert rowAngle (compass rose) to unit circle degrees
  float rowAngle_u = -(rowAngle - 90);
  float cosA = cos(rowAngle * M_PI / 180);
  float sinA = sin(rowAngle * M_PI / 180);

  float rd[2] = { cos(rowAngle_u * M_PI / 180), sin(rowAngle_u * M_PI / 180) };

  float Rx[2] = { building_cent_x, building_cent_x + rd[0] };
  float Ry[2] = { building_cent_y, building_cent_y + rd[1] };

  float uwdV = -1000000;// a low number
  float dwdV = 1000000;// a high number
  float curr_V[2];
  float d2V;
  float wwdV_current;// windward distance to current vertex
  int idS;
  int idE;
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

  // Keep this (LDU 220323)
  //std::cout << "idS = " << idS << " idE = " << idE << "\n";
  //std::cout << "x,y of upwindest vertex = " << polygonVertices[idS].x_poly << " , " << polygonVertices[idS].y_poly << "\n";
  //std::cout << "x,y of downwindest vertex = " << polygonVertices[idE].x_poly << " , " << polygonVertices[idE].y_poly << "\n";

  // Find streamwise distance between rows (d_dw):
  float betaAngle = acos(rd[0] * u0n_uw + rd[1] * v0n_uw) * 180 / M_PI;// acute angle between wd and row vector, in degrees

  if (betaAngle > 90) {
    betaAngle = 180 - betaAngle;
  }
  if (abs(betaAngle) < 0.5) {
    betaAngle = 0.5;// because we have sin(betaAngle) in a couple denominators
  }
  float d_dw = (rowSpacing - rowWidth) / sin(betaAngle * M_PI / 180);

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
  float Cd_ud = 1.98;//flat plate in perpendicular flow
  //float Cd = 2 * pow(uustar / M0_h, 2);// Heilman 1996 eq. 5
  //float Cd = 0.2;// Chahine 2014
  //float LAI_row = 5.57;
  //float LAI_eff = LAI_row * (rowWidth / rowSpacing);
  //float LAD_eff = LAI_eff / H;

  //float LAD_eff = 0.6667;
  //float LAD_avg = 4.3138;
  //float fenceThickness = H * (8. / 72.);

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
  int N_e = (int)N_e_float;
  N_e += 2;// tuned to Torkelson PIV

  std::cout << "l_e = " << l_e << ", N_e = " << N_e << ", L_c = " << L_c << ", M0_h = " << M0_h << ", uustar = " << uustar << "\n";
  int terrQi = 241;
  int terrQj = 550;
  int icell_2d_source = (terrQi) + (terrQj) * (WGD->nx - 1);
  std::cout << "TERRAIN HEIGHT (ROAD) AT i = " << terrQi << " j = " << terrQj << " is: " << WGD->terrain[icell_2d_source] << std::endl;
  terrQi = 501;
  terrQj = 550;
  icell_2d_source = (terrQi) + (terrQj) * (WGD->nx - 1);
  std::cout << "TERRAIN HEIGHT (VINE) AT i = " << terrQi << " j = " << terrQj << " is: " << WGD->terrain[icell_2d_source] << std::endl;

  // Spread rate calculations
  float udelt = (1 - pow(a_obf, N_e + 1));
  float uave = (1 + pow(a_obf, N_e + 1)) / 2;

  float spreadclassicmix = 0.14 * udelt / uave;

  float spreadupstream_top = 2 * stdw / M0_h;
  float spreadupstream_bot = 2 * stdw / M0_uh;

  //std::cout << "a_obf = " << a_obf << " spreadclassicmix = " << spreadclassicmix << " spreadupstream_top = " << spreadupstream_top << " \n";

  // Avoid divide-by-zero for the no-understory case (where M0_uh will be 0)
  if (abs(M0_uh) < 0.000001) {
    spreadupstream_bot = spreadupstream_top;
  }
  //std::cout << " M0_h = " << M0_h << " spreadupstream_top = " << spreadupstream_top << "\n";
  //std::cout << " M0_uh = " << M0_uh << " spreadupstream_bot = " << spreadupstream_bot << "\n";

  float spreadrate_top = sqrt(pow(spreadclassicmix, 2) + pow(spreadupstream_top, 2));
  float spreadrate_bot = sqrt(pow(spreadclassicmix, 2) + pow(spreadupstream_bot, 2));
  std::cout << "spreadrate_top = " << spreadrate_top << "spreadrate_bot = " << spreadrate_bot << "\n";

  // Shear zone origin (initialized to H but parameterized later)
  float szo_top = H;
  float szo_bot = understory_height;

  // Upwind displacement zone parameters (for no understory)
  float l_ud = 0.5 * H;// horizontal length of UD zone (distance from the vine that it extends to in the windward direction)
  float z_ud;// upper bound of UD zone at current i,j location, if no understory
  float H_ud = 0.6 * H;// max height of UD (occurs where it attaches at the windward side of a vine)
  float br;// blend rate, used in the blending function that smoothes the UD zone into the wake
  float x_ud;// horizontal distance in the UD zone. Measured orthogonally from the "upwindest" side of the vine
  float u_def = 0.0;// velocity deficit at a point in the UD zone
  // UD zone variables (if understory)
  float z_udTOP;// upper bound of UD zone, if there's an understory
  float z_udBOT;// lower bound of UD zone, if there's an understory
  float brTOP;//blend rate for top half of UD zone, if there's an understory
  float brBOT;//blend rate for bottom half of UD zone, if there's an understory
  float a_ud;// a constant

  // V_c parameterization parameters
  float ustar_v_c, psi, psiH, dpsiH, ex, exH, dexH, vH_c, a_v, vref_c, zref, exRef, dexRef, psiRef, dpsiRef;
  int icell_face_ref, zref_k;

  // General geometry parameters
  float dv_c;// distance of current node to upwind-est vertex
  float dv_c_dw;// distance of current node to downwind-est vertex
  float N;// number of rows upwind of current point
  float ld;// "local distance", the orthogonal distance from the current i,j point to the upwind edge of the closest row in the upwind direction
  float z_rel;// height off the terrain
  float z_b;// base of the canopy
  int icell_2d, icell_2d_canopy;
  int k_mid;// mid-canopy k-node
  float a_exp;// a blending constant that smooths the transition from wake to upwind flow
  float szt_uw, szt_local, szt_Lm;// shear zone thickness variables


  float a_uwv[N_e + 1];// the attenuation due to one row (to be raised to N_e, where N_e is number of rows in entry region)
  int nx = WGD->nx;
  int ny = WGD->ny;
  int nz = WGD->nz;
  int np_cc_v = (nz - 1) * (ny - 1) * (nx - 1);

  vineLm.resize(np_cc_v, 0);
  float a_uw = 1;

  float a_local;// the attenuation due to the closest upwind row only
  float u_c0, v_c0;// u0 and v0 rotated into row-aligned coordinates
  float u_c, v_c;// altered (parameterized) u and v, in row-aligned coordinates

  // DEBUG VARIABLES
  bool BF_flag = 1;
  int UD_zone_flag = 1;

  // BEGIN MAIN PARAMETERIZATION LOOPS
  for (auto j = 0; j < ny_canopy; j++) {
    for (auto i = 0; i < nx_canopy; i++) {
      //icell_2d = i + j * nx_canopy;
      icell_2d = (i + i_start) + (j + j_start) * (WGD->nx - 1);

      // base of the canopy
      z_b = WGD->canopy->canopy_base[icell_2d];

      //if (i+i_start == 434 && j+j_start == 385){
      //  std::cout << "TERRAIN HEIGHT AT posX posY = " << WGD->terrain[icell_2d] << std::endl;
      //}
      // calculate row-orthogonal distance to upwind-est row, dv_c
      float cxy[2] = { (i - 1 + i_start) * WGD->dx, (j - 1 + j_start) * WGD->dy };// current x position

      // Row-orthogonal distance between current point and the upwind-est row
      float Rx_o[2] = { polygonVertices[idS].x_poly, polygonVertices[idS].x_poly + rd[0] };// x-components of upwind-est row vector
      float Ry_o[2] = { polygonVertices[idS].y_poly, polygonVertices[idS].y_poly + rd[1] };// y-components "   "   "
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
      float d_dw_local = (ld - rowWidth) / sin(betaAngle * M_PI / 180);
      // find blending factor "a_exp" at the current i,j location (a function of d_dw_local)
      a_exp = exp((log(0.01) / (7.5 * H)) * d_dw_local);

      // find k-node of mid-canopy at current i,j location
      for (auto k = WGD->canopy->canopy_bot_index[icell_2d]; k < WGD->canopy->canopy_top_index[icell_2d]; k++) {
        if (WGD->z[k] > (z_b + understory_height + (H - understory_height) / 2)) {
          break;
        }
        k_mid = k;
      }

      // Find shear zone origin based on Greg Torkelson's slopes from PIV (S_omega1, Table 2)
      // Data are fit to a modified Michaelis-Menten curve: fit_a*x / (fit_b * x) + fit_c, where x is rowSpacing/H
      float fit_a, fit_b, fit_c;

      //fit_a = 0.7583;//with first data point set at 1
      //fit_b = 1.3333;
      //fit_c = -0.7550;
      fit_a = 1.3945;//with first data point set at 2.5/2.16=1.1574
      fit_b = 0.4209;
      fit_c = -1.4524;

      float szo_slope = fit_a * (rowSpacing / H) / ((rowSpacing / H) + fit_b) + fit_c;
      //float szo_slope = -0.43;
      //std::cout << "szo_slope = " << szo_slope << std::endl;
      szo_top = std::max(szo_slope * ld + H, 0.0f);
      float szo_top_uw = std::max(szo_slope * (rowSpacing - rowWidth) + H, 0.0f);// origin right before the next row, e.g. at a distance of "rowSpacing" orthogonally from nearest upwind row

      float z_mid = (H - understory_height) / 2 + understory_height;


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

        /*
        if (i + i_start == i_building_cent && j + j_start == j_building_cent) {
          
          std::cout << "i = " << i_building_cent << "j = " << j_building_cent << "\n";
          std::cout << "canopy_bot_index[" << icell_2d << "] = " << WGD->canopy->canopy_bot_index[icell_2d] << std::endl;
          std::cout << "z_b = " << z_b << std::endl;
          std::cout << "vref_c = " << vref_c << "\n";
          std::cout << "ustar_v_c = " << ustar_v_c << "\n";
          std::cout << "vH_c = " << vH_c << "\n";
          std::cout << "a_v = " << a_v << "\n";
          std::cout << "z0_site = " << z0_site << "\n";
          std::cout << "rL = " << rL << "\n";
          std::cout << "zref = " << zref << "\n";
          std::cout << "zref_k = " << zref_k << "\n";
          std::cout << "d_v = " << d_v << "\n";
        }
*/

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
            a_uwv[0] = 1;
            for (int n = 1; n < N_e + 1; n++) {//create vector for sheltering from all rows in entry region
              szt_uw = spreadrate_bot * (d_dw * n) + 0.00001;
              a_uwv[n] = 1 * (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (szo_bot - z_rel) / szt_uw) + 0.5 * (1 + a_obf)) + (1 - a_exp));//should be a "fac" where the 1* is
            }

            // Sparse understory, above mid-canopy
          } else if (understory_height > 0 && k > k_mid) {
            if (N <= N_e) {
              szt_local = spreadrate_top * (d_dw * N + d_dw_local) + 0.00001;
            } else {
              szt_local = spreadrate_top * (d_dw * N_e + d_dw_local) + 0.00001;
            }
            a_local = (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (z_rel - szo_top) / szt_local) + 0.5 * (1 + a_obf)) + (1 - a_exp));
            a_uwv[0] = 1;
            for (int n = 1; n < N_e + 1; n++) {
              szt_uw = spreadrate_top * (d_dw * n) + 0.00001;
              a_uwv[n] = (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (z_rel - szo_top_uw) / szt_uw) + 0.5 * (1 + a_obf)) + (1 - a_exp));
            }

            // No understory space, all heights
          } else if (understory_height == 0) {
            if (N <= N_e) {
              szt_local = spreadrate_top * (d_dw * N + d_dw_local) + 0.00001;
            } else {
              szt_local = spreadrate_top * (d_dw * N_e + d_dw_local) + 0.00001;
            }
            a_local = (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (z_rel - szo_top) / szt_local) + 0.5 * (1 + a_obf)) + (1 - a_exp));
            a_uwv[0] = 1;
            for (int n = 1; n < N_e + 1; n++) {
              szt_uw = spreadrate_top * (d_dw * n) + 0.00001;
              a_uwv[n] = (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (z_rel - szo_top_uw) / szt_uw) + 0.5 * (1 + a_obf)) + (1 - a_exp));
            }
          }


          // APPLY BLEED FLOW PARAMETERIZATION INSIDE THE ACTUAL VEGETATION
          if (BF_flag == 1 && dv_c > rowSpacing && ld < rowWidth) {// if my x-y position indicates i'm on a vine
            if ((z_rel >= understory_height) && (z_rel <= H)) {// if my z position indicates i'm inside the actual vegetation

              // Am I in entry region or /quilibrated region?
              if (N <= N_e) {// i'm in entry region

                WGD->icellflag[icell_cent] = 28;

                u_c = u_c0 * a_obf;
                for (int n = 0; n <= N; n++) {
                  u_c *= a_uwv[n];
                }

              }

              else {// if i'm in equilibrated region

                WGD->icellflag[icell_cent] = 28;
                u_c = u_c0 * a_obf;
                for (int n = 0; n <= N_e; n++) {
                  u_c *= a_uwv[n];
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
              for (int n = 0; n <= N; n++) {
                u_c *= a_uwv[n];
              }

              szt_Lm = spreadrate_top * (d_dw_local) + 0.00001;
              if (z_rel > szo_top - 0.5 * szt_Lm && z_rel < szo_top + 0.5 * szt_Lm) {
                vineLm[icell_cent] = szt_Lm;
                //vineLm[icell_cent] = spreadrate_top * (d_dw * N + d_dw_local) + 0.00001;
              }

            }

            else if (N > N_e && dv_c > rowSpacing) {// if i'm in equilibrated region

              u_c = u_c0 * a_local;
              for (int n = 0; n <= N_e; n++) {
                u_c *= a_uwv[n];
              }

              szt_Lm = spreadrate_top * (d_dw_local) + 0.00001;
              if (z_rel > szo_top - 0.5 * szt_Lm && z_rel < szo_top + 0.5 * szt_Lm) {
                vineLm[icell_cent] = szt_Lm;
                //vineLm[icell_cent] = spreadrate_top * (d_dw * N_e + d_dw_local) + 0.00001;
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
                if (i + i_start == 65 && j + j_start == 75 && k < 20) {
                  std::cout << "k = " << k << " u_c = " << u_c << " u_c after = " << u_c + u_def * (1 - exp(br * (z_ud - z_rel))) << " deficit: " << u_def << " beta = " << beta << " a_obf = " << a_obf << " rowWidth = " << rowWidth << " l_ud = " << l_ud << " Cd = " << Cd << std::endl;
                }

                u_c = u_c + u_def * (1 - exp(br * (z_ud - z_rel)));
              }
            } else {// else if there's an understory

              if (UD_zone_flag && dv_c_dw > rowSpacing && ((rowSpacing - ld) < l_ud) && (z_rel <= z_udTOP) && (z_rel >= z_udBOT)) {// if i'm ALSO in the UD zone
                WGD->icellflag[icell_cent] = 31;

                if (z_rel > z_mid) {// upper half of UD zone
                  if (i + i_start == 65 && j + j_start == 75 && k < 20) {
                    std::cout << "k = " << k << " u_c = " << u_c << " u_c after = " << u_c + u_def * (1 - exp(brTOP * (z_udTOP - z_rel))) << " deficit: " << u_def << " beta = " << beta << " a_obf = " << a_obf << " rowWidth = " << rowWidth << " l_ud = " << l_ud << " Cd = " << Cd << std::endl;
                  }

                  u_c = u_c + u_def * (1 - exp(brTOP * (z_udTOP - z_rel)));

                } else {// lower half of UD zone
                  if (i + i_start == 65 && j + j_start == 75 && k < 20) {
                    std::cout << "k = " << k << " u_c = " << u_c << " u_c after = " << u_c + u_def * (1 - exp(brBOT * (z_rel - z_udBOT))) << " deficit: " << u_def << " beta = " << beta << " a_obf = " << a_obf << " rowWidth = " << rowWidth << " l_ud = " << l_ud << " Cd = " << Cd << std::endl;
                  }

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

          }// end wake else

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

void CanopyROC::canopyWake(WINDSGeneralData *WGD, int building_id)
{


  // Need local distance and total alpha upwind. Then can calculate alpha_local and multiply it by alpha_upwind_total

  return;
}

