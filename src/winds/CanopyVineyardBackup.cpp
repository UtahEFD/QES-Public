#include "CanopyVineyard.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

float rL, z0_site;// parameters used by multiple functions in this file

void nan_count(float nan_var, int &nan_counter, std::string &location)
{

  if (isnan(nan_var)) {
    if (nan_counter == 0) {
      std::cout << "NaN found in " << location << "\n";
      nan_counter += 1;
    } else {
      nan_counter += 1;
    }
  }
  return;
}
float P2L(float P[2], float Lx[2], float Ly[2])
{
  float d = abs((Lx[1] - Lx[0]) * (Ly[0] - P[1]) - (Lx[0] - P[0]) * (Ly[1] - Ly[0])) / sqrt(pow(Lx[1] - Lx[0], 2) + pow(Ly[1] - Ly[0], 2));
  return d;
}

void orthog_vec(float d, float P[2], float Lx[2], float Ly[2], float row_ortho[2])
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


// set et attenuation coefficient
void CanopyVineyard::setCellFlags(const WINDSInputData *WID, WINDSGeneralData *WGD, int building_id)
{
  // When THIS canopy calls this function, we need to do the
  // following:
  //readCanopy(nx, ny, nz, landuse_flag, num_canopies, lu_canopy_flag,
  //canopy_atten, canopy_top);

  // this function need to be called to defined the boundary of the canopy and the icellflags
  setCanopyGrid(WGD, building_id);

  if (ceil(1.5 * k_end) > WGD->nz - 1) {
    std::cerr << "ERROR domain too short for tree method" << std::endl;
    exit(EXIT_FAILURE);
  }


  // Find nearest sensor (so we use the correct 1/L in the vo parameterization)
  float sensor_distance = 99999;
  float curr_dist;
  int num_sites = WID->metParams->sensors.size();
  int nearest_sensor_i;// index of the sensor nearest to vineyard block centroid
  for (auto i = 0; i < num_sites; i++) {
    curr_dist = sqrt(pow((building_cent_x - WID->metParams->sensors[i]->site_xcoord), 2) + pow((building_cent_y - WID->metParams->sensors[i]->site_ycoord), 2));
    if (curr_dist < sensor_distance) {
      sensor_distance = curr_dist;
      nearest_sensor_i = i;
    }
  }

  rL = WID->metParams->sensors[nearest_sensor_i]->TS[0]->site_one_overL;// uses first time step value only - figure out how to get current time step inside this function

  z0_site = WID->metParams->sensors[nearest_sensor_i]->TS[0]->site_z0;
  std::cout << "z0_site = " << z0_site << "\n";
  std::cout << "rL = " << rL << "\n";
  std::cout << "nearest_sensor_i = " << nearest_sensor_i << "\n";
  // Resize the canopy-related vectors
  /*
      canopy_atten.resize( numcell_cent_3d, 0.0 );

    for (auto j=0; j<ny_canopy; j++) {
        for (auto i=0; i<nx_canopy; i++) {
            int icell_2d = i + j*nx_canopy;
            for (auto k=canopy_bot_index[icell_2d]; k<=canopy_top_index[icell_2d]; k++) {
                int icell_3d = i + j*nx_canopy + k*nx_canopy*ny_canopy;
                // initiate all attenuation coefficients to the canopy coefficient
                canopy_atten[icell_3d] = attenuationCoeff;     
            }
        }
    }
    */

  a_obf = pow(beta, 0.4);

  return;
}


void CanopyVineyard::canopyVegetation(WINDSGeneralData *WGD, int building_id)
{

  std::vector<float> u0_modified, v0_modified;
  std::vector<int> u0_mod_id, v0_mod_id;
  //std::cout << "checkpoint 1 \n";
  // Which aerodynamic porosity model to use
  if (thinFence = 1) {
    a_obf = beta;
  } else {
    a_obf = pow(beta, 0.4);
  }


  int icell_face = i_building_cent + j_building_cent * WGD->nx + (WGD->nz - 5) * WGD->nx * WGD->ny;
  float u0_uw = WGD->u0[icell_face];// u velocity at the centroid, 5 nodes from the top of domain (avoids effect of nearby wakes)
  float v0_uw = WGD->v0[icell_face];// v velocity at the centroid, 5 nodes from the top of domain
  std::cout << "is u0_uw a nan? " << isnan(u0_uw) << "  is v0_uw a nan? " << isnan(v0_uw) << "\n";

  upwind_dir = atan2(v0_uw, u0_uw);


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

  float wwdV = -1000000;// a low number
  float curr_V[2];
  float d2V;
  float wwdV_current;
  int idS;
  for (int id = 0; id < polygonVertices.size() - 1; id++) {
    curr_V[0] = polygonVertices[id].x_poly;
    curr_V[1] = polygonVertices[id].y_poly;
    d2V = P2L(curr_V, Rx, Ry);

    // Find orthogonal vector (of length d2V, starting at the current vertex curr_V and ending on line Rx,Ry
    float orthog[2];
    orthog_vec(d2V, curr_V, Rx, Ry, orthog);

    // Find windward distance of current vertex
    wwdV_current = u0n_uw * orthog[0] + v0n_uw * orthog[1];

    if (wwdV_current > wwdV) {
      wwdV = wwdV_current;
      idS = id;// index of vertex where rows "start" from
    }
  }
  std::cout << "idS = " << idS << "\n";
  std::cout << "x_poly = " << polygonVertices[idS].x_poly << "\n";
  std::cout << "y_poly = " << polygonVertices[idS].y_poly << "\n";

  // Find streamwise distance between rows (d_sw):
  // 1. Find acute angle between wd and row vector
  float beta = acos(rd[0] * u0n_uw + rd[1] * v0n_uw) * 180 / M_PI;// beta is in degrees

  if (beta > 90) {
    beta = 180 - beta;
  }
  if (abs(beta) < 0.5) {
    beta = 0.5;// because we have sin(beta) in a couple denominators
  }
  // 2. Find d_sw (streamwise distance between end of one row and beginning of next)
  float d_dw = (rowSpacing - rowWidth) / sin(beta * M_PI / 180);

  // 3. Find shear zone width at d_dw
  //    To get spread rate:
  //    - Need U_h from upwind of the vineyard block (should actually be component of velocity orthogonal to row, not U)

  // Find appropriate reference node for finding U_h
  // Use the centroid of the vineyard block. This code is run before the vineyard param, so it should be the unaltered uo sensor profile (or building/tree wake, if present)

  // Ref velocity for top shear zone
  int k_top = 0;
  while (WGD->z_face[k_top] < (H + WGD->terrain[i_building_cent + j_building_cent * (WGD->nx - 1)])) {
    k_top += 1;
  }
  icell_face = i_building_cent + j_building_cent * WGD->nx + k_top * WGD->nx * WGD->ny;
  float u0_h = WGD->u0[icell_face];
  float v0_h = WGD->v0[icell_face];
  float rd_o[2] = { rd[1], -rd[0] };// row-orthogonal unit vector
  float M0_h = abs(u0_h * rd_o[0] + v0_h * rd_o[1]);// the component of the wind vector in the row-orthogonal direction

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
  float L_c = rowWidth * (1 / (1 - a_obf));// from eq. 3.1, Belcher 2003, approximating du/dx using a finite difference about the upwind-est vine, then dividing through by u
  float l_e = L_c * log((M0_h / uustar) * (H / L_c));
  float N_e = ceil(l_e / rowSpacing);
  //N_e = 10;

  std::cout << "l_e = " << l_e << ", N_e = " << N_e << "\n";
  // Spread rate calculations
  float udelt = (1 - a_obf);
  float uave = (1 + a_obf) / 2;

  float spreadclassicmix = 0.14 * udelt / uave;
  float spreadupstream_top = 2 * stdw / M0_h;
  float spreadupstream_bot = 2 * stdw / M0_uh;
  float spreadrate_top = sqrt(pow(spreadclassicmix, 2) + pow(spreadupstream_top, 2));
  float spreadrate_bot = sqrt(pow(spreadclassicmix, 2) + pow(spreadupstream_bot, 2));

  // Shear zone origin (for now, keep it at H, but later parameterize its descent)
  float szo_top = H;
  float szo_bot = understory_height;

  // Upwind displacement zone parameters
  float l_ud = 0.5 * H;// horizontal length of UD zone (distance from the vine that it extends to in the windward direction)
  float z_ud;// height of UD zone at current i,j location
  float H_ud = 0.6 * H;// max height of UD (occurs where it attaches at the windward side of a vine)
  float br;// blend rate, used in the blending function that smoothes the UD zone into the wake
  float x_ud;// horizontal distance in the UD zone. Measured orthogonally from the "upwindest" side of the vine

  // V_c parameterization parameters
  float ustar_v_c, psi, psiH, dpsiH, ex, exH, dexH, vH_c, a_v, vref_c, zref;
  int icell_face_ref, zref_k;

  // WGD->terrain[i+j*(WGD->nx-1)] appears to be height (in meters) of the terrain (may be cell-centered because it's always called with nx-1 rather than nx)
  // WGD->terrain_id[i+j*(WGD->nx)] appears to be height (in nodes) of the terrain (may be face-centered because it's always called with nx)
  // max_terrain_id is calculated in Sensor.cpp. Index (using nx-1) of highest terrain node

  float dv_c;// distance of current node to upwind-est vertex
  float N;// number of rows upwind of current point
  float ld;// "local distance", the orthogonal distance from the current i,j point to the upwind edge of the closest row in the upwind direction
  float z_rel;// height off the terrain
  float z_b;// base of the canopy
  int icell_2d;
  float fac;// multiplicative factor used to match lower and upper tanh profiles at mid-canopy height (multiplies the lower profile)
  float szt_top, szt_bot;// shear zone thickness, used in calculating fac
  int k_mid;// mid-canopy k-node
  float a_exp;// a blending constant that smooths the transition from wake to upwind flow
  float szt_uw, szt_local;// shear zone thicknesses
  float a_uw;// the attenuation due to one row (to be raised to N_e, where N_e is number of rows in entry region)
  float a_local;// the attenuation due to the closest upwind row only
  float u_c0, v_c0;// u0 and v0 rotated into row-aligned coordinates
  float u_c, v_c;// altered (parameterized) u and v, in row-aligned coordinates

  // DEBUG VARIABLES
  bool BF_flag = 1;
  int UD_zone_flag = 1;

  // BEGIN MAIN PARAMETERIZATION LOOPS
  for (auto j = 0; j < ny_canopy; j++) {
    for (auto i = 0; i < nx_canopy; i++) {
      icell_2d = i + j * nx_canopy;


      // CALCULATE VARIOUS QUANTITIES USED IN THE PARAMETERIZATIONS

      // base of the canopy
      z_b = canopy_base[icell_2d];

      // calculate row-orthogonal distance to upwind-est row, dv_c
      float cxy[2] = { (i - 1 + i_start) * WGD->dx, (j - 1 + j_start) * WGD->dy };// current x position

      float Rx_o[2] = { polygonVertices[idS].x_poly, polygonVertices[idS].x_poly + rd[0] };// x-components of upwind-est row vector
      float Ry_o[2] = { polygonVertices[idS].y_poly, polygonVertices[idS].y_poly + rd[1] };// y-components "   "   "

      dv_c = abs(P2L(cxy, Rx_o, Ry_o));

      // calculate number of rows upwind of this point, N (not including nearest upwind row)
      N = floor(dv_c / rowSpacing);

      // calculate "local row-orthogonal distance" from upwind-est row
      ld = dv_c - N * rowSpacing;

      // calculate "local streamwise distance" from nearest row
      float d_dw_local = (ld - rowWidth) / sin(beta * M_PI / 180);

      // find blending factor "a_exp" at the current i,j location (a function of d_dw_local)
      a_exp = exp((log(0.01) / (7.5 * H)) * d_dw_local);

      // find k-node of mid-canopy at current i,j location
      for (auto k = canopy_bot_index[icell_2d]; k < canopy_top_index[icell_2d]; k++) {
        if (WGD->z[k] > (z_b + understory_height + (H - understory_height) / 2)) {
          break;
        }
        k_mid = k;
      }


      // Find matching factor that matches upper/lower tanh profiles at k_mid

      szt_top = spreadrate_top * d_dw_local + 0.00001;// add a small number to avoid dividing by zero
      szt_bot = spreadrate_bot * d_dw_local + 0.00001;

      fac = (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (WGD->z[k_mid] - (szo_top + z_b)) / szt_top) + 0.5 * (1 + a_obf)) + (1 - a_exp)) / (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (szo_bot + z_b - WGD->z[k_mid]) / szt_bot) + 0.5 * (1 + a_obf)) + (1 - a_exp));// upper/lower

      // Calculate quantities for UD zone
      x_ud = rowSpacing - ld;// right now I'm using the row-orthogonal distance, not streamwise distance. Perhaps should use streamwise distance so that UD zone responds appropriately to wind angle.
      z_ud = H_ud * sqrt(abs(1 - x_ud / l_ud));
      br = log(1 - 0.99) / (z_ud * (1 - 0.2));// reaches 0.99 at 0.2*z_ud below z_ud
      //std::cout << "starting zref_k search \n";
      // v_c parameterization variables
      zref = 10.;// arbitrarily choose 10m for reference velocity. Doesn't really matter.
      zref_k = 0;
      for (auto k = canopy_bot_index[icell_2d]; k < WGD->nz; k++) {
        //std::cout << " z[k] = " << WGD->z[k] << " zref + z_b = " << zref + z_b << "\n";
        if (WGD->z[k] > (zref + z_b)) {
          //      std::cout << "zref_k found to be " << zref_k << "\n";
          break;
        }
        zref_k = k;
      }

      // BEGIN PARAMETERIZATIONS

      if (canopy_bot_index[icell_2d] < canopy_top_index[icell_2d]) {// if i'm inside the canopy (vineyard block) polygon
        //std::cout << "C1 \n";
        icell_face_ref = (i - 1 + i_start) + (j - 1 + j_start) * WGD->nx + zref_k * WGD->nx * WGD->ny;
        //std::cout << "icell_face_ref = " << icell_face_ref << "zref_k = " << zref_k << "\n";
        //std::cout << "z_b = " << z_b << "\n";
        vref_c = sinA * WGD->u0[icell_face_ref] + cosA * WGD->v0[icell_face_ref];// The sign of vref_c (the rotated reference velocity) is preserved (rather than abs() ) so that rotation of the final parameterized v0_c is correct. It determines the sign of ustar_v_c which determines sign of vH_c which determines sign of v_c
        //std::cout << "C2 \n";
        ustar_v_c = WGD->vk * vref_c / log((zref - d_v) / z0_site);// d_v (displacement height from Nate's data)
        vH_c = ustar_v_c / WGD->vk * (log((H - d_v) / z0_site)) - psiH;
        a_v = (H / abs(vH_c + 0.00001)) * (abs(ustar_v_c) / (WGD->vk * (H - d_v)) - dpsiH);// abs() here because the attenuation coefficient should be always positive. add 0.00001 to avoid divide by zero

        if (i + i_start == i_building_cent && j + j_start == j_building_cent) {
          std::cout << "vref_c = " << vref_c << "\n";
          std::cout << "ustar_v_c = " << ustar_v_c << "\n";
          std::cout << "vH_c = " << vH_c << "\n";
          std::cout << "a_v = " << a_v << "\n";
          std::cout << "z0_site = " << z0_site << "\n";
          std::cout << "rL = " << rL << "\n";
        }

        for (auto k = canopy_bot_index[icell_2d]; k < WGD->nz; k++) {

          z_rel = WGD->z_face[k - 1] - z_b;
          int icell_face = (i - 1 + i_start) + (j - 1 + j_start) * WGD->nx + k * WGD->nx * WGD->ny;
          int icell_cent = (i - 1 + i_start) + (j - 1 + j_start) * (WGD->nx - 1) + k * (WGD->nx - 1) * (WGD->ny - 1);


          // Rotate u0 and v0 into row-aligned coords
          u_c0 = cosA * WGD->u0[icell_face] - sinA * WGD->v0[icell_face];
          v_c0 = sinA * WGD->u0[icell_face] + cosA * WGD->v0[icell_face];


          // Include effect of UD zone on the upwind u_c0 velocity
          if (z_rel < H_ud) {
            u_c0 = u_c0 + (a_obf * u_c0 - 2 * u_c0) * (1 - exp(br * (H_ud - z_rel)));
          }


          if (k <= k_mid) {// if i'm below mid-canopy, use bottom shear layer quantities
            szt_uw = spreadrate_bot * d_dw + 0.00001;// d_dw = streamwise downwind distance
            szt_local = spreadrate_bot * d_dw_local + 0.00001;
            a_uw = fac * (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (szo_bot - z_rel) / szt_uw) + 0.5 * (1 + a_obf)) + (1 - a_exp));
            a_local = fac * (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (szo_bot - z_rel) / szt_local) + 0.5 * (1 + a_obf)) + (1 - a_exp));
          } else {
            szt_uw = spreadrate_top * d_dw + 0.00001;
            szt_local = spreadrate_top * d_dw_local + 0.00001;
            a_uw = (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (z_rel - szo_top) / szt_uw) + 0.5 * (1 + a_obf)) + (1 - a_exp));
            a_local = (a_exp * (0.5 * (1 - a_obf) * tanh(1.5 * (z_rel - szo_top) / szt_local) + 0.5 * (1 + a_obf)) + (1 - a_exp));
          }


          // APPLY BLEED FLOW PARAMETERIZATION INSIDE THE ACTUAL VEGETATION
          if (BF_flag == 1 && ld < rowWidth) {// if my x-y position indicates i'm on a vine
            if ((z_rel >= understory_height) && (z_rel <= H)) {// if my z position indicates i'm inside the actual vegetation
              //int icell_3d = i + j*nx_canopy + k*nx_canopy*ny_canopy;


              // Am I in entry region or /quilibrated region?
              if (N <= N_e) {// i'm in entry region

                WGD->icellflag[icell_cent] = 28;
                u0_mod_id.push_back(icell_face);
                u0_modified.push_back(pow(a_uw, N - 1) * a_obf * (cosA * WGD->u0[icell_face] - sinA * WGD->v0[icell_face]) * cosA + (sinA * WGD->u0[icell_face] + cosA * WGD->v0[icell_face]) * sinA);
                v0_mod_id.push_back(icell_face);
                v0_modified.push_back(pow(a_uw, N - 1) * a_obf * (cosA * WGD->u0[icell_face] - sinA * WGD->v0[icell_face]) * (-sinA) + (sinA * WGD->u0[icell_face] + cosA * WGD->v0[icell_face]) * cosA);

              }

              else {// if i'm in equilibrated region

                WGD->icellflag[icell_cent] = 29;
                u0_mod_id.push_back(icell_face);
                u0_modified.push_back(pow(a_uw, N_e) * a_obf * (cosA * WGD->u0[icell_face] - sinA * WGD->v0[icell_face]) * cosA + (sinA * WGD->u0[icell_face] + cosA * WGD->v0[icell_face]) * sinA);
                v0_mod_id.push_back(icell_face);
                v0_modified.push_back(pow(a_uw, N_e) * a_obf * (cosA * WGD->u0[icell_face] - sinA * WGD->v0[icell_face]) * (-sinA) + (sinA * WGD->u0[icell_face] + cosA * WGD->v0[icell_face]) * cosA);
              }
            }// end bleed flow z if
          }// end bleed flow x-y if

          // APPLY WAKE PARAMETERIZATION
          else {// if i'm in the wake

            WGD->icellflag[icell_cent] = 30;


            // Am I in entry region or equilibrated region?
            if (N <= N_e) {// i'm in entry region
              u_c = u_c0 * pow(a_uw, N) * a_local;
              // v_c = v_c0;
            }

            else {// if i'm in equilibrated region
              u_c = u_c0 * pow(a_uw, N_e) * a_local;
              // v_c = v_c0;
            }

            // APPLY UD ZONE PARAMETERIZATION
            if (UD_zone_flag && ((rowSpacing - ld) < l_ud) && (z_rel <= z_ud)) {// if i'm ALSO in the UD zone
              //  If the rotated u-velocity was negative in the UD zone, this reverses it and then SUBTRACTS bleed flow. If the rotated u-velocity was positive in the UD zone, this reverses it and then ADDS bleed flow
              WGD->icellflag[icell_cent] = 31;

              if (N <= N_e) {
                if (i + i_start == 156 && j + j_start == 60 && k < 100) {
                  std::cout << "k = " << k << "     u_c = " << u_c << "       gz = " << (a_obf * pow(a_uw, N) * u_c0 - 2 * u_c) * (1 - exp(br * (z_ud - z_rel))) << "     br = " << br << "     x_ud = " << x_ud << "     1st chunk = " << a_obf * pow(a_uw, N) * u_c0 << "\n";
                }
                u_c = u_c + (a_obf * pow(a_uw, N) * u_c0 - 2 * u_c) * (1 - exp(br * (z_ud - z_rel)));
                //u_c = -u_c;
              } else {
                if (i + i_start == 156 && j + j_start == 60 && k < 100) {
                  std::cout << "k = " << k << "     u_c = " << u_c << "     gz = " << (a_obf * pow(a_uw, N_e) * u_c0 - 2 * u_c) * (1 - exp(br * (z_ud - z_rel))) << "     br = " << br << "     x_ud = " << x_ud << "     z_ud = " << z_ud << "      1st chunk = " << a_obf * pow(a_uw, N_e) * u_c0 << "\n";
                }
                u_c = u_c + (a_obf * pow(a_uw, N_e) * u_c0 - 2 * u_c) * (1 - exp(br * (z_ud - z_rel)));
                //u_c = -u_c;
              }

              // v_c = v_c0;
            }

            // APPLY V_C PARAMETERIZATION
            if (rL >= 0.) {
              psi = -5.0 * z_rel * rL;
              psiH = -5.0 * H * rL;
              dpsiH = -5.0 * rL;
            } else {
              ex = pow((1. - 15. * z_rel * rL), 0.25);
              psi = log((0.5 * (1. + pow(ex, 2))) * pow(0.5 * (1. + ex), 2)) - 2. * atan(ex) + M_PI * 0.5;
              exH = pow(1. - 15. * H * rL, .25);
              psiH = log((0.5 * (1. + pow(exH, 2))) * pow(0.5 * (1. + exH), 2)) - 2. * atan(exH) + M_PI * 0.5;

              dexH = -15. / 4. * rL * pow(1. - 15. * H * rL, -0.75);
              dpsiH = (exH * dexH * (1. + exH) * 0.5 + 0.5 * (1. + pow(exH, 2)) * dexH) / (0.25 * (1. + pow(exH, 2)) * (1. + exH)) - (dexH / (1. + pow(exH, 2)));
            }

            if (z_rel <= H) {
              v_c = vH_c * exp(a_v * (z_rel / H - 1));
            } else {
              v_c = ustar_v_c / WGD->vk * (log((z_rel - d_v) / z0_site)) - psi;
            }

            if (abs(v_c) >= abs(v_c0)) {// if the parameterization gives a higher abs value for v than what's already there (v_c0)...
              v_c = v_c0;// ...then just take v_c0
            }


            if (N == 3 && j + j_start == 240 && z_rel <= H) {
              std::cout << "modified u in UD is: " << cosA * u_c + sinA * v_c << "\n";
            }

          }// end wake else

          // Rotate back into QES-grid coordinates
          u0_mod_id.push_back(icell_face);
          u0_modified.push_back(cosA * u_c + sinA * v_c);
          v0_mod_id.push_back(icell_face);
          v0_modified.push_back(-sinA * u_c + cosA * v_c);

        }// end k-loop
      }// end canopy_bottom_index < canopy_top_index if

      // adding modified velocity to the list of node to modifiy
      /*
                    // all face of the cell i=icell_face & i+1 = icell_face+1
                    u0_mod_id.push_back(icell_face);
                    u0_modified.push_back(a_obf*us_mag*cos(us_dir));
                    u0_mod_id.push_back(icell_face+1);
                    u0_modified.push_back(a_obf*us_mag*cos(us_dir));
                    // all face of the cell j=icell_face & j+1 = icell_face+nx
                    v0_mod_id.push_back(icell_face);
                    v0_modified.push_back(a_obf*us_mag*sin(us_dir));
                    v0_mod_id.push_back(icell_face+WGD->nx);
                    v0_modified.push_back(a_obf*us_mag*sin(us_dir));
                    */

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

void CanopyVineyard::canopyWake(WINDSGeneralData *WGD, int building_id)
{


  return;
}
