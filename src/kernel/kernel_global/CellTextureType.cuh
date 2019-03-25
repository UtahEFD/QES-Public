 /*
* CellTextureType.cuh
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

#ifndef __CELLTEXTURE_CUH__
#define __CELLTEXTURE_CUH__

enum CellTextureType{WINDFIELDTEX, EIGVALTEX, KA0TEX, G2NDTEX,
////////////////  matrix 9////////////////
		     EIGVEC1TEX, EIGVEC2TEX, EIGVEC3TEX,
		     EIGVECINV1TEX, EIGVECINV2TEX, EIGVECINV3TEX,
		     LAM1TEX, LAM2TEX, LAM3TEX, 
//////////////// matrix6 ////////////////
		     SIG1TEX, SIG2TEX, TAUDX1TEX, TAUDX2TEX,
		     TAUDY1TEX, TAUDY2TEX,TAUDZ1TEX, TAUDZ2TEX};


#endif
