#include "math.h"
#include "stdlib.h"

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

/* 
  Define the struct for mesh parameters and add2s2i_int function
*/
struct hpe_mesh_params {
  uint nx1;
  uint ny1;
  uint nz1;
  uint mx;
  uint my;
  uint mz;
  uint ntot;
  uint nelt;
  uint maxsize;
};

void add2s2i_int(
                 void *aptr,
                 void *bptr,
                 double c1,
                 uint ntot,
                 uint find,
                 uint lind,
                 const uint *map);

void add2s2i_bdy(
                 void *aptr,
                 void *bptr,
                 double c1,
                 uint ntot,
                 uint find,
                 uint lind,
                 const uint *map);
