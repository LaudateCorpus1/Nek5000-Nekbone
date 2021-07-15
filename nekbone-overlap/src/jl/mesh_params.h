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
};

void add2s2i_int(
                 void *aptr,
                 void *bptr,
                 double c1,
                 uint ntot,
                 uint find,
                 uint lind,
                 uint *map);
