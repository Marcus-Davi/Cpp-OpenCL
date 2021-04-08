__kernel void matrixMult(__global float *A, __global float *B,
                         __global float *C,int wA, int wB) {

  int tx = get_global_id(0);
  int ty = get_global_id(1);

  float value = 0;
  for (int k = 0; k < wA; ++k){
  //     // dot product
      value += A[ty * wA + k] * B[k * wB + tx];
  }
  C[ty * wA + tx] = value;
  
}   