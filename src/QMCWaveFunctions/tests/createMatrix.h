
#ifndef QMCPLUSPLUS_CREATEMATRIX_H
#define QMCPLUSPLUS_CREATEMATRIX_H

namespace qmcplusplus {


// Create an identity matrix.
// Assumes the input matrix is zero.
// Assumes matrix is square (or at least the number of rows >= number of columns)
template<typename T> void createIdentityMatrix(Matrix<T>& m)
{
  for (int i = 0; i < m.cols(); i++) {
    m(i,i) = 1.0;
  }
}


// Inverse test 1
// Assumes input matrix is 3x3
template<typename T> void testMatrix1_Original(Matrix<T>& a)
{
  a(0, 0) = 2.3;
  a(0, 1) = 4.5;
  a(0, 2) = 2.6;
  a(1, 0) = 0.5;
  a(1, 1) = 8.5;
  a(1, 2) = 3.3;
  a(2, 0) = 1.8;
  a(2, 1) = 4.4;
  a(2, 2) = 4.9;
}

// Inverse of test matrix 1
template<typename T> void testMatrix1_Inverse(Matrix<T>& b)
{
  b(0, 0) = 0.6159749342;
  b(0, 1) = -0.2408954682;
  b(0, 2) = -0.1646081192;
  b(1, 0) = 0.07923894288;
  b(1, 1) = 0.1496231042;
  b(1, 2) = -0.1428117337;
  b(2, 0) = -0.2974298429;
  b(2, 1) = -0.04586322768;
  b(2, 2) = 0.3927890292;
}

// Log of determinant of test matrix 1
double testMatrix1_logDet() {
  return 3.78518913425;
}

}
#endif
