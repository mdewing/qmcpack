//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
//
// File created by: Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#include "catch.hpp"

#include "Configuration.h"
#include "Platforms/CPU/SIMD/aligned_allocator.hpp"
#include "QMCWaveFunctions/Fermion/DiracMatrix.h"
#include "QMCWaveFunctions/Fermion/cuSolverInverter.hpp"
#include "checkMatrix.hpp"
#include "createMatrix.h"

namespace qmcplusplus
{

using ValueType = QMCTraits::ValueType;
using LogValueType = std::complex<QMCTraits::QTFull::RealType>;

TEST_CASE("cuSolverInverter_identity", "[wavefunction][fermion]")
{
  cuSolverInverter<double> solver;
  const int N = 3;

  Matrix<ValueType> m, m_invT;
  Matrix<ValueType, CUDAAllocator<ValueType>> m_invGPU;
  LogValueType log_value;
  m.resize(N, N);
  m_invT.resize(N, N);
  m_invGPU.resize(N, N);

  createIdentityMatrix(m);

  solver.invert_transpose(m, m_invT, m_invGPU, log_value);
  REQUIRE(log_value == LogComplexApprox(0.0));

  Matrix<ValueType> eye;
  eye.resize(3, 3);
  createIdentityMatrix(eye);

  CheckMatrixResult check_matrix_result = checkMatrix(m_invT, eye);
  CHECKED_ELSE(check_matrix_result.result) { FAIL(check_matrix_result.result_message); }

}

TEST_CASE("cuSolverInverter_inverse", "[wavefunction][fermion]")
{
  cuSolverInverter<double> solver;
  const int N = 3;

  Matrix<ValueType> a, a_T, a_inv;
  Matrix<ValueType, CUDAAllocator<ValueType>> m_invGPU;
  LogValueType log_value;
  a.resize(N, N);
  a_T.resize(N, N);
  a_inv.resize(N, N);
  m_invGPU.resize(N, N);

  testMatrix1_Original(a);

  simd::transpose(a.data(), a.rows(), a.cols(), a_T.data(), a_T.rows(), a_T.cols());
  solver.invert_transpose(a_T, a_inv, m_invGPU, log_value);
  REQUIRE(log_value == LogComplexApprox(testMatrix1_logDet()));

  Matrix<ValueType> b;
  b.resize(3, 3);

  testMatrix1_Inverse(b);

  auto check_matrix_result = checkMatrix(a_inv, b);
  CHECKED_ELSE(check_matrix_result.result) { FAIL(check_matrix_result.result_message); }
}

}

