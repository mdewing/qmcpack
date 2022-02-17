

#include "catch.hpp"

#include "Configuration.h"
#include "Platforms/CPU/SIMD/aligned_allocator.hpp"
#include "QMCWaveFunctions/Fermion/DiracMatrix.h"

#include "QMCWaveFunctions/Fermion/cuSolverInverter.hpp"

#include "makeRngSpdMatrix.hpp"


namespace qmcplusplus
{


TEST_CASE("cuSolverInverter_bench","[wavefunction][benchmark]")
{
  cuSolverInverter<double> solver;

  int N = 1024;

  Matrix<double> m, m_invT;
  Matrix<double, CUDAAllocator<double>> m_invGPU;
  std::complex<double> log_value;
  m.resize(N, N);
  m_invT.resize(N, N);
  m_invGPU.resize(N, N);

  testing::MakeRngSpdMatrix<double> makeRngSpdMatrix{};
  makeRngSpdMatrix(m);

  BENCHMARK("cuSolverInverter") {
    solver.invert_transpose(m, m_invT, m_invGPU, log_value);
  };


  DiracMatrix<double> dmat;
  BENCHMARK("CPU") {
    dmat.invert_transpose(m, m_invT, log_value);
  };
}

}
