//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                    Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
//
// File created by: Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include "LinearMethod.h"
#include <vector>
#include "QMCCostFunctionBase.h"
#include <magma_v2.h>
#include <magma_lapack.h>


namespace qmcplusplus
{
namespace MatrixOperators
{

template<typename T>
inline void product(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

}

template<class MatrixA>
inline typename MatrixA::value_type invert_matrix(MatrixA& M, bool getdet = true);

// A - Ham
// B - Overlap

LinearMethod::Real LinearMethod::solveGeneralizedEigenvaluesMagma(Matrix<Real>& A,
                                                                  Matrix<Real>& B,
                                                                  std::vector<Real>& eigenvals,
                                                                  Matrix<Real>& eigenvectors)
{
  magma_queue_t queue;
  if (!magma_initialized_)
  {
    magma_init();
    magma_int_t dev = 0;
    magma_queue_create(dev, &queue);
    magma_initialized_ = true;
    magma_queue_       = (int*)queue;
  }
  else
  {
    queue = (magma_queue_t)magma_queue_;
  }

  const int N = eigenvals.size();
  assert(N == A.rows());
  assert(N == A.cols());
  assert(N == B.rows());
  assert(N == B.cols());
  assert(N == eigenvectors.rows());
  assert(N == eigenvectors.cols());

#if 0
  app_log() << "before inv magma " << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      app_log() << " " << B.data()[N*i+j] << " ";
    }
    app_log() << std::endl;
  }
#endif

  double* d_B;
  magma_int_t ret = magma_dmalloc(&d_B, N * N);
  if (ret != MAGMA_SUCCESS)
  {
    std::cout << "magma dmalloc for d_B failed\n";
  }
  magma_dsetmatrix(N, N, B.data(), N, d_B, N, queue);

  std::vector<int> pivot(N);
  int info   = 0;
  int status = magma_dgetrf_gpu(
      //int status = magma_dgetrf_native(
      N, N, d_B, N, pivot.data(), &info);

  if (status != MAGMA_SUCCESS)
  {
    std::cout << " status from dgetrf = " << status << std::endl;
    std::cout << " info from degetrf = " << info << std::endl;
  }

  int nb    = magma_get_dgetri_nb(N);
  int lwork = nb * N;

  double* d_work;
  ret = magma_dmalloc(&d_work, lwork);
  if (ret != MAGMA_SUCCESS)
  {
    std::cout << "magma dmalloc for dwork failed\n";
  }


  info   = 0;
  status = magma_dgetri_gpu(N, d_B, N, pivot.data(), d_work, lwork, &info);

  if (status != MAGMA_SUCCESS)
  {
    std::cout << " status from dgetri = " << status << std::endl;
    std::cout << " info from degetri = " << info << std::endl;
  }

  double one(1.0);
  double zero(0.0);

  char transa('N');
  char transb('N');

  std::vector<double> prd(N * N);
#if 0
  magma_dgetmatrix(N, N, d_B, N, prd.data(), N, queue);
  app_log() << "after inv magma " << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      app_log() << " " << prd[N*i+j] << " ";
    }
    app_log() << std::endl;
  }
#endif


  double* d_hamiltonian;
  ret = magma_dmalloc(&d_hamiltonian, N * N);
  if (ret != MAGMA_SUCCESS)
  {
    std::cout << "magma dmalloc for d_hamiltonian failed\n";
  }
  magma_dsetmatrix(N, N, A.data(), N, d_hamiltonian, N, queue);
#if 0
  app_log() << "A before mult magma " << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      app_log() << " " << A.data()[N*i+j] << " ";
    }
    app_log() << std::endl;
  }
#endif

  double* d_C;
  ret = magma_dmalloc(&d_C, N * N);
  if (ret != MAGMA_SUCCESS)
  {
    std::cout << "magma dmalloc for d_C failed\n";
  }
  magma_dgemm(MagmaNoTrans, MagmaNoTrans, N, N, N, one, d_hamiltonian, N, d_B, N, zero, d_C, N, queue);


  magma_dgetmatrix(N, N, d_C, N, prd.data(), N, queue);

  magma_queue_sync(queue);


#if 1
  // do transpose (why?)
  for (int i = 0; i < N; i++)
  {
    for (int j = i + 1; j < N; j++)
    {
      std::swap(prd.data()[N * i + j], prd.data()[N * j + i]);
    }
  }
#endif

#if 0
  app_log() << "prod magma " << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      app_log() << " " << prd[N*i+j] << " ";
    }
    app_log() << std::endl;
  }
#endif
  double zerozero = prd[0];

  std::vector<double> alphar(N);
  std::vector<double> alphai(N);
  std::vector<double> vl(N * N);
  //std::vector<double> vr(N*N);

  char jl('N');
  char jr('V');

  info = 0;

  lwork = -1;
  std::vector<double> work(1);
  magma_dgeev(MagmaNoVec, MagmaVec, N, prd.data(), N, alphar.data(), alphai.data(), vl.data(), N, eigenvectors.data(),
              N, work.data(), lwork, &info);


  if (info != 0)
    std::cout << "dgeev lwork error, info = " << info << std::endl;

  lwork = work[0];

  work.resize(lwork);

  magma_dgeev(MagmaNoVec, MagmaVec, N, prd.data(), N, alphar.data(), alphai.data(), vl.data(), N, eigenvectors.data(),
              N, work.data(), lwork, &info);

  if (info != 0)
  {
    std::cout << "dgeev error, info = " << info << std::endl;
  }

  for (int i = 0; i < N; i++)
  {
    eigenvals[i] = alphar[i];
  }

  magma_free(d_C);
  magma_free(d_hamiltonian);
  magma_free(d_work);
  magma_free(d_B);

  return zerozero;
}

LinearMethod::Real LinearMethod::getLowestEigenvectorMagma(Matrix<Real>& A, Matrix<Real>& B, std::vector<Real>& ev)
{
  const int N = A.rows();
  assert(N == ev.size());
  Matrix<Real> eigenvectors(N, N);
  std::vector<Real> alphar(N);
  double zerozero = solveGeneralizedEigenvaluesMagma(A, B, alphar, eigenvectors);


  //double zerozero = prod[0];


#if 0
  std::vector<double> adjusted_ev(N);
  std::transform(alphar.begin(), alphar.end(), adjusted_ev.begin(),
      [&] (double eig) -> double {return (eig - zerozero + 2.0)*(eig - zerozero + 2.0);});

  std::vector<double>::iterator min_elem = std::min_element(adjusted_ev.begin(), adjusted_ev.end());
  int idx = std::distance(adjusted_ev.begin(), min_elem);
  double lowest_ev = alphar[idx];

  std::cout << "Index of min eigenvalue = " << idx << std::endl;
  std::cout << "Eigenvectors" << std::endl;
  for (int i = 0; i < N; i++) {
    double eval = eigenvectors.data()[idx*N + i];
    ev[i] = eval/eigenvectors.data()[idx*N];
    std::cout << i << " " << eval << " " << ev[i] << std::endl;
  }
#endif

  std::vector<std::pair<Real, int>> mappedEigenvalues(N);
  for (int i = 0; i < N; i++)
  {
    Real evi(alphar[i]);
    if ((evi < zerozero) && (evi > (zerozero - 1e2)))
    {
      mappedEigenvalues[i].first  = (evi - zerozero + 2.0) * (evi - zerozero + 2.0);
      mappedEigenvalues[i].second = i;
    }
    else
    {
      mappedEigenvalues[i].first  = std::numeric_limits<Real>::max();
      mappedEigenvalues[i].second = i;
    }
  }
  std::sort(mappedEigenvalues.begin(), mappedEigenvalues.end());

  for (int i = 0; i < N; i++)
  {
    //app_log() << i << " Raw evec " << eigenvectors(mappedEigenvalues[0].second, i) << std::endl;
    ev[i] = eigenvectors(mappedEigenvalues[0].second, i) / eigenvectors(mappedEigenvalues[0].second, 0);
  }

  double lowest_ev = alphar[mappedEigenvalues[0].second];

  return lowest_ev;
}

LinearMethod::~LinearMethod()
{
  if (magma_initialized_)
  {
    magma_queue_destroy((magma_queue_t)magma_queue_);
    magma_finalize();
    magma_initialized_ = false;
  }
}


} // namespace qmcplusplus
