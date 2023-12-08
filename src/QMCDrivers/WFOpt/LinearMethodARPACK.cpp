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
#include "Numerics/DeterminantOperators.h"
#include "CPU/BLAS.hpp"

// ARPACK-NG
// https://github.com/opencollab/arpack-ng.git
// Build with "ICB=ON" to enable the C++ interface
#include <arpack.hpp>


namespace qmcplusplus
{
// A - Ham
// B - Overlap

// Shift-invert generalized eigenproblem
LinearMethod::Real LinearMethod::solveGeneralizedEigenvaluesARPACK(Matrix<Real>& A,
                                                                   Matrix<Real>& B,
                                                                   std::vector<Real>& eigenvals,
                                                                   Matrix<Real>& eigenvectors)
{
  const int N   = A.rows();
  const int Nev = eigenvals.size();
  assert(N == A.cols());
  assert(N == B.rows());
  assert(N == B.cols());
  assert(N == eigenvectors.rows());
  assert(Nev == eigenvectors.cols());

  double zerozero = A(0, 0);
  //double sigma = zerozero - 2.0;
  double sigma = 1.2 * zerozero;
  //double sigma = zerozero;
  //double sigma = zerozero - 100;
  //double sigma = zerozero - 1000;
  app_log() << "H(0,0) = " << zerozero << " sigma = " << sigma << std::endl;

  std::vector<int> pivot(N);
  std::vector<double> C_matrix(N * N);
  // Compute C = A - sigma*B and convert to column-major

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      C_matrix[i + j * N] = A(i, j) - sigma * B(i, j);
  //C_matrix[j + i*N] = A(i,j) - sigma*B(i,j);

  LUFactorization(N, N, C_matrix.data(), N, pivot.data());

  // Set up for ARPACK
  int ido         = 0;
  const int maxit = 200;
  double tol      = 1e-4;
  std::vector<double> resid(N);
  //int ncv = std::min(Nev+2, 2*Nev + 1);
  int ncv = std::min(N, 61);
  //int ncv = 81;
  int ldv = N;
  std::vector<double> v(ncv * N);
  int lworkl = 3 * ncv * ncv + 6 * ncv;
  std::vector<double> workd(3 * N);
  std::vector<double> workl(lworkl);

  std::vector<int> iparam(11);
  std::vector<int> ipntr(14);

  iparam[0] = 1;     // ishifts
  iparam[2] = maxit; // maxitr
  iparam[3] = 1;     // nb(blocksize) (should be 1)
  iparam[6] = 3;     // mode

  int info = 0;

  for (int i = 0; i < 1000 * maxit; i++)
  {
    naupd(ido, arpack::bmat::generalized, N, arpack::which::largest_magnitude, Nev, tol, resid.data(), ncv, v.data(),
          ldv, iparam.data(), ipntr.data(), workd.data(), workl.data(), lworkl, info);

    if (info < 0)
      app_log() << "Error from naupd, info = " << info << std::endl;
    if (info == 1)
      std::cout << "naupd max it reached" << std::endl;
    if (ido == 99)
      break;

    if (ido == -1)
    {
      //  y = inv(C)*B*x
      //
      info = 0;
      char trans('T');
      char notrans('N');
      double alpha = 1.0;
      double beta  = 0.0;
      int inc      = 1;
      BLAS::gemv(N, N, B.data(), &workd[ipntr[0] - 1], &workd[ipntr[1] - 1]);
      //std::cout << "starting dgetrs" << std::endl;
      int nrhs = 1;
      //dgetrs(&trans, &N, &nrhs, overlap_copy.data(), &N, pivot.data(), tmp.data(), &N, &info);
      dgetrs(notrans, N, nrhs, C_matrix.data(), N, pivot.data(), &workd[ipntr[1] - 1], N, info);
      if (info != 0)
      {
        std::cout << "dgetrs error, info = " << info << std::endl;
      }
      //for (int i = 0; i < N; i++) {
      //  workd[ipntr[1]-1 + i] = tmp[i];
      //}
      //std::cout << "done with dgetrs" << std::endl;
    }
    if (ido == 1)
    {
      //  y = inv(C)*B*x
      // B*x is saved in workd[ipntr[2]-1]
      for (int i = 0; i < N; i++)
        workd[ipntr[1] - 1 + i] = workd[ipntr[2] - 1 + i];

      info         = 0;
      double alpha = 1.0;
      double beta  = 0.0;
      int inc      = 1;

      int nrhs = 1;
      char notrans('N');
      //dgetrs(&trans, &N, &nrhs, overlap_copy.data(), &N, pivot.data(), tmp.data(), &N, &info);
      dgetrs(notrans, N, nrhs, C_matrix.data(), N, pivot.data(), &workd[ipntr[1] - 1], N, info);
      if (info != 0)
      {
        std::cout << "dgetrs error, info = " << info << std::endl;
      }
    }


    if (ido == 2)
    {
      // Matrix-vector mult from workd
      // y = B*x
      char trans('N');
      double alpha = 1.0;
      double beta  = 0.0;
      int inc      = 1;
      int info     = 0;
      //dgemv(&trans, &N, &N, &alpha, overlap, &N, &workd[ipntr[0]-1], &inc, &beta, &workd[ipntr[1]-1], &inc);
      BLAS::gemv_trans(N, N, B.data(), &workd[ipntr[0] - 1], &workd[ipntr[1] - 1]);
    }
  }

  // now get the eigenvalues and eigenvectors
  int rvec = true;
  std::vector<int> select(ncv);
  std::vector<double> dr(Nev + 1);
  std::vector<double> di(Nev + 1);
  std::vector<double> Z(N * (Nev + 1));
  int ldz       = N;
  double sigmar = sigma;
  double sigmai(0.0);
  std::vector<double> workev(3 * N);
  arpack::neupd(rvec, arpack::howmny::schur_vectors, select.data(), dr.data(), di.data(), Z.data(), ldz, sigmar, sigmai,
                workev.data(), arpack::bmat::generalized, N, arpack::which::largest_magnitude, Nev, tol, resid.data(),
                ncv, v.data(), ldv, iparam.data(), ipntr.data(), workd.data(), workl.data(), lworkl, info);


  if (info < 0)
  {
    std::cout << "neupd error, info = " << info << std::endl;
  }


  std::cout << "nconv (iparam(4)) = " << iparam[4] << std::endl;
  for (int i = 0; i < Nev; i++)
  {
    std::cout << "evec i " << i << " " << dr[i] << " imag " << di[i] << std::endl;
    eigenvals[i] = dr[i];
  }

  for (int i = 0; i < Nev; i++)
  {
    for (int j = 0; j < N; j++)
    {
      eigenvectors(i, j) = v.data()[j + N * i];
    }
  }

  return A(0, 0);
}

// A - Ham
// B - Overlap

LinearMethod::Real LinearMethod::solveGeneralizedEigenvaluesARPACK_mode2(Matrix<Real>& A,
                                                                         Matrix<Real>& B,
                                                                         std::vector<Real>& eigenvals,
                                                                         Matrix<Real>& eigenvectors)
{
  const int N   = A.rows();
  const int Nev = eigenvals.size();
  assert(N == A.cols());
  assert(N == B.rows());
  assert(N == B.cols());
  assert(N == eigenvectors.rows());
  assert(Nev == eigenvectors.cols());

  std::vector<int> pivot(N);
  std::vector<double> overlap_copy(N * N);
  // Copy and convert to column-major
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      overlap_copy[i + j * N] = B(i, j);

  LUFactorization(N, N, overlap_copy.data(), N, pivot.data());

  // Set up for ARPACK
  int ido         = 0;
  const int maxit = 200;
  double tol      = 1e-3;
  std::vector<double> resid(N);
  int ncv = std::max(N, 3 * Nev + 1);
  int ldv = N;
  std::vector<double> v(ncv * N);
  int lworkl = 3 * ncv * ncv + 6 * ncv;
  std::vector<double> workd(3 * N);
  std::vector<double> workl(lworkl);

  std::vector<int> iparam(11);
  std::vector<int> ipntr(14);

  iparam[0] = 1;     // ishifts
  iparam[2] = maxit; // maxitr
  iparam[3] = 0;     // nb(blocksize) (should be 1)
  iparam[6] = 2;     // mode

  int info = 0;

  for (int i = 0; i < 1000 * maxit; i++)
  {
    naupd(ido, arpack::bmat::generalized, N, arpack::which::largest_magnitude, Nev, tol, resid.data(), ncv, v.data(),
          ldv, iparam.data(), ipntr.data(), workd.data(), workl.data(), lworkl, info);

    if (info < 0)
      app_log() << "Error from naupd, info = " << info << std::endl;
    if (info == 1)
      std::cout << "naupd max it reached" << std::endl;
    if (ido == 99)
      break;

    if (ido == 1 || ido == -1)
    {
      //  y = inv(B)*A*x
      //
      info = 0;
      char trans('T');
      double alpha = 1.0;
      double beta  = 0.0;
      int inc      = 1;
      BLAS::gemv_trans(N, N, A.data(), &workd[ipntr[0] - 1], &workd[ipntr[1] - 1]);
      //std::cout << "starting dgetrs" << std::endl;
      int nrhs = 1;
      //dgetrs(&trans, &N, &nrhs, overlap_copy.data(), &N, pivot.data(), tmp.data(), &N, &info);
      dgetrs(trans, N, nrhs, overlap_copy.data(), N, pivot.data(), &workd[ipntr[1] - 1], N, info);
      if (info != 0)
      {
        std::cout << "dgetrs error, info = " << info << std::endl;
      }
      //for (int i = 0; i < N; i++) {
      //  workd[ipntr[1]-1 + i] = tmp[i];
      //}
      //std::cout << "done with dgetrs" << std::endl;
    }

    if (ido == 2)
    {
      // Matrix-vector mult from workd
      // y = B*x
      char trans('N');
      double alpha = 1.0;
      double beta  = 0.0;
      int inc      = 1;
      int info     = 0;
      //dgemv(&trans, &N, &N, &alpha, overlap, &N, &workd[ipntr[0]-1], &inc, &beta, &workd[ipntr[1]-1], &inc);
      BLAS::gemv_trans(N, N, B.data(), &workd[ipntr[0] - 1], &workd[ipntr[1] - 1]);
    }
  }

  // now get the eigenvalues and eigenvectors
  int rvec = true;
  std::vector<int> select(ncv);
  std::vector<double> dr(Nev + 1);
  std::vector<double> di(Nev + 1);
  std::vector<double> Z(N * (Nev + 1));
  int ldz = N;
  double sigmar;
  double sigmai;
  std::vector<double> workev(3 * N);
  arpack::neupd(rvec, arpack::howmny::schur_vectors, select.data(), dr.data(), di.data(), Z.data(), ldz, sigmar, sigmai,
                workev.data(), arpack::bmat::generalized, N, arpack::which::largest_magnitude, Nev, tol, resid.data(),
                ncv, v.data(), ldv, iparam.data(), ipntr.data(), workd.data(), workl.data(), lworkl, info);


  if (info < 0)
  {
    std::cout << "neupd error, info = " << info << std::endl;
  }


  std::cout << "nconv (iparam(4)) = " << iparam[4] << std::endl;
  for (int i = 0; i < Nev; i++)
  {
    std::cout << "evec i " << i << " " << dr[i] << " imag " << di[i] << std::endl;
    eigenvals[i] = dr[i];
  }

  for (int i = 0; i < Nev; i++)
  {
    for (int j = 0; j < N; j++)
    {
      eigenvectors(i, j) = v.data()[j + N * i];
    }
  }

  return A(0, 0);
}

LinearMethod::Real LinearMethod::getLowestEigenvectorARPACK(Matrix<Real>& A, Matrix<Real>& B, std::vector<Real>& ev)
{
  const int N = A.rows();
  assert(N == ev.size());
  const int Nev = 1;
  Matrix<Real> eigenvectors(N, Nev);
  std::vector<Real> alphar(Nev);
  double zerozero = solveGeneralizedEigenvaluesARPACK(A, B, alphar, eigenvectors);

#if 1
  double ev_target = 2.0;
  std::vector<std::pair<Real, int>> mappedEigenvalues(Nev);
  for (int i = 0; i < Nev; i++)
  {
    Real evi(alphar[i]);
    if ((evi < zerozero) && (evi > (zerozero - 1e2)))
    {
      mappedEigenvalues[i].first  = (evi - zerozero + ev_target) * (evi - zerozero + ev_target);
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
    ev[i] = eigenvectors(i, mappedEigenvalues[0].second) / eigenvectors(0, mappedEigenvalues[0].second);
    if (i < 10)
      app_log() << i << " Raw evec " << eigenvectors(i, mappedEigenvalues[0].second) << " scaled " << ev[i]
                << std::endl;
  }

  double lowest_ev = alphar[mappedEigenvalues[0].second];
  app_log() << "lowest ev = " << alphar[0] << std::endl;
  app_log() << "mapped ev = " << lowest_ev << std::endl;

#else
  double lowest_ev = alphar[0];

  for (int i = 0; i < N; i++)
    ev[i] = eigenvectors(i, 0);
#endif

  return lowest_ev;
}


} // namespace qmcplusplus
