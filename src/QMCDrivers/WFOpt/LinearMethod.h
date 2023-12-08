//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


/** @file LinearMethod.h
 * @brief LinearMethod related functions.
 */
#ifndef QMCPLUSPLUS_LINEARMETHOD_H
#define QMCPLUSPLUS_LINEARMETHOD_H

#include <Configuration.h>
#include <NewTimer.h>
#include <OhmmsPETE/OhmmsMatrix.h>


namespace qmcplusplus
{
///forward declaration of a cost function
class QMCCostFunctionBase;

class LinearMethod
{
  using Real = QMCTraits::RealType;

  // obtain the range of non-linear parameters
  void getNonLinearRange(int& first, int& last, const QMCCostFunctionBase& optTarget) const;

  bool magma_initialized_ = false;
  int* magma_queue_;

public:
  void solveGeneralizedEigenvalues(Matrix<Real>& A,
                                   Matrix<Real>& B,
                                   std::vector<Real>& eigenvals,
                                   Matrix<Real>& eigenvectors) const;
  Real solveGeneralizedEigenvalues_Inv(Matrix<Real>& A,
                                       Matrix<Real>& B,
                                       std::vector<Real>& eigenvals,
                                       Matrix<Real>& eigenvectors) const;
  Real solveGeneralizedEigenvaluesMagma(Matrix<Real>& A,
                                        Matrix<Real>& B,
                                        std::vector<Real>& eigenvals,
                                        Matrix<Real>& eigenvectors);
#ifdef QMC_USE_ARPACK
  Real solveGeneralizedEigenvaluesARPACK(Matrix<Real>& A,
                                         Matrix<Real>& B,
                                         std::vector<Real>& eigenvals,
                                         Matrix<Real>& eigenvectors);
  Real solveGeneralizedEigenvaluesARPACK_mode2(Matrix<Real>& A,
                                               Matrix<Real>& B,
                                               std::vector<Real>& eigenvals,
                                               Matrix<Real>& eigenvectors);
  Real getLowestEigenvectorARPACK(Matrix<Real>& A, Matrix<Real>& B, std::vector<Real>& ev);
#endif
  //asymmetric generalized EV
  Real getLowestEigenvector(Matrix<Real>& A, Matrix<Real>& B, std::vector<Real>& ev) const;
  Real getLowestEigenvector_Inv(Matrix<Real>& A, Matrix<Real>& B, std::vector<Real>& ev, double ev_target) const;
  //asymmetric EV
  Real getLowestEigenvector(Matrix<Real>& A, std::vector<Real>& ev) const;


  Real getLowestEigenvectorMagma(Matrix<Real>& A, Matrix<Real>& B, std::vector<Real>& ev);
  // compute a rescale factor. Ye: Where is the method from?
  Real getNonLinearRescale(std::vector<Real>& dP, Matrix<Real>& S, const QMCCostFunctionBase& optTarget) const;

  virtual ~LinearMethod();
};
} // namespace qmcplusplus
#endif
