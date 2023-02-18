list(APPEND HCP_BE_SCALARS "totenergy" "-1.481656 0.0005")
list(APPEND HCP_BE_SCALARS "samples" "416000 0.0")
#  LIST(APPEND HCP_BE_SCALARS "flux" "0.0 0.003")

qmc_run_and_check(
  short-hcpBe_1x1x1_pp-vmc_sdj
  "${qmcpack_SOURCE_DIR}/tests/solids/hcpBe_1x1x1_pp"
  qmc_short
  qmc_short.in.xml
  1
  16
  TRUE
  0
  HCP_BE_SCALARS # VMC
)

#
# Long test
#

list(APPEND LONG_HCP_BE_SCALARS "totenergy" "-1.481656 0.000017")
list(APPEND LONG_HCP_BE_SCALARS "samples" "399360000 0.0")
list(APPEND LONG_HCP_BE_SCALARS "flux" "0.0 0.0001")

qmc_run_and_check(
  long-hcpBe_1x1x1_pp-vmc_sdj
  "${qmcpack_SOURCE_DIR}/tests/solids/hcpBe_1x1x1_pp"
  qmc_long
  qmc_long.in.xml
  1
  16
  TRUE
  0
  LONG_HCP_BE_SCALARS # VMC
)

if(NOT QMC_CUDA)
  if(NOT QMC_COMPLEX)

    # Orbital Rotation

    # Two atomic orbitals, no jastrow
    # The Hamiltonian only contains the kinetic energy - no pseudopotential or electron interactions

    # Derivative value can be computed in src/QMCWaveFunctions/tests/eval_bspline_spo.py,
    # using run_qmc_parameter_derivative_hcpBe modified so that local_energy only computes the kinetic energy.

    list(APPEND HCP_BE_NOJ_NOPP_ORB_ROT_PARAM spo_up_orb_rot_0000_0001  0.096  0.01) # scalar name, value, error
    list(APPEND HCP_BE_NOJ_NOPP_ORB_ROT_PARAM spo_down_orb_rot_0000_0001  0.096  0.01) # scalar name, value, error

    qmc_run_and_check_custom_scalar(
      BASE_NAME short-hcpBe_noj_nopp_orb_rot_param_grad
      BASE_DIR "${qmcpack_SOURCE_DIR}/tests/solids/hcpBe_1x1x1_pp"
      PREFIX orb_rot_noj_nopp_param_grad.param
      INPUT_FILE orb_rot_noj_nopp_param_grad.xml
      PROCS 1
      THREADS 16
      SERIES 0
      SCALAR_VALUES HCP_BE_NOJ_NOPP_ORB_ROT_PARAM)

  endif()
endif()
