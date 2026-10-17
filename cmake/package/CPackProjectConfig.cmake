#**************************************************************#
#Edition:
#**  17/10/2026 by Tsukini
#
#File Name:
#**  CPackProjectConfig.cmake
#
#File Description:
#**  Read by CPack for each generator: install prefix of the
#**  Windows packages (MinGW sysroot of Fedora / Debian)
#**************************************************************#

if(CPACK_GENERATOR STREQUAL "RPM" AND CPACK_BRAIN_RPM_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CPACK_BRAIN_RPM_PREFIX}")
elseif(CPACK_GENERATOR STREQUAL "DEB" AND CPACK_BRAIN_DEB_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CPACK_BRAIN_DEB_PREFIX}")
endif()
