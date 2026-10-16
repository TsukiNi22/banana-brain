cmake_minimum_required(VERSION 3.20)
set(TARGET banana-brain)
if(NOT CMAKE_TOOLCHAIN_FILE AND NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER clang++)
endif()
project(${TARGET} VERSION 1.0.0 LANGUAGES CXX)

# =========================
# Options
# =========================
option(BANANA_BRAIN_DEV_PACKAGE "Build development package only" OFF)
option(BANANA_BRAIN_PACKAGE "Build global package" OFF)
option(BANANA_BRAIN_STABLE_RELEASE "Channel as stable release" OFF)
option(BUILD_TESTS "Build unit tests" OFF)

# =========================
# Compilateur & Standard
# =========================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Moderne compilation -fPIE

# =========================
# Requirement
# =========================
set(UTILS_MIN_VERSION 4.1.0) # Minimal libutils version (also used by the packages & the installed config)
find_package(utils ${UTILS_MIN_VERSION} CONFIG REQUIRED) # Check utils dependencies existance
find_package(Python3 REQUIRED) # Used for the exception header generation
find_program(CCACHE_PROGRAM ccache) # Used for cache during compilation
if (CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# =========================
# Warnings
# =========================
add_compile_options(
    -W
    -Wall
    -Wextra
    -Wpedantic
    -Wunused-parameter
    -Wshadow
    -Wuninitialized
)

# =========================
# Sources
# =========================
set(SRC
    ## Brain
    src/brain/NeuralNetwork.cpp

    ## Brain (sub-part: network)
    src/brain/network/ANetwork.cpp
    src/brain/network/LayerNetwork.cpp
    src/brain/network/BrainNetwork.cpp

    ## Brain (sub-part: tools)
    src/brain/tools/dataset.cpp
    src/brain/tools/random.cpp
)
add_library(${TARGET} STATIC ${SRC})
#add_library(${TARGET} SHARED ${SRC})

# =========================
# Special Targets
# =========================
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

file(GLOB_RECURSE EXCEPTION_CONFIG_FILES
    CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/cmake/config/exceptions/*.json"
)
set(GENERATED_EXCEPTION_HEADER
    "${CMAKE_SOURCE_DIR}/include/brain/exception/generated_external_exception_header.hpp"
)
add_custom_command(
    OUTPUT ${GENERATED_EXCEPTION_HEADER}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/scripts/generate_exception_header.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS ${EXCEPTION_CONFIG_FILES} ${CMAKE_SOURCE_DIR}/cmake/scripts/generate_exception_header.py
    COMMENT "Generating exception header (config changed)"
    VERBATIM
)
add_custom_target(generated_external_exception_header
    DEPENDS ${GENERATED_EXCEPTION_HEADER}
)

# =========================
# Release (build Debug & Optimized)
# =========================
add_custom_target(release
    COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR} -DCMAKE_BUILD_TYPE=Debug
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --parallel
    COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR} -DCMAKE_BUILD_TYPE=Asan
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --parallel
    COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR} -DCMAKE_BUILD_TYPE=Optimized
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --parallel
    COMMENT "Building both debug versions Debug & Asan (libbanana-brain_debug.a, libbanana-brain_asan.a) and Optimized (libbanana-brain.a) one"
    VERBATIM
)
add_custom_target(install_release
    COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR} -DCMAKE_BUILD_TYPE=Debug
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target install --parallel
    COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR} -DCMAKE_BUILD_TYPE=Asan
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target install --parallel
    COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR} -DCMAKE_BUILD_TYPE=Optimized
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target install --parallel
    COMMENT "Installing both debug versions Debug & Asan (libbanana-brain_debug.a, libbanana-brain_asan.a) and Optimized (libbanana-brain.a) one"
    VERBATIM
)
# Asan packages: not with MinGW (Windows packages mingw64-libbanana-brain*, built with the MinGW toolchain)
set(PACKAGE_ASAN_COMMANDS "")
if(NOT WIN32)
    list(APPEND PACKAGE_ASAN_COMMANDS
        COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR} -DCMAKE_BUILD_TYPE=Asan -DBANANA_BRAIN_DEV_PACKAGE=OFF -DBANANA_BRAIN_PACKAGE=OFF
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target package --parallel
    )
endif()

add_custom_target(package_release
    COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR} -DCMAKE_BUILD_TYPE=Debug -DBANANA_BRAIN_DEV_PACKAGE=OFF -DBANANA_BRAIN_PACKAGE=OFF
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target package --parallel
    ${PACKAGE_ASAN_COMMANDS}
    COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR} -DCMAKE_BUILD_TYPE=Optimized -DBANANA_BRAIN_DEV_PACKAGE=OFF -DBANANA_BRAIN_PACKAGE=OFF
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target package --parallel
    COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR} -DCMAKE_BUILD_TYPE=Optimized -DBANANA_BRAIN_DEV_PACKAGE=ON -DBANANA_BRAIN_PACKAGE=OFF
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target package --parallel
    COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR} -DCMAKE_BUILD_TYPE=Optimized -DBANANA_BRAIN_DEV_PACKAGE=OFF -DBANANA_BRAIN_PACKAGE=ON
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target package --parallel
    COMMENT "Building the debug versions Debug & Asan (libbanana-brain-db, libbanana-brain-as) and the Optimized one (libbanana-brain-op) with the header (libbanana-brain-dev), mingw64-libbanana-brain* without Asan with the MinGW toolchain"
    VERBATIM
)

# =========================
# Dependencies
# =========================
add_dependencies(${TARGET} generated_external_exception_header)
target_link_libraries(${TARGET} PUBLIC utils::utils) # The public headers include <utils/utils.hpp>

# =========================
# Includes
# =========================
target_include_directories(${TARGET} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/brain/exception> # found by libutils with __has_include
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated/brain> # "version.hpp" of brain.hpp (next to it once installed)
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/brain/exception>
)

# =========================
# Headers configuration
# =========================

# version injection using @VAR@
set(BANANA_BRAIN_VERSION "v${PROJECT_VERSION}")
configure_file(
    ${CMAKE_SOURCE_DIR}/include/brain/version.hpp.in
    ${CMAKE_BINARY_DIR}/generated/brain/version.hpp
    @ONLY
)

# =========================
# Modes
# =========================
set_target_properties(${TARGET} PROPERTIES
    OUTPUT_NAME "banana-brain$<$<CONFIG:Debug>:_debug>$<$<CONFIG:Asan>:_asan>"
)

if(WIN32)
    # MinGW: no AddressSanitizer (Asan = Debug), no PLT (not ELF)
    target_compile_options(${TARGET} PRIVATE
        $<$<CONFIG:Debug,Asan>:-g -ggdb3>
        $<$<CONFIG:Optimized>:-O3 -ffast-math -funroll-loops -pipe -DNDEBUG>
    )
else()
    target_compile_options(${TARGET} PRIVATE
        $<$<CONFIG:Debug>:-g -ggdb3>
        $<$<CONFIG:Asan>:-fsanitize=address -fno-omit-frame-pointer -g -ggdb3>
        $<$<CONFIG:Optimized>:-O3 -ffast-math -funroll-loops -fno-plt -pipe -DNDEBUG>
    )
    target_link_options(${TARGET} PRIVATE
        $<$<CONFIG:Asan>:-fsanitize=address>
    )
endif()

# =========================
# Installation
# =========================
if(NOT BANANA_BRAIN_PACKAGE)
include(GNUInstallDirs) # load variable used for installation
include(CMakePackageConfigHelpers)

# licenses
install(
    FILES ${CMAKE_SOURCE_DIR}/LICENSE.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/${TARGET}
    COMPONENT headers
)

# library
install(TARGETS ${TARGET}
    EXPORT ${TARGET}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT lib
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# header (the generated exception header included: the library exposes its own codes)
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT headers
    PATTERN "*.in" EXCLUDE
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/generated/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT headers
)

# CMake setup for find_package(banana-brain CONFIG REQUIRED)
install(EXPORT ${TARGET}Targets
    FILE ${TARGET}Targets.cmake
    NAMESPACE ${TARGET}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${TARGET}
)

# Per mode export files (one headers-<mode> component for each library package), depends on the 'CMakeFiles/Export' path
set(EXPORT_PATH "CMakeFiles/Export")
if(BANANA_BRAIN_DEV_PACKAGE)
    # main-headers
    file(GLOB_RECURSE BANANA_BRAIN_EXPORT_FILES "${CMAKE_BINARY_DIR}/${EXPORT_PATH}/*/${TARGET}Targets.cmake")
    install(
        FILES ${BANANA_BRAIN_EXPORT_FILES}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${TARGET}
        COMPONENT headers
    )

    # sub-headers
    file(GLOB_RECURSE BANANA_BRAIN_EXPORT_FILES "${CMAKE_BINARY_DIR}/${EXPORT_PATH}/*/${TARGET}Targets-debug.cmake")
    install(
        FILES ${BANANA_BRAIN_EXPORT_FILES}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${TARGET}
        COMPONENT headers-db
    )
    file(GLOB_RECURSE BANANA_BRAIN_EXPORT_FILES "${CMAKE_BINARY_DIR}/${EXPORT_PATH}/*/${TARGET}Targets-asan.cmake")
    install(
        FILES ${BANANA_BRAIN_EXPORT_FILES}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${TARGET}
        COMPONENT headers-as
    )
    file(GLOB_RECURSE BANANA_BRAIN_EXPORT_FILES
        "${CMAKE_BINARY_DIR}/${EXPORT_PATH}/*/${TARGET}Targets-optimized.cmake"
        "${CMAKE_BINARY_DIR}/${EXPORT_PATH}/*/${TARGET}Targets-noconfig.cmake" # No config select optimized by default
    )
    install(
        FILES ${BANANA_BRAIN_EXPORT_FILES}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${TARGET}
        COMPONENT headers-op
    )
endif()

set(CONFIG_INSTALL_DIR
    ${CMAKE_INSTALL_LIBDIR}/cmake/${TARGET}
)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/package/${TARGET}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}Config.cmake
    INSTALL_DESTINATION ${CONFIG_INSTALL_DIR}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}ConfigVersion.cmake
    DESTINATION ${CONFIG_INSTALL_DIR}
    COMPONENT headers
)

endif()
# =========================
# Utils
# =========================
add_custom_target(get_unregistered_files
    COMMAND bash ${CMAKE_SOURCE_DIR}/cmake/scripts/check_files.sh unregistered ${SRC}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# =========================
# Package Building
# =========================
set(CPACK_PACKAGE_VENDOR "TsukiNi22")
set(CPACK_PACKAGE_CONTACT "xartania.contact@gmail.com")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.md")

set(CPACK_GENERATOR "RPM;DEB")

# Windows (cross compiled): mingw64-libbanana-brain*, installed in the MinGW sysroot (found by the MinGW toolchain)
if(WIN32)
    set(PACKAGE_BASE "mingw64-libbanana-brain")
    set(UTILS_PACKAGE_BASE "mingw64-libutils")
    set(CPACK_PROJECT_CONFIG_FILE "${CMAKE_SOURCE_DIR}/cmake/package/CPackProjectConfig.cmake") # prefix per generator
    set(CPACK_BANANA_BRAIN_RPM_PREFIX "/usr/x86_64-w64-mingw32/sys-root/mingw") # Fedora (mingw64-filesystem)
    set(CPACK_BANANA_BRAIN_DEB_PREFIX "/usr/x86_64-w64-mingw32") # Debian / Ubuntu (mingw-w64)
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "noarch") # Windows binaries: no host architecture
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "all")
    set(CPACK_RPM_PACKAGE_AUTOREQPROV OFF)
    set(CPACK_RPM_SPEC_MORE_DEFINE "%define __os_install_post %{nil}\n%define debug_package %{nil}") # no host strip / debuginfo
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
        /usr/x86_64-w64-mingw32
        /usr/x86_64-w64-mingw32/sys-root
        /usr/x86_64-w64-mingw32/sys-root/mingw
        /usr/x86_64-w64-mingw32/sys-root/mingw/include
        /usr/x86_64-w64-mingw32/sys-root/mingw/lib
        /usr/x86_64-w64-mingw32/sys-root/mingw/lib/cmake
        /usr/x86_64-w64-mingw32/sys-root/mingw/share
        /usr/x86_64-w64-mingw32/sys-root/mingw/share/licenses
    )
else()
    set(PACKAGE_BASE "libbanana-brain")
    set(UTILS_PACKAGE_BASE "libutils")
endif()

# Setup the channel name
if(BANANA_BRAIN_STABLE_RELEASE)
    set(CHANNEL "")
    set(MIRROR_CHANNEL "-pre")
else()
    set(CHANNEL "-pre")
    set(MIRROR_CHANNEL "")
endif()

# Setup the package name
if(BANANA_BRAIN_PACKAGE)
    set(CPACK_PACKAGE_NAME "${PACKAGE_BASE}${CHANNEL}")
    set(MIRROR_PACKAGE_NAME "${PACKAGE_BASE}${MIRROR_CHANNEL}")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "C++20 neural network library (global)")
elseif(BANANA_BRAIN_DEV_PACKAGE)
    set(CPACK_PACKAGE_NAME "${PACKAGE_BASE}-dev${CHANNEL}")
    set(MIRROR_PACKAGE_NAME "${PACKAGE_BASE}-dev${MIRROR_CHANNEL}")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "C++20 neural network library (headers/cmake)")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CPACK_PACKAGE_NAME "${PACKAGE_BASE}-db${CHANNEL}")
    set(MIRROR_PACKAGE_NAME "${PACKAGE_BASE}-db${MIRROR_CHANNEL}")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "C++20 neural network library (debug)")
    set(PREFIX "db")
elseif(CMAKE_BUILD_TYPE STREQUAL "Asan")
    set(CPACK_PACKAGE_NAME "${PACKAGE_BASE}-as${CHANNEL}")
    set(MIRROR_PACKAGE_NAME "${PACKAGE_BASE}-as${MIRROR_CHANNEL}")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "C++20 neural network library (asan)")
    set(PREFIX "as")
else()
    set(CPACK_PACKAGE_NAME "${PACKAGE_BASE}-op${CHANNEL}")
    set(MIRROR_PACKAGE_NAME "${PACKAGE_BASE}-op${MIRROR_CHANNEL}")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "C++20 neural network library (optimized)")
    set(PREFIX "op")
endif()

# Setup the conflict between stable & unstable packages
if(BANANA_BRAIN_STABLE_RELEASE)
    # If a recent stable version exist the libbanana-brain-*-pre installation will result in the stable version
    set(CPACK_RPM_PACKAGE_OBSOLETES "${MIRROR_PACKAGE_NAME} <= ${PROJECT_VERSION}")
    set(CPACK_DEBIAN_PACKAGE_REPLACES "${MIRROR_PACKAGE_NAME} (<= ${PROJECT_VERSION})")
endif()

set(CPACK_RPM_PACKAGE_CONFLICTS "${MIRROR_PACKAGE_NAME}")
set(CPACK_DEBIAN_PACKAGE_BREAKS "${MIRROR_PACKAGE_NAME}")
set(CPACK_DEBIAN_PACKAGE_CONFLICTS "${MIRROR_PACKAGE_NAME}")

# Separate headers & lib
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)
if(BANANA_BRAIN_PACKAGE)
    set(CPACK_COMPONENTS_ALL "")
    set(CPACK_COMPONENT_INCLUDE_TOPLEVEL_DIRECTORY OFF) # In case~
elseif(BANANA_BRAIN_DEV_PACKAGE)
    if(WIN32)
        set(CPACK_COMPONENTS_ALL headers headers-db headers-op) # no Asan with MinGW
    else()
        set(CPACK_COMPONENTS_ALL headers headers-db headers-as headers-op)
    endif()

    set(CPACK_RPM_HEADERS-DB_PACKAGE_NAME "${PACKAGE_BASE}-dev-db${CHANNEL}")
    set(CPACK_RPM_HEADERS-AS_PACKAGE_NAME "${PACKAGE_BASE}-dev-as${CHANNEL}")
    set(CPACK_RPM_HEADERS-OP_PACKAGE_NAME "${PACKAGE_BASE}-dev-op${CHANNEL}")

    set(CPACK_DEBIAN_HEADERS-DB_PACKAGE_NAME "${PACKAGE_BASE}-dev-db${CHANNEL}")
    set(CPACK_DEBIAN_HEADERS-AS_PACKAGE_NAME "${PACKAGE_BASE}-dev-as${CHANNEL}")
    set(CPACK_DEBIAN_HEADERS-OP_PACKAGE_NAME "${PACKAGE_BASE}-dev-op${CHANNEL}")

    # Setup the conflict between stable & unstable packages
    if(BANANA_BRAIN_STABLE_RELEASE)
        set(CPACK_RPM_HEADERS-DB_PACKAGE_OBSOLETES "${PACKAGE_BASE}-dev-db${MIRROR_CHANNEL} <= ${PROJECT_VERSION}")
        set(CPACK_RPM_HEADERS-AS_PACKAGE_OBSOLETES "${PACKAGE_BASE}-dev-as${MIRROR_CHANNEL} <= ${PROJECT_VERSION}")
        set(CPACK_RPM_HEADERS-OP_PACKAGE_OBSOLETES "${PACKAGE_BASE}-dev-op${MIRROR_CHANNEL} <= ${PROJECT_VERSION}")

        set(CPACK_DEBIAN_HEADERS-DB_PACKAGE_REPLACES "${PACKAGE_BASE}-dev-db${MIRROR_CHANNEL} (<= ${PROJECT_VERSION})")
        set(CPACK_DEBIAN_HEADERS-AS_PACKAGE_REPLACES "${PACKAGE_BASE}-dev-as${MIRROR_CHANNEL} (<= ${PROJECT_VERSION})")
        set(CPACK_DEBIAN_HEADERS-OP_PACKAGE_REPLACES "${PACKAGE_BASE}-dev-op${MIRROR_CHANNEL} (<= ${PROJECT_VERSION})")
    endif()

    set(CPACK_RPM_HEADERS-DB_PACKAGE_CONFLICTS "${PACKAGE_BASE}-dev-db${MIRROR_CHANNEL}")
    set(CPACK_RPM_HEADERS-AS_PACKAGE_CONFLICTS "${PACKAGE_BASE}-dev-as${MIRROR_CHANNEL}")
    set(CPACK_RPM_HEADERS-OP_PACKAGE_CONFLICTS "${PACKAGE_BASE}-dev-op${MIRROR_CHANNEL}")

    set(CPACK_DEBIAN_HEADERS-DB_PACKAGE_BREAKS "${PACKAGE_BASE}-dev-db${MIRROR_CHANNEL}")
    set(CPACK_DEBIAN_HEADERS-DB_PACKAGE_CONFLICTS "${PACKAGE_BASE}-dev-db${MIRROR_CHANNEL}")
    set(CPACK_DEBIAN_HEADERS-AS_PACKAGE_BREAKS "${PACKAGE_BASE}-dev-as${MIRROR_CHANNEL}")
    set(CPACK_DEBIAN_HEADERS-AS_PACKAGE_CONFLICTS "${PACKAGE_BASE}-dev-as${MIRROR_CHANNEL}")
    set(CPACK_DEBIAN_HEADERS-OP_PACKAGE_BREAKS "${PACKAGE_BASE}-dev-op${MIRROR_CHANNEL}")
    set(CPACK_DEBIAN_HEADERS-OP_PACKAGE_CONFLICTS "${PACKAGE_BASE}-dev-op${MIRROR_CHANNEL}")
else()
    set(CPACK_COMPONENTS_ALL lib)
endif()

# Edit release value (run count)
if(BANANA_BRAIN_STABLE_RELEASE)
    # A stable version should never have to be rebuild and should always be removed or marked as deprecated
    set(CPACK_RPM_PACKAGE_RELEASE "1")
    set(CPACK_DEBIAN_PACKAGE_RELEASE "1")
elseif(DEFINED PACKAGE_RELEASE)
    set(CPACK_RPM_PACKAGE_RELEASE "${PACKAGE_RELEASE}")
    set(CPACK_DEBIAN_PACKAGE_RELEASE "${PACKAGE_RELEASE}")
else() # 0 is impossible for github.run_number
    set(CPACK_RPM_PACKAGE_RELEASE "0")
    set(CPACK_DEBIAN_PACKAGE_RELEASE "0")
endif()

# Dependencies (libutils of the same mode: the static library doesn't embed it)
if(BANANA_BRAIN_PACKAGE)
    if(BANANA_BRAIN_STABLE_RELEASE)
        set(CPACK_RPM_PACKAGE_REQUIRES "${PACKAGE_BASE}-op${CHANNEL} >= ${PROJECT_VERSION}")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "${PACKAGE_BASE}-op${CHANNEL} (>= ${PROJECT_VERSION})")
    else()
        set(CPACK_RPM_PACKAGE_REQUIRES "${PACKAGE_BASE}-op${CHANNEL} = ${PROJECT_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "${PACKAGE_BASE}-op${CHANNEL} (= ${PROJECT_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE})")
    endif()
elseif(BANANA_BRAIN_DEV_PACKAGE)
    set(CPACK_RPM_PACKAGE_REQUIRES "${UTILS_PACKAGE_BASE}-dev >= ${UTILS_MIN_VERSION}") # public headers include <utils/utils.hpp>
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "${UTILS_PACKAGE_BASE}-dev (>= ${UTILS_MIN_VERSION})")
else()
    if(BANANA_BRAIN_STABLE_RELEASE)
        set(CPACK_RPM_PACKAGE_REQUIRES "${PACKAGE_BASE}-dev${CHANNEL} >= ${PROJECT_VERSION}, ${PACKAGE_BASE}-dev-${PREFIX}${CHANNEL} >= ${PROJECT_VERSION}, ${UTILS_PACKAGE_BASE}-${PREFIX} >= ${UTILS_MIN_VERSION}")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "${PACKAGE_BASE}-dev${CHANNEL} (>= ${PROJECT_VERSION}), ${PACKAGE_BASE}-dev-${PREFIX}${CHANNEL} (>= ${PROJECT_VERSION}), ${UTILS_PACKAGE_BASE}-${PREFIX} (>= ${UTILS_MIN_VERSION})")
    else()
        set(CPACK_RPM_PACKAGE_REQUIRES "${PACKAGE_BASE}-dev${CHANNEL} = ${PROJECT_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}, ${PACKAGE_BASE}-dev-${PREFIX}${CHANNEL} = ${PROJECT_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}, ${UTILS_PACKAGE_BASE}-${PREFIX} >= ${UTILS_MIN_VERSION}")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "${PACKAGE_BASE}-dev${CHANNEL} (= ${PROJECT_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE}), ${PACKAGE_BASE}-dev-${PREFIX}${CHANNEL} (= ${PROJECT_VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE}), ${UTILS_PACKAGE_BASE}-${PREFIX} (>= ${UTILS_MIN_VERSION})")
    endif()

    # Suggests the installation of gdb along side the debug version
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT WIN32)
        set(CPACK_RPM_PACKAGE_SUGGESTS "gdb")
        set(CPACK_DEBIAN_PACKAGE_SUGGESTS "gdb")
    endif()
endif()

# RPM
set(CPACK_RPM_PACKAGE_LICENSE "CC-BY-NC-SA-4.0")
set(CPACK_RPM_LIB_PACKAGE_NAME ${CPACK_PACKAGE_NAME})
set(CPACK_RPM_HEADERS_PACKAGE_NAME ${CPACK_PACKAGE_NAME})

# Debian
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "TsukiNi22")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/TsukiNi22/banana-brain")
set(CPACK_DEBIAN_LIB_PACKAGE_NAME ${CPACK_PACKAGE_NAME})
set(CPACK_DEBIAN_HEADERS_PACKAGE_NAME ${CPACK_PACKAGE_NAME})

include(CPack)
