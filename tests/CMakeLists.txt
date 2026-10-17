set(TARGET unit_tests)

# =========================
# Compilateur & Standard
# =========================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}) # TARGET destination

# =========================
# Requirement
# =========================
if(CMAKE_CROSSCOMPILING)
    find_package(GTest CONFIG QUIET) # built for the target in the sysroot (ex: mingw64-unit-tests image)
endif()
if(CMAKE_CROSSCOMPILING AND NOT GTest_FOUND)
    # No GTest package for the target (ex: MinGW): built from the sources
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP ON
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
elseif(NOT CMAKE_CROSSCOMPILING)
    find_package(GTest REQUIRED)
endif()

# =========================
# Sources
# =========================
# Project sources: the library target when it exists, else the shared list of the parent (never its entry point)
set(SRC_PROJECT "")
if(NOT BANANA_BRAIN_PACKAGE_LIB)
    set(SRC_PROJECT ${SRC_SHARED})
    list(TRANSFORM SRC_PROJECT PREPEND ${CMAKE_SOURCE_DIR}/)
endif()

set(SRC
    # Brain
    NetworkType.cpp
    NeuralNetwork.cpp

    # Brain (sub-part: network)
    network/LayerNetwork.cpp
    network/BrainNetwork.cpp

    # Brain (sub-part: tools)
    tools/Random.cpp
    tools/RandomScope.cpp
    tools/Dataset.cpp
    tools/Neuron.cpp
)
add_executable(${TARGET} ${SRC} ${SRC_PROJECT})

# =========================
# Dependencies
# =========================
target_include_directories(${TARGET} PRIVATE ../include/ ../include/brain/exception/ .)
target_link_libraries(${TARGET} PRIVATE utils::utils GTest::gtest GTest::gtest_main)
if(BANANA_BRAIN_PACKAGE_LIB)
    target_link_libraries(${TARGET} PRIVATE ${TARGET_LIB})
endif()
add_dependencies(${TARGET} generated_external_exception_header)
if(WIN32)
    target_link_options(${TARGET} PRIVATE -static-libgcc -static-libstdc++ -static) # standalone .exe (no MinGW runtime dll)
endif()

# =========================
# Modes
# =========================
target_compile_options(${TARGET} PRIVATE -g -ggdb3)
if(NOT WIN32)
    # libutils_asan.a needs the AddressSanitizer runtime at the link
    target_compile_options(${TARGET} PRIVATE $<$<CONFIG:Asan>:-fsanitize=address -fno-omit-frame-pointer>)
    target_link_options(${TARGET} PRIVATE $<$<CONFIG:Asan>:-fsanitize=address>)
endif()

include(GoogleTest)
if(CMAKE_CROSSCOMPILING)
    # Listed when ctest runs (CMAKE_CROSSCOMPILING_EMULATOR, ex: wine), not at the build
    gtest_discover_tests(${TARGET}
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 120 # first start of wine
    )
else()
    gtest_discover_tests(${TARGET})
endif()
