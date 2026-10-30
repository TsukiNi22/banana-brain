cmake_minimum_required(VERSION 3.20)
set(TARGET lab_server)
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER clang++)
endif()
project(${TARGET} LANGUAGES CXX)

# =========================
# Compilateur & Standard
# =========================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Optimized) # the variant of libutils follows it (as the library)
endif()

# =========================
# Requirement
# =========================
# The library is built on its own (lab/server.sh): its CMake uses CMAKE_SOURCE_DIR, it can't be a subdirectory
set(BRAIN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. CACHE PATH "Root of banana-brain")
set(BRAIN_BUILD ${CMAKE_CURRENT_SOURCE_DIR}/build/lib CACHE PATH "Build folder of the library")
find_package(utils 4.1.0 CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)
find_library(BRAIN_LIBRARY NAMES brain PATHS ${BRAIN_BUILD} NO_DEFAULT_PATH REQUIRED)

# =========================
# Sources
# =========================
set(SRC
    # Entry point
    src/main.cpp

    # Server
    src/LabServer.cpp
    src/BrainSession.cpp
)

# =========================
# Target
# =========================
add_executable(${TARGET} ${SRC})
target_include_directories(${TARGET} PRIVATE
    include
    ${BRAIN_ROOT}/include
    ${BRAIN_ROOT}/include/brain/exception # found by libutils with __has_include
    ${BRAIN_BUILD}/generated/brain
)
target_link_libraries(${TARGET} PRIVATE ${BRAIN_LIBRARY} utils::utils nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
