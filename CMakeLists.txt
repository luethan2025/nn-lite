cmake_minimum_required(VERSION 3.10)
project(nn-lite LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "You're using ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  message(STATUS "You're using ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ##############################################################################
# COMPILER SETUP
# ##############################################################################
set(FOCUS_SRC_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src/include)

include_directories(${FOCUS_SRC_INCLUDE_DIR})
include_directories(BEFORE src)

# ##############################################################################
# DEPENDENCIES
# ##############################################################################
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ##############################################################################
# CMAKE MODULES
# ##############################################################################
add_subdirectory(src)
add_subdirectory(test)

# ##############################################################################
# MAKE TARGETS
# ##############################################################################

# #########################################
# "make format"
# #########################################
file(GLOB_RECURSE NN_LITE_FORMAT_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/test/*.h"
)

add_custom_target(
  format
  COMMAND clang-format -i ${NN_LITE_FORMAT_FILES}
)
