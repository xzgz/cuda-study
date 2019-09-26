cmake_minimum_required(VERSION 2.8)

project(cuda-study C CXX)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    message("Have set c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    message("Have set c++0x.")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")

foreach(val ${CMAKE_CXX_FLAGS})
    message(${val})
endforeach()

find_package(Threads REQUIRED)
message("Threads libs:")
foreach(val ${CMAKE_THREAD_LIBS_INIT})
    message(${val})
endforeach()
message("\n")
set(PROJECT_LINK_LIBS "")
list(APPEND PROJECT_LINK_LIBS ${CMAKE_THREAD_LIBS_INIT})

list(APPEND INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
list(APPEND INCLUDE_DIRS /usr/local/cuda/include)
list(APPEND LINK_DIRS /usr/local/cuda/lib64)

include_directories(${INCLUDE_DIRS})
link_directories(${LINK_DIRS})

add_subdirectory(sse-study)
add_subdirectory(utilities)
add_subdirectory(simple)
