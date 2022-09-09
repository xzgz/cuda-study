# collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(c_plus_utils ${srcs})
