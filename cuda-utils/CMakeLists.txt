# collect source files
file(GLOB_RECURSE src ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

set(CUDA_LINK_LIBS "")
list(APPEND LINK_LIBS_CUDA -lcuda -lcudart -lcublas -lcurand)
find_package(CUDA)

get_filename_component(directory_name ${CMAKE_CURRENT_SOURCE_DIR} NAME_WE)
#cuda_add_library(cuda_utils STATIC ${src})
cuda_add_library(cuda_utils SHARED ${src})
target_link_libraries(cuda_utils ${LINK_LIBS_CUDA})
set_target_properties(cuda_utils PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${directory_name})
set_target_properties(cuda_utils PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS cuda_utils DESTINATION ${PROJECT_BINARY_DIR}/install/lib)
