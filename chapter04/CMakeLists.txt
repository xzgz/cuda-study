# collect source files
file(GLOB_RECURSE src_cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE src ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

set(CUDA_LINK_LIBS "")
list(APPEND LINK_LIBS_CUDA -lcuda -lcudart -lcublas -lcurand)
find_package(CUDA)

get_filename_component(directory_name ${CMAKE_CURRENT_SOURCE_DIR} NAME_WE)
# build each source file independently
foreach(source ${src})
    get_filename_component(filename ${source} NAME_WE)
    set(executable_name ${directory_name}_${filename})
    cuda_add_executable(${executable_name} ${source} OPTIONS -arch sm_21)
    target_link_libraries(${executable_name} ${PROJECT_LINK_LIBS} ${LINK_LIBS_CUDA})
    set_target_properties(${executable_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${directory_name})
    set_target_properties(${executable_name} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
    install(TARGETS ${executable_name} DESTINATION ${PROJECT_BINARY_DIR}/install/bin)
endforeach(source)

add_executable(${directory_name} ${src_cpp})
