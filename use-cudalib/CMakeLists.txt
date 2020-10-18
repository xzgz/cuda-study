# collect source files
file(GLOB_RECURSE src_cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

get_filename_component(directory_name ${CMAKE_CURRENT_SOURCE_DIR} NAME_WE)
# build each source file independently
foreach(source ${src_cpp})
    get_filename_component(filename ${source} NAME_WE)
    set(executable_name ${directory_name}_${filename})
    add_executable(${executable_name} ${source})
    target_link_libraries(${executable_name} cuda_utils ${PROJECT_LINK_LIBS})
    set_target_properties(${executable_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${directory_name})
    set_target_properties(${executable_name} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
    install(TARGETS ${executable_name} DESTINATION ${PROJECT_BINARY_DIR}/install/bin)
endforeach(source)
