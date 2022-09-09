get_filename_component(directory_name ${CMAKE_CURRENT_SOURCE_DIR} NAME_WE)

set(executable_name ${directory_name}_matrix_mul_main)
set_source_files_properties(matrix_mul_main.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
cuda_add_executable(${executable_name} matrix_mul_main.cpp)
target_link_libraries(${executable_name} ${PROJECT_LINK_LIBS} ${CUDA_LINK_LIBS})
set_target_properties(${executable_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${directory_name})
set_target_properties(${executable_name} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS ${executable_name} DESTINATION ${PROJECT_BINARY_DIR}/install/bin)
