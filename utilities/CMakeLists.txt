# collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(CUDA_LINK_LIBS "")
list(APPEND LINK_LIBS_CUDA -lcuda -lcudart -lcublas -lcurand)
find_package(CUDA)

# build each source file independently
foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)
    cuda_add_executable(${name} ${source} OPTIONS -arch sm_61)
    target_link_libraries(${name} ${PROJECT_LINK_LIBS} ${LINK_LIBS_CUDA})
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/utilities)
    set_target_properties(${name} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
    install(TARGETS ${name} DESTINATION ${PROJECT_BINARY_DIR}/install/bin)
endforeach(source)
