# collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE srcs_asm ${CMAKE_CURRENT_SOURCE_DIR}/*.S)
list(APPEND srcs ${srcs_asm})

enable_language(ASM)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
#cmake_policy(SET CMP0037 OLD)

# build each source file independently
foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    target_link_libraries(${name} ${PROJECT_LINK_LIBS})
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/sse-study)
    set_target_properties(${name} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
    install(TARGETS ${name} DESTINATION ${PROJECT_BINARY_DIR}/install/bin)
endforeach(source)
