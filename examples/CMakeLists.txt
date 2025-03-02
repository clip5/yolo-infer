
function(add_exe name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} yoloinfer ${THIRD_LIBS})
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/src/common ${PROJECT_SOURCE_DIR}/src/)
    install(TARGETS ${name})
endfunction()

add_exe(yolo_example)