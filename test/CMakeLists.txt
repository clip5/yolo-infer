include_directories(${PROJECT_SOURCE_DIR}/src/core ${PROJECT_SOURCE_DIR}/src/common
                ${PROJECT_SOURCE_DIR}/src/
                ${PROJECT_SOURCE_DIR}/src/cv
                ${PROJECT_SOURCE_DIR}/src/nn/preprocess
                ${PROJECT_SOURCE_DIR}/src/nn/postprocess
                ${PROJECT_SOURCE_DIR}/src/nn/backend)

function(add_exe name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} yoloinfer ${THIRD_LIBS})
    install(TARGETS ${name})
endfunction()

add_exe(testimage)
add_exe(testwarp)
add_exe(testforward)