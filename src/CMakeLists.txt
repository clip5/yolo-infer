include_directories(
    ${TENSORRT_DIR}/include 
    ${CUDA_TOOLKIT_ROOT_DIR}/include)

include_directories( 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/cv
    ${CMAKE_CURRENT_SOURCE_DIR}/tasks)

include_directories( 
    ${CMAKE_CURRENT_SOURCE_DIR}/nn/backend/
    ${CMAKE_CURRENT_SOURCE_DIR}/nn/preprocess/
    ${CMAKE_CURRENT_SOURCE_DIR}/nn/postprocess/)
    
set(SOURCES "")
list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/yolo_infer.cpp)

add_subdirectory(core)
add_subdirectory(cv)
add_subdirectory(nn)
add_subdirectory(tasks)


cuda_add_library(yoloinfer SHARED ${SOURCES})

target_link_libraries(yoloinfer ${THIRD_LIBS})
install(TARGETS yoloinfer
        DESTINATION lib
        COMPONENT library)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/yolo_infer.hpp ${CMAKE_CURRENT_SOURCE_DIR}/common/yi_type.hpp
        DESTINATION include/)