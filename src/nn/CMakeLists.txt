list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/backend/tensorrt/engine_trt.cpp)
list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/preprocess/preprocess.cpp)
list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/postprocess/postprocess.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/backend
                    ${CMAKE_CURRENT_SOURCE_DIR}/preprocess
                    ${CMAKE_CURRENT_SOURCE_DIR}/postprocess)

set(SOURCES ${SOURCES} PARENT_SCOPE)
