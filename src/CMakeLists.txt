add_subdirectory(type)

add_library(nn-lite STATIC ${ALL_OBJECT_FILES})

set(FOCUS_LIBS
        focus_type
        )

target_link_libraries(
        nn-lite
        ${FOCUS_LIBS}
        )

target_include_directories(
        nn-lite PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
