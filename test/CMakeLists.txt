cmake_minimum_required(VERSION 3.10)

include(GoogleTest)

file(GLOB_RECURSE FOCUS_TEST_SOURCES "${PROJECT_SOURCE_DIR}/test/*/*test.cpp")

# ##############################################################################
# MAKE TARGETS
# ##############################################################################

foreach (focus_test_source ${FOCUS_TEST_SOURCES})
  get_filename_component(focus_test_filename ${focus_test_source} NAME)
  string(REPLACE ".cpp" "" focus_test_name ${focus_test_filename})

  add_executable(${focus_test_name} EXCLUDE_FROM_ALL ${focus_test_source})
  gtest_discover_tests(${focus_test_name}
            EXTRA_ARGS
            --gtest_output=xml:${CMAKE_BINARY_DIR}/test/${focus_test_name}.xml
            --gtest_catch_exceptions=0
            DISCOVERY_TIMEOUT 120
            PROPERTIES
            TIMEOUT 120
            )
  
  target_link_libraries(${focus_test_name} nn-lite gtest gmock_main)

  # Set test target properties and dependencies.
  set_target_properties(${focus_test_name}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
            COMMAND ${focus_test_name}
            )

endforeach ()
