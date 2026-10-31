cmake_minimum_required(VERSION 3.16)
project(pydeltarcm_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(pyd_core STATIC
  src/rng.c
  src/npy.c
  src/json.c
  src/grid.c
  src/model.c
)
target_include_directories(pyd_core PUBLIC src)
# strict level-3 optimization; -ffp-contract=off keeps FMA-free arithmetic so
# the port matches the reference's operation order exactly (spec: no FMA
# refactors). Compile-time warnings stay enabled.
target_compile_options(pyd_core PRIVATE -O3 -ffp-contract=off -Wall -Wextra)

add_executable(pydeltarcm_c src/main.c)
target_link_libraries(pydeltarcm_c PRIVATE pyd_core m)

enable_testing()
add_executable(test_rng tests/test_rng.c)
target_link_libraries(test_rng PRIVATE pyd_core m)
add_test(NAME rng COMMAND test_rng)

add_executable(test_fixture tests/test_fixture.c)
target_link_libraries(test_fixture PRIVATE pyd_core m)
add_test(NAME fixture COMMAND test_fixture
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
