cmake_minimum_required(VERSION 4.0)
project(simd_hashmap)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(simd_hashmap INTERFACE hashmap.hpp)
target_include_directories(simd_hashmap INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(simd_hashmap INTERFACE -mavx2)
endif()

enable_testing()

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


add_executable(hashmap_tests hashmap_tests.cpp)
target_link_libraries(hashmap_tests GTest::gtest_main simd_hashmap)

include(GoogleTest)
gtest_discover_tests(hashmap_tests)

FetchContent_Declare(
        googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.zip
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(hashmap_bench hashmap_bench.cpp)
target_link_libraries(hashmap_bench benchmark::benchmark simd_hashmap)