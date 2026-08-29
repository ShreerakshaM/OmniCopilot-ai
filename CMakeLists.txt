# Copyright 2024 OmniCopilot Authors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)

project(omnicopilot
  VERSION 0.1.0
  LANGUAGES CXX
  DESCRIPTION "Cooperative Multimodal AI Under Communication Constraints"
)

# ─── Global Settings ─────────────────────────────────────────────────────────

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # For clang-tidy / IDE integration

# Build type defaults to RelWithDebInfo if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Options
option(OMNICOPILOT_BUILD_TESTS "Build unit tests" ON)
option(OMNICOPILOT_BUILD_BINDINGS "Build Python bindings (pybind11)" ON)
option(OMNICOPILOT_ENABLE_SANITIZERS "Enable ASan/UBSan in Debug builds" OFF)
option(OMNICOPILOT_ENABLE_CLANG_TIDY "Run clang-tidy during build" OFF)

# ─── Compiler Warnings ───────────────────────────────────────────────────────

add_compile_options(
  -Wall
  -Wextra
  -Wpedantic
  -Werror
  -Wno-unused-parameter  # Allow unused params in interface stubs during scaffolding
  -Wshadow
  -Wnon-virtual-dtor
  -Woverloaded-virtual
  -Wcast-align
  -Wformat=2
)

# Sanitizers for debug builds
if(OMNICOPILOT_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# clang-tidy integration
if(OMNICOPILOT_ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE})
  else()
    message(WARNING "clang-tidy not found, static analysis disabled")
  endif()
endif()

# ─── Dependencies ────────────────────────────────────────────────────────────

# Protobuf
find_package(Protobuf REQUIRED)

# pybind11 (optional — for Python bindings)
if(OMNICOPILOT_BUILD_BINDINGS)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    # Fallback: fetch from GitHub
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
  endif()
endif()

# Google Test (for unit tests)
if(OMNICOPILOT_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  # Prevent overriding parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
endif()

# spdlog (logging)
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0
  )
  FetchContent_MakeAvailable(spdlog)
endif()

# ─── Proto Generation ────────────────────────────────────────────────────────

set(PROTO_DIR ${CMAKE_SOURCE_DIR}/proto)
set(PROTO_GEN_DIR ${CMAKE_BINARY_DIR}/generated/proto)
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

set(PROTO_FILES
  ${PROTO_DIR}/observation.proto
  ${PROTO_DIR}/entity.proto
  ${PROTO_DIR}/world_state.proto
  ${PROTO_DIR}/communication.proto
  ${PROTO_DIR}/metrics.proto
)

# Generate C++ from proto files
set(PROTO_SRCS)
set(PROTO_HDRS)

foreach(PROTO_FILE ${PROTO_FILES})
  get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
  list(APPEND PROTO_SRCS ${PROTO_GEN_DIR}/${PROTO_NAME}.pb.cc)
  list(APPEND PROTO_HDRS ${PROTO_GEN_DIR}/${PROTO_NAME}.pb.h)
endforeach()

add_custom_command(
  OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    --proto_path=${PROTO_DIR}
    --cpp_out=${PROTO_GEN_DIR}
    ${PROTO_FILES}
  DEPENDS ${PROTO_FILES}
  COMMENT "Generating C++ protobuf sources"
)

add_library(omnicopilot_proto STATIC ${PROTO_SRCS})
target_include_directories(omnicopilot_proto PUBLIC ${PROTO_GEN_DIR})
target_link_libraries(omnicopilot_proto PUBLIC protobuf::libprotobuf)

# ─── Subdirectories ──────────────────────────────────────────────────────────

add_subdirectory(cpp)
