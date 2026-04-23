# PEIRASTIC_control: 基于 libfranka 的 Franka Panda 控制栈（独立副本）
# 含：franka-interface (C++) + peirastic (Python ZMQ 客户端)

cmake_minimum_required(VERSION 3.10)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

project(PEIRASTIC_control VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(DEFAULT_PYTHON_MODULE_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
else()
  set(DEFAULT_PYTHON_MODULE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
set(PYTHON_MODULE_OUTPUT_DIRECTORY "${DEFAULT_PYTHON_MODULE_OUTPUT_DIRECTORY}" CACHE STRING "")

add_compile_options(-Wall -Wextra)
find_package(Protobuf REQUIRED)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/protobuf.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/libfranka/cmake/FindEigen3.cmake)

set(FRANKA_PROTO_FILES
  proto/franka-interface/franka_robot_state.proto
  proto/franka-interface/franka_controller.proto
)
set(PROTO_FILES ${FRANKA_PROTO_FILES})

set(BUILD_FRANKA ON CACHE BOOL "")
set(BUILD_PEIRASTIC ON CACHE BOOL "")
# 不设 PROTO_NAMESPACE，使生成到 proto/${PROJECT_NAME}/，include 路径正确

if(BUILD_FRANKA)
  add_proto_cc_library(shared_protobuf
    PROTO_FILES ${FRANKA_PROTO_FILES}
    PROTO_LIB protobuf::libprotobuf)
  set_target_properties(shared_protobuf PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set(BUILD_TESTS OFF CACHE BOOL "")
  set(BUILD_EXAMPLES OFF CACHE BOOL "")
  find_package(Eigen3 REQUIRED)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)

  add_subdirectory(libfranka)
  add_subdirectory(zmqpp)
  add_subdirectory(spdlog)
  set(YAML_BUILD_SHARED_LIBS ON CACHE BOOL "")
  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "")
  add_subdirectory(yaml-cpp)
  set(BUILD_EXAMPLES OFF CACHE BOOL "")
  set(BUILD_PYTHON_MODULE OFF CACHE BOOL "")
  set(BUILD_CLOUD_CLIENT OFF CACHE BOOL "")
  set(BUILD_BENCHMARK OFF CACHE BOOL "")
  add_subdirectory(third_party/ruckig)
  include_directories(zmqpp/src)
  include_directories(franka-interface/include)

  set(FRANKA_LIBS Eigen3::Eigen3 Franka::Franka Threads::Threads zmqpp yaml-cpp spdlog ruckig::ruckig)

  set(CONTROLLERS
    franka-interface/src/controllers/osc_impedance.cpp
    franka-interface/src/controllers/osc_position_impedance.cpp
    franka-interface/src/controllers/osc_yaw_impedance.cpp
    franka-interface/src/controllers/joint_position.cpp
    franka-interface/src/controllers/joint_impedance.cpp
    franka-interface/src/controllers/cartesian_velocity.cpp
  )
  add_library(shared_utils
    franka-interface/src/utils/common_utils.cpp
    franka-interface/src/utils/control_utils.cpp
    franka-interface/src/utils/robot_utils.cpp
    franka-interface/src/utils/zmq_utils.cpp
    ${CONTROLLERS})
  target_compile_features(shared_utils PUBLIC cxx_std_20)
  target_link_libraries(shared_utils PEIRASTIC_control::shared_protobuf protobuf::libprotobuf ${FRANKA_LIBS})

  add_executable(franka-interface franka-interface/src/franka_control_node.cpp)
  target_link_libraries(franka-interface PEIRASTIC_control::shared_protobuf ${FRANKA_LIBS} shared_utils protobuf::libprotobuf)

  add_executable(gripper-interface franka-interface/src/gripper_control_node.cpp)
  target_link_libraries(gripper-interface PEIRASTIC_control::shared_protobuf ${FRANKA_LIBS} shared_utils protobuf::libprotobuf)

  install(TARGETS franka-interface gripper-interface RUNTIME DESTINATION bin)
endif()

if(BUILD_PEIRASTIC)
  add_proto_py_library(py_protobuf PROTO_FILES ${PROTO_FILES} PROTO_LIB protobuf NAMESPACE franka_interface)
  install(DIRECTORY "${PY_PROTOBUF_OUTPUT_DIR}/" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/peirastic/proto/franka_interface")
endif()
