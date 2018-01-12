cmake_minimum_required(VERSION 3.2)
project(CameraThread)

# default built type
IF(NOT CMAKE_BUILD_TYPE)
        SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_STANDARD 11)
add_definitions(-DBOOST_NO_CXX11_SCOPED_ENUMS)
add_definitions(-DBOOST_ALL_DYN_LINK)

find_package(OpenCV REQUIRED)

find_package(Qt5Gui)
find_package(Qt5Core)
find_package(Qt5Widgets)
find_package(Qt5Test)
find_package(Qt5Concurrent)
find_package(Qt5OpenGL)

set(SOURCE_FILES main.cpp)
add_executable(CameraThread ${SOURCE_FILES})

target_link_libraries(
        CameraThread

        pthread

        boost_system
        boost_thread
        boost_filesystem
        boost_date_time
        boost_chrono

        opencv_core
        opencv_highgui
        opencv_calib3d
        )