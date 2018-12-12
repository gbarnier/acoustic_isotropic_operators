cmake_minimum_required(VERSION 2.8)
project(pySolver)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

#Installing all modules
install( DIRECTORY ./ DESTINATION lib/python)
