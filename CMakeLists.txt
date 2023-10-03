cmake_minimum_required(VERSION 3.24)
project(opp_lab_3 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -fopenmp")

add_executable(opp_lab_3 main.c)
