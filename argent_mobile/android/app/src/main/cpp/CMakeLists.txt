cmake_minimum_required(VERSION 3.18.1)
project(argent_llama)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Отключаем ненужные части llama.cpp для быстрой сборки и минимального размера
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Добавляем исходный код llama.cpp
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/CMakeLists.txt")
    add_subdirectory(llama.cpp)
else()
    message(WARNING "Каталог llama.cpp не найден в src/main/cpp/")
endif()

add_library(argent_llama SHARED
    argent_llama_jni.cpp
)

target_include_directories(argent_llama PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
)

target_link_libraries(argent_llama
    llama
    log
    android
)
