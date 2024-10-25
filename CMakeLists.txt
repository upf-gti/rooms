cmake_minimum_required(VERSION 3.16)

set(ROOMS_DIR_ROOT        ${CMAKE_CURRENT_SOURCE_DIR})
set(ROOMS_DIR_SOURCES     "${ROOMS_DIR_ROOT}/src")
set(ROOMS_DIR_LIBS        "${ROOMS_DIR_ROOT}/libraries")

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)
    endif()
endif()

project(rooms LANGUAGES C CXX)

# Enable multicore and simd compile on VS solution
if(MSVC)
    add_definitions(/MP)
    add_definitions(/arch:AVX2)

    # enable link time optimization
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        add_definitions(/GL)
        add_definitions(/LTCG)
    endif()

    # disable RTTI (dynamic cast)
#    string(REGEX REPLACE "/GR" "/GR-" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    # disable exceptions
	string(REGEX REPLACE "/EHsc" "/EHs-c-" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

add_definitions(-D_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING)

# Sources
macro(ROOMS_FILES_APPEND)
    file(GLOB FILES_APPEND CONFIGURE_DEPENDS ${ARGV})
    list(APPEND ROOMS_SOURCES ${FILES_APPEND})
endmacro()
macro(ROOMS_SOURCES_APPEND)
    ROOMS_FILES_APPEND(${ARGV0}/*.h)
    ROOMS_FILES_APPEND(${ARGV0}/*.cpp)
endmacro()

ROOMS_SOURCES_APPEND(${ROOMS_DIR_SOURCES})
ROOMS_SOURCES_APPEND(${ROOMS_DIR_SOURCES}/engine)
ROOMS_SOURCES_APPEND(${ROOMS_DIR_SOURCES}/graphics)
ROOMS_SOURCES_APPEND(${ROOMS_DIR_SOURCES}/graphics/renderers)
ROOMS_SOURCES_APPEND(${ROOMS_DIR_SOURCES}/graphics/managers)
ROOMS_SOURCES_APPEND(${ROOMS_DIR_SOURCES}/tools)
ROOMS_SOURCES_APPEND(${ROOMS_DIR_SOURCES}/framework)
ROOMS_SOURCES_APPEND(${ROOMS_DIR_SOURCES}/framework/resources)
ROOMS_SOURCES_APPEND(${ROOMS_DIR_SOURCES}/framework/nodes)
ROOMS_SOURCES_APPEND(${ROOMS_DIR_SOURCES}/framework/ui)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ROOMS_SOURCES})

add_executable(${PROJECT_NAME} ${ROOMS_SOURCES})

add_compile_definitions(_HAS_EXCEPTIONS=0)

target_include_directories(${PROJECT_NAME} PUBLIC ${ROOMS_DIR_SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_ENVIRONMENT "DAWN_DEBUG_BREAK_ON_ERROR=1")

set_property(DIRECTORY ${ROOMS_DIR_ROOT} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${ROOMS_DIR_ROOT}")

if (MSVC)
    add_compile_options(/Zc:__cplusplus)
endif()

if (EMSCRIPTEN)
    set(SHELL_FILE shell.html)

    include_directories(BEFORE
        "${CMAKE_BINARY_DIR}/_deps/emdawn-src/include/"
    )

    add_library(webgpu_layer
        "${CMAKE_BINARY_DIR}/_deps/emdawn-src/include/webgpu.cpp"
    )
    target_link_libraries(${PROJECT_NAME}
        webgpu_layer
    )

    add_compile_options(
    #"-fno-exceptions"
    "$<$<CONFIG:Release>:-flto>"
    )

    add_link_options(
    #"-fno-exceptions"
    "$<$<CONFIG:Release>:-flto>"
    )

    target_link_options(${PROJECT_NAME} PRIVATE

        # -sVERBOSE

        # We are using Dawn-generated bindings, not built-in ones
        -sUSE_WEBGPU=0
        # The JS libraries needed for bindings
        "--js-library=${CMAKE_BINARY_DIR}/_deps/emdawn-src/library_webgpu_enum_tables.js"
        "--js-library=${CMAKE_BINARY_DIR}/_deps/emdawn-src/library_webgpu_generated_struct_info.js"
        "--js-library=${CMAKE_BINARY_DIR}/_deps/emdawn-src/library_webgpu_generated_sig_info.js"
        "--js-library=${CMAKE_BINARY_DIR}/_deps/emdawn-src/library_webgpu.js"
        "--js-library=${CMAKE_BINARY_DIR}/_deps/emdawn-src/library_html5_webgpu.js"
        "--closure-args=--externs=${CMAKE_BINARY_DIR}/_deps/emdawn-src/webgpu-externs.js"

        -sUSE_GLFW=3
        -sALLOW_MEMORY_GROWTH
#        -sASYNCIFY
#        -sWASM=0
#        -sASSERTIONS
#        -fsanitize=address
        -Wdeprecated-literal-operator
        -sSTACK_SIZE=5MB
#        -sASYNCIFY_STACK_SIZE=10000
        -lembind # to bind functions from c++ to javascript
        -O3
#        -sWASM_BIGINT
        # only for dev
#        -sERROR_ON_WASM_CHANGES_AFTER_LINK
#        -O0
        # ------------
        --shell-file "${ROOMS_DIR_ROOT}/${SHELL_FILE}"
        --preload-file "${ROOMS_DIR_ROOT}/data@/data"
	)

	# Make sure to re-link when the shell file changes
    set_property(
        TARGET ${PROJECT_NAME}
        PROPERTY LINK_DEPENDS
        "${ROOMS_DIR_ROOT}/${SHELL_FILE}"
    )

    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".html")
endif()

# wgpuEngine
add_subdirectory(libraries/wgpuEngine)
target_link_libraries(${PROJECT_NAME} webgpuEngine)
