
cmake_minimum_required(VERSION 3.13)

set(ROOMS_DIR_ROOT        ${CMAKE_CURRENT_LIST_DIR})
set(ROOMS_DIR_SOURCES     "${ROOMS_DIR_ROOT}/src")

project(rooms LANGUAGES C CXX)

# Macro to map filters to folder structure for MSVC projects
macro(GroupSources curdir)
    if(MSVC)
		file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir} ${PROJECT_SOURCE_DIR}/${curdir}/*)

        foreach(child ${children})
            if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
                GroupSources(${curdir}/${child})
            else()
                string(REPLACE "/" "\\" groupname ${curdir})
                source_group(${groupname} FILES ${PROJECT_SOURCE_DIR}/${curdir}/${child})
            endif()
        endforeach()
    endif()
endmacro()

GroupSources(src)

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
ROOMS_SOURCES_APPEND(${ROOMS_DIR_SOURCES}/graphics)

add_executable(${PROJECT_NAME} ${ROOMS_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC ${ROOMS_DIR_SOURCES})

# set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_ENVIRONMENT "DAWN_DEBUG_BREAK_ON_ERROR=1")

set_property(DIRECTORY ${ROOMS_DIR_ROOT} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${ROOMS_DIR_ROOT}")

if (EMSCRIPTEN)
    set(SHELL_FILE shell.html)

    target_link_options(${PROJECT_NAME} PRIVATE
        -O3
        -sUSE_GLFW=3
        -sUSE_WEBGPU
        -sALLOW_MEMORY_GROWTH
        -sASYNCIFY
        --shell-file "${ROOMS_DIR_ROOT}/${SHELL_FILE}"
        --preload-file "${ROOMS_DIR_ROOT}/data/shaders@/data/shaders"
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

# vulkan
include_directories(${Vulkan_INCLUDE_DIR})
