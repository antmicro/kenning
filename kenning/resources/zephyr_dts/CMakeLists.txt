# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr COMPONENTS yaml REQUIRED HINTS $ENV{ZEPHYR_BASE})
project("MakeDTS")

include(python)
include(boards)
include(hwm_v2)
include(west)
list(APPEND DTS_ROOT ${WEST_TOPDIR}/modules/hal/stm32 ${WEST_TOPDIR}/modules/hal/microchip ${WEST_TOPDIR}/modules/hal/nuvoton ${WEST_TOPDIR}/modules/hal/nxp ${WEST_TOPDIR}/modules/hal/atmel ${WEST_TOPDIR}/modules/hal/gigadevice)
include(pre_dt)
find_package(HostTools)
find_package(Dtc 1.4.6)

# The generated file containing the final DTS, for debugging.
set(ZEPHYR_DTS                  ${PROJECT_BINARY_DIR}/zephyr.dts)
# Generated build system internals.
set(DTS_POST_CPP                ${PROJECT_BINARY_DIR}/zephyr.dts.pre)

if(NOT DEFINED DTS_SOURCE)
  zephyr_build_string(board_string SHORT shortened_board_string
                      BOARD ${BOARD} BOARD_QUALIFIERS ${BOARD_QUALIFIERS}
  )
  foreach(dir ${BOARD_DIRECTORIES})
    if(EXISTS ${dir}/${shortened_board_string}.dts AND NOT BOARD_${BOARD}_SINGLE_SOC)
      message(FATAL_ERROR "Board ${ZFILE_BOARD} defines multiple SoCs.\nShortened file name "
              "(${shortened_board_string}.dts) not allowed, use '<board>_<soc>.dts' naming"
      )
    elseif(EXISTS ${dir}/${board_string}.dts AND EXISTS ${dir}/${shortened_board_string}.dts)
      message(FATAL_ERROR "Conflicting file names discovered. Cannot use both "
              "${board_string}.dts and ${shortened_board_string}.dts. "
              "Please choose one naming style, ${board_string}.dts is recommended."
      )
    elseif(EXISTS ${dir}/${board_string}.dts)
      set(DTS_SOURCE ${dir}/${board_string}.dts)
    elseif(EXISTS ${dir}/${shortened_board_string}.dts)
      set(DTS_SOURCE ${dir}/${shortened_board_string}.dts)
    endif()
  endforeach()
endif()

if(EXISTS ${DTS_SOURCE})
  # We found a devicetree. Append all relevant dts overlays we can find...
  zephyr_file(CONF_FILES ${BOARD_DIRECTORIES} DTS DTS_SOURCE)

  zephyr_file(
    CONF_FILES ${BOARD_DIRECTORIES}
    DTS no_rev_suffix_dts_board_overlays
    BOARD ${BOARD}
    BOARD_QUALIFIERS ${BOARD_QUALIFIERS}
  )

  # ...but remove the ones that do not include the revision suffix
  list(REMOVE_ITEM DTS_SOURCE ${no_rev_suffix_dts_board_overlays})
else()
  # If we don't have a devicetree, provide an empty stub
  set(DTS_SOURCE ${ZEPHYR_BASE}/boards/common/stub.dts)
endif()

#
# Find all the DTS files we need to concatenate and preprocess, as
# well as all the devicetree bindings and vendor prefixes associated
# with them.
#

zephyr_file(CONF_FILES ${BOARD_EXTENSION_DIRS} DTS board_extension_dts_files)

set(dts_files
  ${DTS_SOURCE}
  ${board_extension_dts_files}
  ${shield_dts_files}
  )

if(DTC_OVERLAY_FILE)
  zephyr_list(TRANSFORM DTC_OVERLAY_FILE NORMALIZE_PATHS
              OUTPUT_VARIABLE DTC_OVERLAY_FILE_AS_LIST)
  build_info(devicetree user-files PATH ${DTC_OVERLAY_FILE_AS_LIST})
  list(APPEND
    dts_files
    ${DTC_OVERLAY_FILE_AS_LIST}
    )
endif()

set(i 0)
foreach(dts_file ${dts_files})
  if(i EQUAL 0)
    message(STATUS "Found BOARD.dts: ${dts_file}")
  else()
    message(STATUS "Found devicetree overlay: ${dts_file}")
  endif()

  math(EXPR i "${i}+1")
endforeach()

unset(DTS_ROOT_BINDINGS)
foreach(dts_root ${DTS_ROOT})
  set(bindings_path ${dts_root}/dts/bindings)
  if(EXISTS ${bindings_path})
    list(APPEND
      DTS_ROOT_BINDINGS
      ${bindings_path}
      )
  endif()
endforeach()

#
# Run the C preprocessor on the devicetree source.
#

if(DEFINED CMAKE_DTS_PREPROCESSOR)
  set(dts_preprocessor ${CMAKE_DTS_PREPROCESSOR})
else()
  set(dts_preprocessor ${CMAKE_C_COMPILER})
endif()
make_directory(${PROJECT_BINARY_DIR})

set(include_opts)
foreach(dir ${DTS_ROOT_SYSTEM_INCLUDE_DIRS})
  list(APPEND include_opts -isystem ${dir})
endforeach()

set(source_opts)
foreach(file ${dts_files})
  list(APPEND source_opts -include ${file})
endforeach()

set(workdir_opts)
if(DEFINED APPLICATION_SOURCE_DIR)
  list(APPEND workdir_opts WORKING_DIRECTORY ${APPLICATION_SOURCE_DIR})
endif()

set(preprocess_cmd ${dts_preprocessor}
  -H -E -P -x assembler-with-cpp
  ${include_opts}
  ${source_opts}
  ${NOSYSDEF_CFLAG}
  -D__DTS__
  ${DTS_EXTRA_CPPFLAGS}
  -o ${DTS_POST_CPP}
  ${ZEPHYR_BASE}/misc/empty_file.c
  ${workdir_opts})

execute_process(COMMAND ${preprocess_cmd} RESULT_VARIABLE ret COMMAND_ECHO "STDERR")
if(NOT "${ret}" STREQUAL "0")
  message(FATAL_ERROR "failed to preprocess devicetree files (error code ${ret}): ${dts_files}")
endif()
