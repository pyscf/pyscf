set(FFTW_FOUND FALSE)

#
#   1. Inspect $FFTW_DIR
#
set(FFTW_DIR $ENV{FFTW_DIR})
if(FFTW_DIR)
    if(EXISTS "${FFTW_DIR}")
        set(PATH_FFTW "${FFTW_DIR}")
    endif(EXISTS "${FFTW_DIR}")
endif(FFTW_DIR)

find_file(FFTW_H fftw3.h PATHS ${PATH_FFTW} PATH_SUFFIXES "include"
    NO_DEFAULT_PATH)

if(FFTW_H)
    set(FFTW_NAME fftw3)
endif(FFTW_H)
set(FFTW_H_FOUND TRUE)

find_library(FFTW_A_PATH ${FFTW_NAME} PATHS ${PATH_FFTW} PATH_SUFFIXES "lib"
    NO_DEFAULT_PATH)
if(FFTW_H AND (NOT FFTW_A_PATH))
    message(SEND_ERROR "Proper FFTW was not found in ${PATH_FFTW}")
endif()

#
#   2. Inspect standard locations
#
if(NOT FFTW_H_FOUND)
    find_file(FFTW_H fftw3.h)
    find_library(FFTW_A_PATH fftw3)
endif(NOT FFTW_H_FOUND)

#
#   3. Set output variables
#
if(FFTW_A_PATH)
    get_filename_component(PATH_FFTW_INCLUDE "${FFTW_H}" PATH)
    get_filename_component(PATH_FFTW_LIB "${FFTW_A_PATH}" PATH)
    add_library(${FFTW_NAME} SHARED IMPORTED)
    set_target_properties(${FFTW_NAME} PROPERTIES
        IMPORTED_LOCATION ${FFTW_A_PATH})
    set(FFTW_LIBRARIES ${FFTW_NAME})
    set(FFTW_LIBRARIES_EXPLICIT ${FFTW_A_PATH})

    set(FFTW_FOUND TRUE)
endif()

