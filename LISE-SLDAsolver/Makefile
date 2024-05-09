#
# Parameters for compiling LISE solver:
#
# CC                    -the C compiler command
# LISE_LA_INC   	-the path to resolve header files for 'blas.h' (or equivalent)
# LISE_LA_LNK   	-the link line for ScaLAPACK and dependencies
# LISE_FFTW_INC         -the path to resolve header files for 'fftw.h' 
# LISE_FFTW_LNK         -the link line for FFTW 
# LISE_MPI_INC          -the path to resolve 'mpi.h' header file
# LISE_MPI_LNK          -the link line for MPI 
#
# summit.olcf.ornl.gov: IBM AC922 - IBM Power9 + NVIDIA VT100
#
# compiler: IBM XL (xl-16.1.1-5)
# operating system: linux-rhel7-ppc64le
# MPI: IBM Spectrum MPI (spectrum-mpi/10.3.1.2-20200121)
# ScaLAPACK: Netlib ScaLAPACK 2.0.2 
# BLAS: IBM ESSL (essl/6.1.0-2)
# FFTW: FFTW 3.3.8 (fftw/3.3.8) 
#
# NOTE: 
# 	It is recommended that Darshan, a default, automated I/O profiler, be excluded from building and running LISE codes:
# 	module unload darshan-runtime
#
# Modules:
# hsi/5.0.2.p5
# xalt/1.2.1
# lsf-tools/2.0
# DefApps
# essl/6.1.0-2
# xl/16.1.1-5
# spectrum-mpi/10.3.1.2-20200121
# fftw/3.3.8
# netlib-scalapack/2.0.2
#
# Environment variables:
#
# OLCF_ESSL_ROOT 		= /sw/summit/essl/6.1.0-2/essl/6.1
# OLCF_NETLIB_SCALAPACK_ROOT 	= /autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-5/netlib-scalapack-2.0.2-xo4jhy4ydnx3qqmnegbzfrvz4am6q2v6
# OLCF_FFTW_ROOT 		= /autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-5/fftw-3.3.8-azzdjlzx2j6dpqvzdir2nwvxypohyfq4
# MPI_ROOT 			= /autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-5/spectrum-mpi-10.3.1.2-20200121-p6nrnt6vtvkn356wqg6f74n6jspnpjd2
#
#

LINK= $(LISE_FFTW_LNK) $(LISE_LA_LNK) -lm $(LISE_MPI_LNK)

EXE := xlise-static

BUILD_DIR := ./builds
SRC_DIRS := ./src

SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' -or -name '*.c' -or -name '*.s')
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS := $(INC_FLAGS) -MMD -MP

all: $(EXE)

$(EXE): $(OBJS) 
	$(CC) -o $(EXE) $(OBJS) $(LINK) 
	cp $(EXE) $(LISE_BIN_DIR)

%.o : %.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

.PHONY : clean
clean:
	rm -f $(OBJS) $(EXE) $(COBSJ)

-include $(DEPS)