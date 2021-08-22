#! Detect CUDA
find_package(CUDA 11.0 REQUIRED)
message(STATUS ">>> Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CUDA_ARCHITECTURES sm_80)
# Enable optimization flags
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-std=c++11;-arch=${CMAKE_CUDA_ARCHITECTURES};-Xptxas;-O3;-restrict;-D_FORCE_INLINES;-D_DEBUG;-lineinfo;--compiler-options
    # --default-stream per-thread -std=c++14 -Wno-deprecated-gpu-targets -O3 --compiler-options -restrict -D_FORCE_INLINES -D_DEBUG -lineinfo	
)

aux_source_directory(../src ASTRO)
aux_source_directory(../src FOURIERTRAN)

set(libsrc
	../src/astro/cugridder.cu
	../src/astro/precomp.cu
	../src/astro/ra_exec.cu 
	../src/fouriertran/conv.cu
	../src/fouriertran/conv_interp_invoker.cu
	../src/fouriertran/cuft.cu
	../src/fouriertran/deconv.cu
	../src/fouriertran/interp.cu
	../src/utils.cu
	../contrib/common.cpp
	../contrib/legendre_rule_fast.c
	../contrib/utils_fp.cpp	
)

cuda_add_library(curafft_static STATIC 
    ${libsrc}

)

cuda_add_library(curafft SHARED 
	${libsrc}
)

cuda_add_executable(	
	adjointness_1d_test adjointness_1d_test.cu

)

target_link_libraries(adjointness_1d_test curafft_static m cudart cufft stdc++ nvToolsExt cuda)
