#ifndef __CURAFFT_OPTS_H__
#define __CURAFFT_OPTS_H__

struct curafft_opts {  
	/*
		upsampfac - upsampling ratio sigma, only 2.0 (standard) is implemented
		gpu_device_id - for choosing GPU
		gpu_sort - 1 using sort, 0 not

		add content explaination
	*/

	double upsampfac;   // upsampling ratio sigma, only 2.0 (standard) is implemented

    /* multi-gpu support */
	int gpu_device_id;

	/* For GM_sort method*/
	int gpu_sort; // 1 using sort.
	int gpu_binsizex; 
	int gpu_binsizey;
	int gpu_binsizez;
	int gpu_kerevalmeth; // 0: direct exp(sqrt()), 1: Horner ppval default 0
	int gpu_conv_only; // 0: NUFFT, 1: spread or interpolation only
	int gpu_method;


    curafft_opts(){
        gpu_device_id = 0;
        upsampfac = 2.0;
		gpu_sort = 1;
		gpu_binsizex = -1;
		gpu_binsizey = -1;
		gpu_binsizez = -1;
		gpu_kerevalmeth = 0;
		gpu_conv_only = 0;
    }


	/* following options are for gpu */
    /*
        int gpu_method;  // 1: nonuniform-pts driven, 2: shared mem (SM)
	int gpu_sort;    // when NU-pts driven: 0: no sort (GM), 1: sort (GM-sort)


	int gpu_binsizex; // used for 2D, 3D subproblem method
	int gpu_binsizey;
	int gpu_binsizez;

	int gpu_obinsizex; // used for 3D spread block gather method
	int gpu_obinsizey;
	int gpu_obinsizez;

	int gpu_maxsubprobsize;
	int gpu_nstreams;
	int gpu_kerevalmeth; // 0: direct exp(sqrt()), 1: Horner ppval default 0

	int gpu_spreadinterponly; // 0: NUFFT, 1: spread or interpolation only
    */
};

#endif