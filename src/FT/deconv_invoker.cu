/* 
invoke deconv related kernels
*/
#include "deconv_invoker.h"

int 2d_nupt_deconv(){
    int ier = 0;
    // niffty + cufinufft
    
    return ier;
}

int 2d_sorted_deconv(){
    int ier = 0;

    
    return ier;
}

int w_term_correction(){
    int ier = 0;

    return ier;
}

int curafft_deconv(curafft_plan *plan, int cur_bth_size){
    int ier = 0;
    int nf1 = plan->nf1;
    int nf2 = plan->nf2;
    int nf3 = plan->num_w;
    int M = plan->M;

    switch (plan->opts.gpu_gridder_method)
    {
        case 1:{
            ier = nupt_deconv(nf1, nf2, nf3, M, plan, cur_bth_size);
            break;
        }
        
        case 2:{
            ier = sorted_deconv(nf1, nf2, nf3, M, plan, cur_bth_size);
            break;
        }
        default:{
            cout<<"incorrect degridding method\n";
            ier = 2;
            break;
        }
    }
    return ier;
}