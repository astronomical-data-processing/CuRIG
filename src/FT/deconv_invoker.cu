/* 
invoke deconv related kernels
*/
#include "deconv_invoker.h"

int nupt_deconv(){

}

int sorted_deconv(){

}

int curafft_deconv(curafft_plan *plan){
    int nf1 = plan->nf1;
    int nf2 = plan->nf2;
    int num_w = plan->num_w;

    if(plan->opts.gpu_gridder_method == ){
        nupt_deconv()
    }
}