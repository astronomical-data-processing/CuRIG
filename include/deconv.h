#ifndef __DECONV_H__
#define __DECONV_H__

#include "curafft_plan.h"
#ifdef __cplusplus
extern "C"
{
#include "legendre_rule_fast.h"
}
#else
#include "legendre_rule_fast.h"
#endif
void fourier_series_appro_invoker(PCS *fseries, PCS *k, conv_opts opts, int N)
int curafft_deconv(curafft_plan *plan);
#endif