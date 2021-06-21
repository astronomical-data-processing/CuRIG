#ifndef __PRECOMP__H__
#define __PRECOMP__H__

#include <cuda.h>
#include <helper_cuda.h>
#include "dataType.h"
#include "ragridder_plan.h"
#include "curafft_plan.h"

void explicit_gridder_invoker(ragridder_plan *gridder_plan);
void pre_setting(PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_vis, curafft_plan *plan, ragridder_plan *gridder_plan);

#endif