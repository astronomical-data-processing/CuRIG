#ifndef __VISIBILITY_H__
#define __VISIBILITY_H__


#include "dataType.h"

struct visibility
{
    /*
    visbility related componment
    u,v,w - coordinate in frequent domain, unit wavelength
    vis - complex number
    weight
    */ 
    PCS* u;
    PCS* v;
    PCS* w;

    CPX* vis;
    PCS* weight;
    bool* flag;



    //PCS time;
    PCS* frequency;
    //int antenna_1;
    //int antenna_2;
    //PCS image_weight;
    //PCS channel_bandwidth;
    //PCS integration_time;
};

#endif
