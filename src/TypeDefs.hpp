#pragma once

typedef struct {
    double e11=0.0;
    double e12=0.0;
    double e13=0.0;
    double e22=0.0;
    double e23=0.0;
    double e33=0.0;
} matrix6;

typedef struct {
    double e11 = 0.0;
    double e12 = 0.0;
    double e13 = 0.0;
    double e21 = 0.0;
    double e22 = 0.0;
    double e23 = 0.0;
    double e31 = 0.0;
    double e32 = 0.0;
    double e33 = 0.0;
} matrix9;

typedef struct {
    double u;
    double v;
    double w;   
} Wind;

typedef struct{
    double e11;
    double e21;
    double e31;
} vec3;

typedef struct{
    double e11;
    double e22;
    double e33;
} diagonal;

typedef struct{
    double x;
    double y;
    double z;
} pos;
