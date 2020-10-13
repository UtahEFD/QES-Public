#pragma once

struct polyVert
{
    polyVert()
        : x_poly(0.0), y_poly(0.0) {}

    polyVert(float x, float y)
        : x_poly(x), y_poly(y) {}

    float x_poly, y_poly;
};
