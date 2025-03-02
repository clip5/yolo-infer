#pragma once
#include "common.hpp"
#include "database.hpp"
#include "preprocess.hpp"

namespace yoloinfer
{
    void warp_affine_normal(const Image& src, Image& dst, const Norm& norm, float *matrix_2_3, void* stream = nullptr);

}