#include <libsgm.h>

#define API extern "C" __declspec(dllexport)

API void* sgm_create(
    int width,
    int height,
    int disparities,
    int p1,
    int p2,
    float uniqueness,
    int paths,
    int min_disp,
    int lr_max_diff)
{
    auto path = paths == 4
        ? sgm::PathType::SCAN_4PATH
        : sgm::PathType::SCAN_8PATH;

    sgm::StereoSGM::Parameters param(
        p1,
        p2,
        uniqueness,
        true,   // subpixel
        path,
        min_disp,
        lr_max_diff,
        sgm::CensusType::SYMMETRIC_CENSUS_9x7
    );

    return new sgm::StereoSGM(
        width,
        height,
        disparities,
        8,      // uint8 input
        16,     // int16 disparity
        sgm::EXECUTE_INOUT_CUDA2CUDA,
        param
    );
}

API void sgm_execute(
    void* handle,
    const void* left,
    const void* right,
    void* disparity)
{
    static_cast<sgm::StereoSGM*>(handle)->execute(
        left, right, disparity
    );
}

API int sgm_invalid(void* handle)
{
    return static_cast<sgm::StereoSGM*>(handle)
        ->get_invalid_disparity();
}

API void sgm_destroy(void* handle)
{
    delete static_cast<sgm::StereoSGM*>(handle);
}