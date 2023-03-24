#ifndef __UTILS_H
#define __UTILS_H

#include <iostream>
#include <initializer_list>
#include <memory>
#include <string>
#include <cstring>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdarg.h>
#include <unordered_map>

#define GPU_BLOCK_THREADS 512
#define INFO(...) __log_func(__FILE__, __LINE__, __VA_ARGS__)
void __log_func(const char *file, int line, const char *fmt, ...);

#define checkRuntime(call)                                                     \
do {                                                                         \
    auto ___call__ret_code__ = (call);                                         \
    if (___call__ret_code__ != cudaSuccess) {                                  \
    INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call,             \
        cudaGetErrorString(___call__ret_code__),                            \
        cudaGetErrorName(___call__ret_code__), ___call__ret_code__);        \
    abort();                                                                 \
    }                                                                          \
} while (0)
#define checkKernel(...)                                                       \
    do {                                                                         \
        { (__VA_ARGS__); }                                                         \
        checkRuntime(cudaPeekAtLastError());                                       \
    } while (0)

#define Assert(op)                                                                 \
do {                                                                           \
    bool cond = !(!(op));                                                      \
    if (!cond) {                                                               \
    INFO("Assert failed, " #op);                                               \
    abort();                                                                   \
    }                                                                          \
} while (0)
#define Assertf(op, ...)                                                           \
    do {                                                                           \
        bool cond = !(!(op));                                                      \
        if (!cond) {                                                               \
        INFO("Assert failed, " #op " : " __VA_ARGS__);                             \
        abort();                                                                   \
        }                                                                          \
    } while (0)

const int NUM_BOX_ELEMENT = 8; // left, top, right, bottom, confidence, class,
                               // keepflag, row_index(output)
const int MAX_IMAGE_BOXES = 1024;

enum class DType : int
{
    FLOAT = 0,
    HALF = 1,
    INT8 = 2,
    INT32 = 3,
    BOOL = 4,
    UINT8 = 5
};

enum class Type : int
{
    V5 = 0,
    X = 1,
    V3 = 2,
    V7 = 3,
    V8 = 5,
    V8Seg = 6 // yolov8 instance segmentation
};

enum class NormType : int { None = 0, MeanStd = 1, AlphaBeta = 2 };

enum class ChannelType : int { None = 0, SwapRB = 1 };

/* 归一化操作，可以支持均值标准差，alpha beta，和swap RB */
struct Norm
{
    float mean[3];
    float std[3];
    float alpha, beta;
    NormType type = NormType::None;
    ChannelType channel_type = ChannelType::None;

    // out = (x * alpha - mean) / std
    static Norm mean_std(const float mean[3], const float std[3],
                        float alpha = 1 / 255.0f,
                        ChannelType channel_type = ChannelType::None);

    // out = x * alpha + beta
    static Norm alpha_beta(float alpha, float beta = 0,
                            ChannelType channel_type = ChannelType::None);

    // None
    static Norm None();
};

static const char *type_name(Type type)
{
    switch (type)
    {
        case Type::V5:
            return "YoloV5";
        case Type::V3:
            return "YoloV3";
        case Type::V7:
            return "YoloV7";
        case Type::X:
            return "YoloX";
        case Type::V8:
            return "YoloV8";
        default:
            return "Unknow";
    }
}

// const char *type_name(Type type);
std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);
std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);
#endif