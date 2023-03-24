#ifndef __YOLO_DETECTOR_H
#define __YOLO_DETECTOR_H

#include <future>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "infer.h"
#include "utils.h"
#include "yolo_kernel.h"

namespace yolo_detector
{
    struct InstanceSegmentMap
    {
        int width = 0, height = 0; // width % 8 == 0
        unsigned char* data = nullptr; // is width * height memory

        InstanceSegmentMap(int width, int height);
        virtual ~InstanceSegmentMap();
    };

    struct Box
    {
        float left, top, right, bottom, confidence;
        int class_label;
        std::shared_ptr<InstanceSegmentMap> seg; // valid only in segment task

        Box() = default;
        Box(float left, float top, float right, float bottom, float confidence, int class_label):
            left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {}
    };

    typedef std::vector<Box> BoxArray;
    
    struct Image
    {
        const void* bgrptr = nullptr;
        int width = 0, height = 0;

        Image() = default;
        Image(const void *bgrptr, int width, int height):
            bgrptr(bgrptr), width(width), height(height) {}
    };
    
    struct AffineMatrix
    {
        float i2d[6]; // image to dst(network), 2x3 matrix
        float d2i[6]; // dst to image, 2x3 matrix

        void compute(const std::tuple<int, int> &from,
                    const std::tuple<int, int> &to)
        {
            float scale_x = std::get<0>(to) / (float)std::get<0>(from);
            float scale_y = std::get<1>(to) / (float)std::get<1>(from);
            float scale = std::min(scale_x, scale_y);
            i2d[0] = scale;
            i2d[1] = 0;
            i2d[2] = -scale * std::get<0>(from) * 0.5 + std::get<0>(to) * 0.5 + scale * 0.5 - 0.5;
            i2d[3] = 0;
            i2d[4] = scale;
            i2d[5] = -scale * std::get<1>(from) * 0.5 + std::get<1>(to) * 0.5 + scale * 0.5 - 0.5;

            double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
            D = D != 0. ? double(1.) / D : double(0.);
            double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D,
                A21 = -i2d[3] * D;
            double b1 = -A11 * i2d[2] - A12 * i2d[5];
            double b2 = -A21 * i2d[2] - A22 * i2d[5];

            d2i[0] = A11;
            d2i[1] = A12;
            d2i[2] = b1;
            d2i[3] = A21;
            d2i[4] = A22;
            d2i[5] = b2;
        }
    };

    // [Preprocess]: 0.50736 ms
    // [Forward]: 3.96410 ms
    // [BoxDecode]: 0.12016 ms
    // [SegmentDecode]: 0.15610 ms
    class Infer
    {
    public:
        virtual BoxArray forward(const Image &image, void *stream = nullptr) = 0;
        virtual std::vector<BoxArray> forwards(
            const std::vector<Image> &images, void *stream = nullptr) = 0;
    };

    class InferImpl : public Infer
    {
    private:
        /* data */
    public:
        std::shared_ptr<trt::Infer> trt_;
        std::string engine_file_;
        Type type_;
        float confidence_threshold_;
        float nms_threshold_;
        std::vector<std::shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
        trt::Memory<float> input_buffer_, bbox_predict_, output_boxarray_;
        trt::Memory<float> segment_predict_;
        int network_input_width_, network_input_height_;
        Norm normalize_;
        std::vector<int> bbox_head_dims_;
        std::vector<int> segment_head_dims_;
        int num_classes_ = 0;
        bool has_segment_ = false;
        bool isdynamic_model_ = false;
        std::vector<std::shared_ptr<trt::Memory<unsigned char>>> box_segment_cache_;

        void adjust_memory(int batch_size);
        void preprocess(int ibatch, const Image &image,
                  std::shared_ptr<trt::Memory<unsigned char>> preprocess_buffer,
                  AffineMatrix &affine, void *stream = nullptr);
        bool load(const std::string &engine_file, Type type, float confidence_threshold,
                float nms_threshold);
        virtual BoxArray forward(const Image &image, void *stream = nullptr) override;
        virtual std::vector<BoxArray> forwards(const std::vector<Image> &images, void *stream = nullptr) override;
    };


    std::shared_ptr<Infer> load(const std::string &engine_file, Type type,
                            float confidence_threshold = 0.25f,
                            float nms_threshold = 0.5f);
} // namespace yolo_detector


#endif