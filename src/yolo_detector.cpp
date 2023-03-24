#include "../include/yolo_detector.h"

namespace yolo_detector
{
    InstanceSegmentMap::InstanceSegmentMap(int width, int height)
    {
        this->width = width;
        this->height = height;
        checkRuntime(cudaMallocHost(&this->data, width * height));
    }

    InstanceSegmentMap::~InstanceSegmentMap()
    {
        if (this->data)
        {
            checkRuntime(cudaFreeHost(this->data));
            this->data = nullptr;
        }
        this->width = 0;
        this->height = 0;
    }

    inline int upbound(int n, int align = 32)
    {
        return (n + align - 1) / align * align;
    }

    void InferImpl::adjust_memory(int batch_size)
    {
        // the inference batch_size
        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        input_buffer_.gpu(batch_size * input_numel);
        bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
        output_boxarray_.gpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
        output_boxarray_.cpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));

        if (has_segment_)
            segment_predict_.gpu(batch_size * segment_head_dims_[1] *
                                segment_head_dims_[2] * segment_head_dims_[3]);

        if (preprocess_buffers_.size() < batch_size)
        {
            for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
                preprocess_buffers_.push_back(
                    std::make_shared<trt::Memory<unsigned char>>());
        }
    }

    void InferImpl::preprocess(int ibatch, const Image &image,
                  std::shared_ptr<trt::Memory<unsigned char>> preprocess_buffer,
                  AffineMatrix &affine, void *stream)
    {
        affine.compute(std::make_tuple(image.width, image.height),
                    std::make_tuple(network_input_width_, network_input_height_));

        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        float *input_device = input_buffer_.gpu() + ibatch * input_numel;
        size_t size_image = image.width * image.height * 3;
        size_t size_matrix = upbound(sizeof(affine.d2i), 32);
        uint8_t *gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
        float *affine_matrix_device = (float *)gpu_workspace;
        uint8_t *image_device = gpu_workspace + size_matrix;

        uint8_t *cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
        float *affine_matrix_host = (float *)cpu_workspace;
        uint8_t *image_host = cpu_workspace + size_matrix;

        // speed up
        cudaStream_t stream_ = (cudaStream_t)stream;
        memcpy(image_host, image.bgrptr, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkRuntime(cudaMemcpyAsync(image_device, image_host, size_image,
                                    cudaMemcpyHostToDevice, stream_));
        checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host,
                                    sizeof(affine.d2i), cudaMemcpyHostToDevice,
                                    stream_));

        warp_affine_bilinear_and_normalize_plane(
            image_device, image.width * 3, image.width, image.height, input_device,
            network_input_width_, network_input_height_, affine_matrix_device, 114,
            normalize_, stream_);
    }

    bool InferImpl::load(const std::string &engine_file, Type type, float confidence_threshold, float nms_threshold)
    {
        trt_ = trt::load(engine_file);
        if (trt_ == nullptr)
        return false;

        trt_->print();

        this->type_ = type;
        this->confidence_threshold_ = confidence_threshold;
        this->nms_threshold_ = nms_threshold;

        auto input_dim = trt_->static_dims(0);
        bbox_head_dims_ = trt_->static_dims(1);
        has_segment_ = type == Type::V8Seg;
        if (has_segment_)
        {
            bbox_head_dims_ = trt_->static_dims(2);
            segment_head_dims_ = trt_->static_dims(1);
        }
        network_input_width_ = input_dim[3];
        network_input_height_ = input_dim[2];
        isdynamic_model_ = trt_->has_dynamic_dim();

        if (type == Type::V5 || type == Type::V3 || type == Type::V7)
        {
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 5;
        }
        else if (type == Type::V8)
        {
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 4;
        }
        else if (type == Type::V8Seg)
        {
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 4 - segment_head_dims_[1];
        }
        else if (type == Type::X)
        {
            // float mean[] = {0.485, 0.456, 0.406};
            // float std[]  = {0.229, 0.224, 0.225};
            // normalize_ = Norm::mean_std(mean, std, 1/255.0f, ChannelType::SwapRB);
            normalize_ = Norm::None();
            num_classes_ = bbox_head_dims_[2] - 5;
        }
        else
        {
            INFO("Unsupport type %d", type);
        }
        return true;
    }

    BoxArray InferImpl::forward(const Image &image, void *stream)
    {
        auto output = forwards({image}, stream);
        if (output.empty())
            return {};
        return output[0];
    }

    std::vector<BoxArray> InferImpl::forwards(const std::vector<Image> &images, void *stream)
    {
        int num_image = images.size();
        if (num_image == 0)
            return {};

        auto input_dims = trt_->static_dims(0);
        int infer_batch_size = input_dims[0];
        if (infer_batch_size != num_image)
        {
            if (isdynamic_model_)
            {
                infer_batch_size = num_image;
                input_dims[0] = num_image;
                if (!trt_->set_run_dims(0, input_dims))
                    return {};
            }
            else
            {
                if (infer_batch_size < num_image)
                {
                    INFO("When using static shape model, number of images[%d] must be "
                        "less than or equal to the maximum batch[%d].",
                        num_image, infer_batch_size);
                    return {};
                }
            }
        }
        adjust_memory(infer_batch_size);

        std::vector<AffineMatrix> affine_matrixs(num_image);
        cudaStream_t stream_ = (cudaStream_t)stream;
        for (int i = 0; i < num_image; ++i)
        {
            preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);
        }

        float *bbox_output_device = bbox_predict_.gpu();
        std::vector<void *> bindings{input_buffer_.gpu(), bbox_output_device};

        if (has_segment_)
        {
            bindings = {input_buffer_.gpu(), segment_predict_.gpu(),
                        bbox_output_device};
        }

        if (!trt_->forward(bindings, stream))
        {
            INFO("Failed to tensorRT forward.");
            return {};
        }

        for (int ib = 0; ib < num_image; ++ib)
        {
            float *boxarray_device = output_boxarray_.gpu() +
                                    ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            float *affine_matrix_device = (float *)preprocess_buffers_[ib]->gpu();
            float *image_based_bbox_output =
                bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
            checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
            decode_kernel_invoker(image_based_bbox_output, bbox_head_dims_[1],
                            num_classes_, bbox_head_dims_[2],
                            confidence_threshold_, nms_threshold_,
                            affine_matrix_device, boxarray_device,
                            type_, stream_);
        }
        checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(), output_boxarray_.gpu(),
                                    output_boxarray_.gpu_bytes(),
                                    cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaStreamSynchronize(stream_));

        std::vector<BoxArray> arrout(num_image);
        int imemory = 0;
        for (int ib = 0; ib < num_image; ++ib)
        {
            float *parray = output_boxarray_.cpu() +
                            ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            int count = std::min(MAX_IMAGE_BOXES, (int)*parray);
            BoxArray &output = arrout[ib];
            output.reserve(count);
            for (int i = 0; i < count; ++i)
            {
                float *pbox = parray + 1 + i * NUM_BOX_ELEMENT;
                int label = pbox[5];
                int keepflag = pbox[6];
                if (keepflag == 1)
                {
                    Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4],
                                         label);
                    if (has_segment_)
                    {
                        int row_index = pbox[7];
                        int mask_dim = segment_head_dims_[1];
                        float *mask_weights =
                            bbox_output_device +
                            (ib * bbox_head_dims_[1] + row_index) * bbox_head_dims_[2] +
                            num_classes_ + 4;

                        float *mask_head_predict = segment_predict_.gpu();
                        float left, top, right, bottom;
                        float *i2d = affine_matrixs[ib].i2d;
                        affine_project(i2d, pbox[0], pbox[1], &left, &top);
                        affine_project(i2d, pbox[2], pbox[3], &right, &bottom);

                        float box_width = right - left;
                        float box_height = bottom - top;
                        bool compress_to_bit = true;

                        float scale_to_predict_x =
                            segment_head_dims_[3] / (float)network_input_width_;
                        float scale_to_predict_y =
                            segment_head_dims_[2] / (float)network_input_height_;
                        int mask_out_width = box_width * scale_to_predict_x + 0.5f;
                        int mask_out_height = box_height * scale_to_predict_y + 0.5f;

                        if (mask_out_width > 0 && mask_out_height > 0)
                        {
                            if (imemory >= box_segment_cache_.size())
                            {
                                box_segment_cache_.push_back(
                                    std::make_shared<trt::Memory<unsigned char>>());
                            }

                            int bytes_of_mask_out = mask_out_width * mask_out_height;
                            auto box_segment_output_memory = box_segment_cache_[imemory];
                            result_object_box.seg = std::make_shared<InstanceSegmentMap>(
                                mask_out_width, mask_out_height);

                            unsigned char *mask_out_device =
                                box_segment_output_memory->gpu(bytes_of_mask_out);
                            unsigned char *mask_out_host = result_object_box.seg->data;
                            decode_single_mask(
                                left * scale_to_predict_x, top * scale_to_predict_y,
                                right * scale_to_predict_x, bottom * scale_to_predict_y,
                                mask_weights,
                                mask_head_predict +
                                    ib * segment_head_dims_[1] * segment_head_dims_[2] *
                                        segment_head_dims_[3],
                                segment_head_dims_[3], segment_head_dims_[2], mask_out_device,
                                mask_dim, mask_out_width, mask_out_height, stream_);
                            checkRuntime(
                                cudaMemcpyAsync(mask_out_host, mask_out_device,
                                                box_segment_output_memory->gpu_bytes(),
                                                cudaMemcpyDeviceToHost, stream_));
                        }
                    }
                    output.emplace_back(result_object_box);
                }
            }
        }

        if (has_segment_)
            checkRuntime(cudaStreamSynchronize(stream_));

        return arrout;
    }

    Infer *loadraw(const std::string &engine_file, Type type,
                float confidence_threshold, float nms_threshold)
    {
        InferImpl *impl = new InferImpl();
        if (!impl->load(engine_file, type, confidence_threshold, nms_threshold)) {
            delete impl;
            impl = nullptr;
        }
        return impl;
    }

    std::shared_ptr<Infer> load(const std::string &engine_file, Type type,
                        float confidence_threshold, float nms_threshold)
    {
        return std::shared_ptr<InferImpl>((InferImpl *)loadraw(
            engine_file, type, confidence_threshold, nms_threshold));
    }
}; // namespace yolo_detector
