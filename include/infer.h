#ifndef __INFER_H_
#define __INFER_H_

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
#include "utils.h"

namespace  trt
{
    class Timer
    {
    private:
        void *start_, *stop_;
        void *stream_;
    public:
        Timer();
        virtual ~Timer();
        void start(void *stream = nullptr);
        float stop(const char* prefix = "Timer", bool print = true);
    };


    class BaseMemory
    {
    protected:
        void* cpu_ = nullptr;
        size_t cpu_bytes_ = 0;
        size_t cpu_capacity_ = 0;
        bool owner_cpu_ = true;

        void* gpu_ = nullptr;
        size_t gpu_bytes_ = 0;
        size_t gpu_capacity_ = 0;
        bool owner_gpu_ = true;
    public:
        BaseMemory() = default;
        BaseMemory(void* cpu, size_t cpu_bytes, void* gpu, size_t gpu_bytes);
        virtual ~BaseMemory();
        virtual void* cpu(size_t bytes);
        virtual void* gpu(size_t bytes);
        void release_cpu();
        void release_gpu();
        void release();
        inline bool owner_cpu() const { return owner_cpu_; }
        inline bool owner_gpu() const { return owner_gpu_; }
        inline size_t cpu_bytes() const { return cpu_bytes_; }
        inline size_t gpu_bytes() const { return gpu_bytes_; }
        virtual inline void* cpu() const { return cpu_; }
        virtual inline void* gpu() const { return gpu_; }
        void reference(void* cpu, size_t cpu_bytes, void* gpu, size_t gpu_bytes);
    };


    template <typename _DT>
    class Memory : public BaseMemory
    {
    private:
        /* data */
    public:
        Memory() = default;
        Memory(const Memory& other) = delete;
        Memory& operator=(const Memory& other) = delete;
        virtual _DT* gpu(size_t size) override
        {
            BaseMemory::gpu(size * sizeof(_DT));
        }
        virtual _DT* cpu(size_t size) override
        {
            BaseMemory::cpu(size * sizeof(_DT));
        }

        inline size_t cpu_size() const { return cpu_bytes_ / sizeof(_DT); }
        inline size_t gpu_size() const { return gpu_bytes_ / sizeof(_DT); }

        virtual inline _DT* cpu() const override { return (_DT*)cpu_; }
        virtual inline _DT* gpu() const override { return (_DT*)gpu_; }
    };


    /* class Infer*/
    class Infer
    {
    public:
        virtual bool forward(
                    const std::vector<void*>& bindings,
                    void* stream = nullptr,
                    void* input_consum_event = nullptr) = 0;
        virtual int index(const std::string& name) = 0;
        virtual std::vector<int> run_dims(const std::string& name) = 0;
        virtual std::vector<int> run_dims(int ibinding) = 0;
        virtual std::vector<int> static_dims(const std::string& name) = 0;
        virtual std::vector<int> static_dims(int ibinding) = 0;
        virtual int numel(const std::string& name) = 0;
        virtual int numel(int ibinding) = 0;
        virtual int num_bindings() = 0;
        virtual bool is_input(int ibinding) = 0;
        virtual bool set_run_dims(
                    const std::string& name,
                    const std::vector<int>& dims) = 0;
        virtual bool set_run_dims(int ibinding, const std::vector<int>& dims) = 0;
        virtual DType dtype(const std::string& name) = 0;
        virtual DType dtype(int ibinding) = 0;
        virtual bool has_dynamic_dim() = 0;
        virtual void print() = 0;
    };


    class __native_nvinfer_logger : public nvinfer1::ILogger
    {
    public:
        virtual void log(Severity severity, const char *msg) noexcept override
        {
            if (severity == Severity::kINTERNAL_ERROR)
            {
                INFO("NVInfer INTERNAL_ERROR: %s", msg);
                abort();
            }
            else if (severity == Severity::kERROR)
            {
                INFO("NVInfer: %s", msg);
            }
            // else  if (severity == Severity::kWARNING) {
            //     INFO("NVInfer: %s", msg);
            // }
            // else  if (severity == Severity::kINFO) {
            //     INFO("NVInfer: %s", msg);
            // }
            // else {
            //     INFO("%s", msg);
            // }
        }
    };
    static __native_nvinfer_logger gLogger;

    template <typename _T>
    static void destroy_nvidia_pointer(_T *ptr)
    {
        if (ptr)
            ptr->destroy();
    }

    /* class __native_engine_context */
    class __native_engine_context
    {
    private:
        void destroy();
    public:
        __native_engine_context();
        virtual ~__native_engine_context();
        
        bool construct(const void* pdata, size_t size)
        {
            destroy();

            if (pdata == nullptr || size == 0)
            {
                return false;
            }

            runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
                nvinfer1::createInferRuntime(gLogger),
                destroy_nvidia_pointer<nvinfer1::IRuntime>
            );

            if (runtime_ == nullptr)
            {
                return false;
            }

            engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
                runtime_->deserializeCudaEngine(pdata, size, nullptr),
                destroy_nvidia_pointer<nvinfer1::ICudaEngine>
            );

            if (engine_ == nullptr)
            {
                return false;
            }
            
            context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
                engine_->createExecutionContext(),
                destroy_nvidia_pointer<nvinfer1::IExecutionContext>
            );
            
            return context_ != nullptr;
        }

    public:
        std::shared_ptr<nvinfer1::IExecutionContext> context_;
        std::shared_ptr<nvinfer1::ICudaEngine> engine_;
        std::shared_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
    };



    class InferImpl : public Infer
    {
    private:
        /* data */
    public:
        std::shared_ptr<__native_engine_context> context_;
        std::unordered_map<std::string, int> binding_name_to_index_;

        bool construct(const void* data, size_t size);
        bool load(const std::string& file);
        void setup();
        virtual int index(const std::string& name);
        virtual bool forward(const std::vector<void*>& bindings, void* stream, void* input_consum_event) override;
        virtual std::vector<int> run_dims(const std::string& name) override;
        virtual std::vector<int> run_dims(int ibinding) override;
        virtual std::vector<int> static_dims(const std::string& name) override;
        virtual std::vector<int> static_dims(int ibinding) override;
        virtual int num_bindings() override;
        virtual bool is_input(int ibinding) override;
        virtual bool set_run_dims(const std::string& name, const std::vector<int>& dims) override;
        virtual bool set_run_dims(int ibinding, const std::vector<int>& dims) override;
        virtual int numel(const std::string& name) override;
        virtual int numel(int ibinding) override;
        virtual DType dtype(const std::string& name) override;
        virtual DType dtype(int ibinding) override;
        virtual bool has_dynamic_dim() override;
        virtual void print() override;
    };


    Infer* loadraw(const std::string &file);
    std::shared_ptr<Infer> load(const std::string& file);
    std::string format_shape(const std::vector<int>& shape);
}; // namespace trt

#endif // __INFER_H_