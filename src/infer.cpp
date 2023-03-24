#include "../include/infer.h"

namespace trt
{
    static std::vector<uint8_t> load_file(const std::string &file)
    {
        std::ifstream in(file, std::ios::in | std::ios::binary);
        if (!in.is_open())
            return {};

        in.seekg(0, std::ios::end);
        size_t length = in.tellg();

        std::vector<uint8_t> data;
        if (length > 0)
        {
            in.seekg(0, std::ios::beg);
            data.resize(length);

            in.read((char *)&data[0], length);
        }
        in.close();
        return data;
    }

    /* class __native_engine_context*/
    void __native_engine_context::destroy()
    {
        context_.reset();
        engine_.reset();
        runtime_.reset();
    }

    __native_engine_context::__native_engine_context() {}
    __native_engine_context::~__native_engine_context() { destroy(); }


    static std::string format_shape(const nvinfer1::Dims& shape)
    {
        std::stringstream output;
        char buf[64];
        const char* fmts[] = {"%d", "x%d"};
        for (int i = 0; i < shape.nbDims; i++)
        {
            snprintf(buf, sizeof(buf), fmts[i != 0], shape.d[i]);
            output << buf;
        }
        return output.str();
    }

    /* class InferImpl */
    bool InferImpl::construct(const void *data, size_t size)
    {
        context_ = std::make_shared<__native_engine_context>();
        if (!context_->construct(data, size))
        {
            return false;
        }

        setup();
        return true;
    }

    bool InferImpl::load(const std::string &file)
    {
        auto data = load_file(file);
        if (data.empty())
        {
            INFO("An empty file has been loaded. Please confirm your file path: %s",
                file.c_str());
            return false;
        }
        return this->construct(data.data(), data.size());
    }

    void InferImpl::setup()
    {
        auto engine = this->context_->engine_;
        int nbBindings = engine->getNbBindings();

        binding_name_to_index_.clear();
        for (int i = 0; i < nbBindings; ++i)
        {
            const char *bindingName = engine->getBindingName(i);
            binding_name_to_index_[bindingName] = i;
        }
    }

    int InferImpl::index(const std::string &name)
    {
        auto iter = binding_name_to_index_.find(name);
        Assertf(iter != binding_name_to_index_.end(),
                "Can not found the binding name: %s", name.c_str());
        return iter->second;
    }

    bool InferImpl::forward(const std::vector<void *> &bindings, void *stream,
                void *input_consum_event)
    {
        return this->context_->context_->enqueueV2(
            bindings.data(), (cudaStream_t)stream,
            (cudaEvent_t *)input_consum_event
        );
    }

    std::vector<int> InferImpl::run_dims(const std::string &name)
    {
        return run_dims(index(name));
    }

    std::vector<int> InferImpl::run_dims(int ibinding)
    {
        auto dim = this->context_->context_->getBindingDimensions(ibinding);
        return std::vector<int>(dim.d, dim.d + dim.nbDims);
    }

    std::vector<int> InferImpl::static_dims(const std::string &name)
    {
        return static_dims(index(name));
    }

    std::vector<int> InferImpl::static_dims(int ibinding)
    {
        auto dim = this->context_->engine_->getBindingDimensions(ibinding);
        return std::vector<int>(dim.d, dim.d + dim.nbDims);
    }

    int InferImpl::num_bindings()
    {
        return this->context_->engine_->getNbBindings();
    }

    bool InferImpl::is_input(int ibinding)
    {
        return this->context_->engine_->bindingIsInput(ibinding);
    }

    bool InferImpl::set_run_dims(const std::string &name,
                                const std::vector<int> &dims)
    {
        return this->set_run_dims(index(name), dims);
    }

    bool InferImpl::set_run_dims(int ibinding,
                                const std::vector<int> &dims)
    {
        nvinfer1::Dims d;
        memcpy(d.d, dims.data(), sizeof(int) * dims.size());
        d.nbDims = dims.size();
        return this->context_->context_->setBindingDimensions(ibinding, d);
    }

    int InferImpl::numel(const std::string &name)
    {
        return numel(index(name));
    }

    int InferImpl::numel(int ibinding)
    {
        auto dim = this->context_->context_->getBindingDimensions(ibinding);
        return std::accumulate(dim.d, dim.d + dim.nbDims, 1,
                            std::multiplies<int>());
    }

    DType InferImpl::dtype(const std::string &name)
    {
        return dtype(index(name));
    }

    DType InferImpl::dtype(int ibinding)
    {
        return (DType) this->context_->engine_->getBindingDataType(ibinding);
    }

    bool InferImpl::has_dynamic_dim()
    {
        // check if any input or output bindings have dynamic shapes
        // code from ChatGPT
        int numBindings = this->context_->engine_->getNbBindings();
        for (int i = 0; i < numBindings; ++i)
        {
            nvinfer1::Dims dims = this->context_->engine_->getBindingDimensions(i);
            for (int j = 0; j < dims.nbDims; ++j)
            {
                if (dims.d[j] == -1)
                return true;
            }
        }
        return false;
    }

    void InferImpl::print()
    {
        INFO("Infer %p [%s]", this,
            has_dynamic_dim() ? "DynamicShape" : "StaticShape");

        int num_input = 0;
        int num_output = 0;
        auto engine = this->context_->engine_;
        for (int i = 0; i < engine->getNbBindings(); ++i)
        {
            if (engine->bindingIsInput(i))
                num_input++;
            else
                num_output++;
        }

        INFO("Inputs: %d", num_input);
        for (int i = 0; i < num_input; ++i)
        {
            auto name = engine->getBindingName(i);
            auto dim = engine->getBindingDimensions(i);
            INFO("\t%d.%s : shape {%s}", i, name, format_shape(dim).c_str());
        }

        INFO("Outputs: %d", num_output);
        for (int i = 0; i < num_output; ++i)
        {
            auto name = engine->getBindingName(i + num_input);
            auto dim = engine->getBindingDimensions(i + num_input);
            INFO("\t%d.%s : shape {%s}", i, name, format_shape(dim).c_str());
        }
    }

    /* class Timer*/
    Timer::Timer()
    {
        checkRuntime(cudaEventCreate((cudaEvent_t *)&start_));
        checkRuntime(cudaEventCreate((cudaEvent_t *)&stop_));
    }

    Timer::~Timer()
    {
        checkRuntime(cudaEventDestroy((cudaEvent_t)start_));
        checkRuntime(cudaEventDestroy((cudaEvent_t)stop_));
    }

    void Timer::start(void* stream)
    {
        stream_ = stream;
        checkRuntime(cudaEventRecord((cudaEvent_t)start_, (cudaStream_t)stream_));
    }

    float Timer::stop(const char *prefix, bool print)
    {
        checkRuntime(cudaEventRecord((cudaEvent_t)stop_, (cudaStream_t)stream_));
        checkRuntime(cudaEventSynchronize((cudaEvent_t)stop_));

        float latency = 0;
        checkRuntime(
            cudaEventElapsedTime(&latency, (cudaEvent_t)start_, (cudaEvent_t)stop_));

        if (print) {
            printf("[%s]: %.5f ms\n", prefix, latency);
        }

        return latency;
    }

    /* class BaseMemory */
    void BaseMemory::release_cpu()
    {
        if (cpu_)
        {
            if (owner_cpu_)
            {
                checkRuntime(cudaFreeHost(cpu_));
            }
            cpu_ = nullptr;
        }
        cpu_capacity_ = 0;
        cpu_bytes_ = 0;
    }

    void BaseMemory::release_gpu()
    {
        if (gpu_)
        {
            if (owner_gpu_)
            {
                checkRuntime(cudaFree(gpu_));
            }
            gpu_ = nullptr;
        }
        gpu_capacity_ = 0;
        gpu_bytes_ = 0;
    }

    void BaseMemory::release()
    {
        release_cpu();
        release_gpu();
    }

    void BaseMemory::reference(void* cpu, size_t cpu_bytes, void* gpu, size_t gpu_bytes)
    {
        release();

        if (cpu == nullptr || cpu_bytes == 0)
        {
            cpu = nullptr;
            cpu_bytes = 0;
        }

        if (gpu == nullptr || gpu_bytes == 0)
        {
            gpu = nullptr;
            gpu_bytes = 0;
        }
        
        this->cpu_ = cpu;
        this->cpu_capacity_ = cpu_bytes;
        this->cpu_bytes_ = cpu_bytes;
        this->gpu_ = gpu;
        this->gpu_capacity_ = gpu_bytes;
        this->gpu_bytes_ = gpu_bytes;

        this->owner_cpu_ = !(cpu && cpu_bytes > 0);
        this->owner_gpu_ = !(gpu && gpu_bytes > 0);
    }

    BaseMemory::BaseMemory(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes)
    {
        reference(cpu, cpu_bytes, gpu, gpu_bytes);
    }

    BaseMemory::~BaseMemory() { release(); }

    void* BaseMemory::cpu(size_t bytes)
    {
        if (cpu_capacity_ < bytes)
        {
            release_cpu();

            cpu_capacity_ = bytes;
            checkRuntime(cudaMallocHost(&cpu_, bytes));
            Assert(cpu_ != nullptr);
        }
        cpu_bytes_ = bytes;

        return cpu_;
    }

    void* BaseMemory::gpu(size_t bytes)
    {
        if (gpu_capacity_ < bytes)
        {
            release_gpu();

            gpu_capacity_ = bytes;
            checkRuntime(cudaMalloc(&gpu_, bytes));
        }
        gpu_bytes_ = bytes;

        return gpu_;
    }

    Infer *loadraw(const std::string &file)
    {
        InferImpl *impl = new InferImpl();
        if (!impl->load(file))
        {
            delete impl;
            impl = nullptr;
        }
        return impl;
    }

    std::shared_ptr<Infer> load(const std::string &file)
    {
        return std::shared_ptr<InferImpl>((InferImpl *)loadraw(file));
    }

    std::string format_shape(const std::vector<int> &shape)
    {
        std::stringstream output;
        char buf[64];
        const char *fmts[] = {"%d", "x%d"};
        for (int i = 0; i < shape.size(); ++i)
        {
            snprintf(buf, sizeof(buf), fmts[i != 0], shape[i]);
            output << buf;
        }
        return output.str();
    }
}; // namespace trt
