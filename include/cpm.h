#ifndef __CPM_H__
#define __CPM_H__

// Consumer Producer Model
#include <algorithm>
#include <condition_variable>
#include <future>
#include <memory>
#include <queue>
#include <thread>

namespace cpm
{
    // Consumer Producer Model 模板类定义
    template <typename Result, typename Input, typename Model>
    class Instance
    {
    protected:
        // 内部结构体定义，表示一个输入和其对应的 Promise 对象
        struct Item
        {
            Input input; // 输入数据
            std::shared_ptr<std::promise<Result>> pro; // 结果承诺
        };

        // 定义条件变量和互斥锁，用于线程同步
        std::condition_variable cond_; // 用于在输入数据可用于处理时通知工作线程的条件变量
        std::queue<Item> input_queue_; // 等待处理的输入数据队列
        std::mutex queue_lock_; // 用于锁定对输入队列的访问的互斥量

        // 定义 worker 线程和运行状态
        std::shared_ptr<std::thread> worker_; // 指向工作线程的共享指针
        volatile bool run_ = false; // 一个布尔标志，指示工作线程是否正在运行

        // 最大处理数和数据流指针
        volatile int max_items_processed_ = 0; // 工作线程在单次迭代中可以处理的最大输入项数
        void* stream_ = nullptr; // 一个空指针，可用于将附加数据传递给模型

    public:
        // 析构函数，停止 worker 线程并清空 input_queue_
        virtual ~Instance() { stop(); }

        // 停止工作线程并清除输入队列
        // 它还将与队列中输入项关联的任何承诺的值设置为结果类型的默认值
        void stop()
        {
            run_ = false; // // 标志位设为false，表示线程要停止工作
            cond_.notify_one();

            // 依次处理 input_queue_ 中的每个输入并设置 Promise 对象的值为空
            {
                std::unique_lock<std::mutex> l(queue_lock_);
                while (!input_queue_.empty())
                {
                    auto &item = input_queue_.front();
                    if (item.pro)
                    {
                        item.pro->set_value(Result());
                    }
                    input_queue_.pop();
                }
            }

            // 等待 worker 线程结束并重置线程对象
            if (worker_)
            {
                worker_->join();
                worker_.reset();
            }
        }

        // 提交一条输入数据，并返回一个future对象
        // 将单个输入项添加到输入队列并返回可用于检索计算结果的共享future
        virtual std::shared_future<Result> commit(const Input& input)
        {
            Item item;
            item.input = input;
            item.pro.reset(new std::promise<Result>());

            // 将输入和 Promise 对象打包成 Item 对象，加入到 input_queue_ 中
            {
                std::unique_lock<std::mutex> __lock_(queue_lock_);
                input_queue_.push(item);
            }

            // 通知等待条件变量的线程
            cond_.notify_one(); // 唤醒等待在条件变量上的线程
            return item.pro->get_future();
        }

        // 提交多个输入并返回对应的 Future 对象的 vector
        // 将多个输入项添加到输入队列并返回可用于检索计算结果的共享 future
        virtual std::vector<std::shared_future<Result>> commits(const std::vector<Input>& inputs)
        {
            std::vector<std::shared_future<Result>> output;
            {
                // 将每个输入和对应的 Promise 对象打包成 Item 对象，加入到 input_queue_ 中
                std::unique_lock<std::mutex> __lock_(queue_lock_);
                for (int i = 0; i < inputs.size(); i++)
                {
                    Item item;
                    item.input = inputs[i];
                    item.pro.reset(new std::promise<Result>());
                    output.emplace_back(item.pro->get_future());
                    input_queue_.push(item);
                }
            }
            cond_.notify_one(); // 唤醒等待在条件变量上的线程
            return output;
        }

        // 启动工作线程并使用提供的加载方法加载指定的模型
        // max_items_processed 参数指定工作线程在单次迭代中可以处理的最大输入项数
        // 流参数是一个空指针，可用于将附加数据传递给模型
        template <typename LoadMethod>
        bool start(const LoadMethod& loadmethod, int max_items_processed = 1, void* stream = nullptr)
        {
            stop();

            this->stream_ = stream;
            this->max_items_processed_ = max_items_processed;
            std::promise<bool> status;
            worker_ = std::make_shared<std::thread>(
                &Instance::worker<LoadMethod>, this, std::ref(loadmethod), std::ref(status)
            );
            return status.get_future().get();
        }

    private:
        // 此函数从输入队列中检索指定数量的项目，
        // 如有必要，等待队列包含至少一项。
        // 如果工作线程应该终止（表明
        // `run_` 标志为假）。
        virtual bool get_items_and_wait(std::vector<Item>& fetch_items, int max_size)
        {
            // 获取输入队列的锁
            std::unique_lock<std::mutex> l(queue_lock_);
            // 等待输入队列不为空或工作线程收到停止信号
            cond_.wait(l, [&]() {return !run_ || !input_queue_.empty(); });

            // 如果工作线程收到停止信号，则返回 false
            if (!run_)
                return false;

            // 从输入队列中检索项目并将它们添加到 `fetch_items` 向量中
            fetch_items.clear();
            for (int i = 0; i < max_size && !input_queue_.empty(); i++)
            {
                fetch_items.emplace_back(std::move(input_queue_.front()));
                input_queue_.pop();
            }

            // 返回 true 以指示项目已被检索
            return true;
        }

        // 此函数类似于 `get_items_and_wait`，但一次只检索一个项目
        virtual bool get_item_and_wait(Item& fetch_item)
        {
            // 获取输入队列的锁
            std::unique_lock<std::mutex> l(queue_lock_);
            // 等待输入队列不为空或工作线程收到停止信号
            cond_.wait(l, [&]() { return !run_ || !input_queue_.empty(); });

            // 如果工作线程收到停止信号，则返回 false
            if (run_)
                return false;

            // 从输入队列中检索下一个项目并返回 true 表示成功
            fetch_item = std::move(input_queue_.front());
            input_queue_.pop();

            return true;
        }


        // 这是一个模板函数，它接受一个 LoadMethod 类型的函数对象
        // 和一个通过引用的 promise 对象。该函数使用 LoadMethod 加载模型，
        // 使用模型处理输入项，并用结果履行承诺。
        template <typename LoadMethod>
        void worker(const LoadMethod& loadmethod, std::promise<bool>& status)
        {
            // 创建指向 Model 对象的共享指针并调用 LoadMethod 加载模型。
            std::shared_ptr<Model> model = loadmethod();
            if (model == nullptr)
            {
                // 如果模型加载失败，将 promise 的值设置为 false 并返回。
                status.set_value(false);
                return;
            }

            // 将 run_ 标志设置为 true 并将 promise 的值设置为 true 以指示成功。
            run_ = true;
            status.set_value(true);

            // 初始化向量以存储获取的项目和输入以进行推理。
            std::vector<Item> fetch_items;
            std::vector<Input> inputs;

            // 进入一个循环，获取一批项目并对它们进行推理。
            while (get_items_and_wait(fetch_items, max_items_processed_))
            {
                // 从获取的项目中提取输入并将它们存储在输入向量中。
                inputs.resize(fetch_items.size());
                std::transform(
                    fetch_items.begin(), fetch_items.end(), inputs.begin(),
                    [](Item& item) { return item.input; }
                );

                // 使用模型和给定的流对输入执行前向推理。
                auto ret = model->forwards(inputs, stream_);

                // 用相应的推理结果履行与每个获取的项目相关联的承诺。
                for (int i = 0; i < fetch_items.size(); i++)
                {
                    if (i < ret.size())
                    {
                        fetch_items[i].pro->set_value(ret[i]);
                    }
                    else
                    {
                        fetch_items[i].pro->set_value(Result());
                    }                    
                }

                // 清除输入和获取的项目向量，为下一批项目做准备。
                inputs.clear();
                fetch_items.clear();
            }

            // 将共享指针重置为模型并将 run_ 标志设置为 false。
            model.reset();
            run_ = false;
        }
    };
};

#endif // __CPM_H__