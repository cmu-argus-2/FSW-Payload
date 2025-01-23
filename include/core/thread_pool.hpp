#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <future>


class ThreadPool 
{
public:

    explicit ThreadPool(size_t num_threads); // prevents implicit argument conversion 
    ~ThreadPool();

    void shutdown();

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> 
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<return_type> res = task->get_future();

        {
            std::unique_lock<std::mutex> lock(mtx_);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");

            tasks.emplace([task]() { (*task)(); });
        }

        cv_.notify_one();
        return res;
    }
    // return results

    // A busy thread is a thread that is currently executing a task
    size_t GetBusyThreadCount() const;
    // A waiting thread is a thread that is waiting for a task
    size_t GetWaitingThreadCount() const;

private:

    // Number of threads in the pool
    size_t num_threads; 

    // Stop flag for shutdown
    std::atomic<bool> stop{false};    

    void worker_thread();

    // Worker threads
    std::vector<std::thread> workers;                   

    // Task queue
    std::queue<std::function<void()>> tasks;  

    // Synchronize access to task queue
    std::mutex mtx_;                             

    // Notify worker threads of new tasks
    std::condition_variable cv_;                  

    // Tracks busy threads
    std::atomic<size_t> busy_threads{0};  

    // last execution times per thread ids
    std::unordered_map<std::thread::id, int64_t> last_exec_times_us;
    std::hash<std::thread::id> hasher;
                 
                                     
};

#endif // THREAD_POOL_HPP
