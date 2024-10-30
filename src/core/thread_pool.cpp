#include "core/thread_pool.hpp"
#include <algorithm>


ThreadPool::ThreadPool(size_t num_threads) 
: 
num_threads(num_threads),
stop(false)
{
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] { worker_thread(); });
    }
}

ThreadPool::~ThreadPool() 
{
    // atomic set to true
    stop = true;

    cv_.notify_all();

    // Join all threads
    for (std::thread &worker : workers) {
        worker.join();
    }
}


void ThreadPool::worker_thread() {
    while (true) {

        std::function<void()> task;

        // Extracting the task
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this] { return stop || !tasks.empty(); });

            if (stop) return;

            task = std::move(tasks.front());
            tasks.pop();
            ++busy_threads; /// atomic increment
        }

        task();
        // finished execution

        --busy_threads; /// atomic decrement

    }
}

size_t ThreadPool::GetBusyThreadCount() const 
{
    return busy_threads.load();
}

size_t ThreadPool::GetWaitingThreadCount() const 
{
    return num_threads - busy_threads.load();
}


