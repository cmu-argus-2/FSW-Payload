#include "spdlog/spdlog.h"
#include "core/thread_pool.hpp"
#include <algorithm>


ThreadPool::ThreadPool(size_t num_threads) 
: 
num_threads(num_threads),
stop(false)
{
    SPDLOG_INFO("Creating ThreadPool with {} threads", num_threads);
    
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] { worker_thread(); });
    }
}

void ThreadPool::shutdown() 
{
    {
        std::unique_lock<std::mutex> lock(mtx_);
        if (stop) return; // if already stopped, return - TODO: check if this is necessary
        stop = true;  

        // discard all tasks
        while (!tasks.empty()) {
            tasks.pop();
        }
    } 

    cv_.notify_all();

    // Join all threads
    int i = 0;
    for (std::thread &worker : workers) 
    {
        if (worker.joinable()) {
            worker.join(); 
        }
        ++i;
        SPDLOG_INFO("Joined {} threads out of {}", i, num_threads);
    }

}

ThreadPool::~ThreadPool() 
{
    shutdown();
}


void ThreadPool::worker_thread() {
    while (true) {

        std::function<void()> task;

        // Extracting the task
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this] { return stop || !tasks.empty(); });

            if (stop)
            {
                break;
            }

            task = std::move(tasks.front());
            tasks.pop();
            ++busy_threads; /// atomic increment
        }

        // Execute task outside lock scope
        if (task) {
            task(); 
            // finished execution
            --busy_threads; /// atomic decrement
        }  

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


