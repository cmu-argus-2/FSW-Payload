#include "spdlog/spdlog.h"
#include "core/thread_pool.hpp"
#include <algorithm>
#include <chrono>

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


void ThreadPool::worker_thread() 
{
    last_exec_times_us[std::this_thread::get_id()] = 0;
    
    while (!stop) 
    {

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
        if (task) 
        {
            auto start = std::chrono::high_resolution_clock::now();
            task(); 
            auto end = std::chrono::high_resolution_clock::now();
            // finished execution
            --busy_threads; // atomic decrement
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            last_exec_times_us[std::this_thread::get_id()] = duration.count();
            SPDLOG_INFO("Task executed in {} μs from thread id (hash) {}.", duration.count(), hasher(std::this_thread::get_id()));
        }  

    }

    SPDLOG_INFO("Exiting ThreadPool worker thread");

}

size_t ThreadPool::GetBusyThreadCount() const 
{
    return busy_threads.load();
}

size_t ThreadPool::GetWaitingThreadCount() const 
{
    return num_threads - busy_threads.load();
}


