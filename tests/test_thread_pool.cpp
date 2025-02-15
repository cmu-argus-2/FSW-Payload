#include "core/thread_pool.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <vector>


// Fixture for ThreadPool 
class ThreadPoolTest : public ::testing::Test 
{
protected:
    void SetUp() override 
    {
        pool = new ThreadPool(4); // Initialize a pool with 4 threads
    }

    void TearDown() override 
    {
        delete pool;
    }

    ThreadPool* pool;
};

// Basic Task Execution
TEST_F(ThreadPoolTest, BasicTaskExecution) 
{
    std::atomic<int> counter{0};
    for (int i = 0; i < 10; ++i) 
    {
        pool->enqueue([&counter] {
            counter++;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
    }
    // Wait for tasks to finish
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_EQ(counter.load(), 10); // each thread increments counter by 1
}

// Return value from task execution and get them
TEST_F(ThreadPoolTest, TaskReturnValue) 
{
    std::vector<std::future<int>> results;
    for (int i = 0; i < 5; ++i) 
    {
        results.emplace_back(pool->enqueue([i] {
            return i * i;
        }));
    }
    for (long unsigned int i = 0; i < 5; ++i) 
    {
        EXPECT_EQ(results[i].get(), i * i);
    }
}

// Counting busy and waiting threads
TEST_F(ThreadPoolTest, ThreadCountCheck) 
{
    // Enqueue tasks that run long enough to measure thread activity
    std::vector<std::future<void>> tasks;
    for (int i = 0; i < 4; ++i) 
    {
        tasks.emplace_back(pool->enqueue([] 
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }));
    }
    // Wait for threads to pick up tasks
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_EQ(pool->GetBusyThreadCount(), 4);
    EXPECT_EQ(pool->GetWaitingThreadCount(), 0);

    // Wait until tasks finish and recheck
    for (auto& task : tasks) 
    {
        task.get();
    }
    EXPECT_EQ(pool->GetBusyThreadCount(), 0);
    EXPECT_EQ(pool->GetWaitingThreadCount(), 4);
}

// Shutdown 
TEST_F(ThreadPoolTest, Shutdown) 
{
    delete pool; // Pool should shut down here

    // initialize a new pool and enqueue tasks to confirm it starts cleanly
    pool = new ThreadPool(2);
    std::atomic<int> counter{0};
    for (int i = 0; i < 3; ++i) 
    {
        pool->enqueue([&counter] { counter++; });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(counter.load(), 3);
}

// Large number of tasks to execute
TEST_F(ThreadPoolTest, LargeNumberOfTasks) {
    std::atomic<int> counter{0};
    for (int i = 0; i < 100; ++i) 
    {
        pool->enqueue([&counter] 
        {
            counter++;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(600)); // Wait to complete
    EXPECT_EQ(counter.load(), 100);
}

