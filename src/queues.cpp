#include "spdlog/spdlog.h"

#include "queues.hpp"
#include <iostream>




// RX_Queue 

RX_Queue::RX_Queue() 
: paused(false)
{
}


void RX_Queue::AddTask(const Task& task) 
{
    if (!paused) {
        task_queue.push(task);
        SPDLOG_INFO("Adding command with ID: {}",task.GetID());
    } else
    {
        SPDLOG_INFO("INFO: Queue is paused. Task not added.");
    }
}

Task RX_Queue::GetNextTask() 
{
    if (task_queue.empty()) {
        SPDLOG_WARN("Queue is empty");
        throw std::runtime_error("Queue is empty");
    }

    Task task = task_queue.top();
    SPDLOG_INFO("Getting command with ID: {}",task.GetID());
    task_queue.pop();
    // SPDLOG_INFO("Getting command with ID (after pop): {}",task.GetID());
    
    return task;
}


void RX_Queue::Pause() {
    paused = true;
}

void RX_Queue::Resume() {
    paused  = false;
}

bool RX_Queue::IsEmpty() const {
    return task_queue.empty();
}

size_t RX_Queue::Size() const {
    return task_queue.size();
}

void RX_Queue::Clear() {
    // Clear the queue by swapping with an empty queue (goes out of scope and calls destructor)
    std::priority_queue<Task, std::vector<Task>, TaskComparator> empty;
    std::swap(task_queue, empty);
}

void RX_Queue::PrintAllTasks() const {
    // Copy the priority queue to iterate without modifying the original queue
    auto copy = task_queue;
    while (!copy.empty()) {
        const Task& task = copy.top();
        std::cout << "Task ID: " << task.GetID()
                  << ", Priority: " << task.GetPriority()
                  << ", Argument Present(s): " << ((task.GetDataSize() > 0) ? "Yes" : "No") << std::endl; // TODO: Implement this
        copy.pop();
    }
}

// TX_Queue 

TX_Queue::TX_Queue() 
{}