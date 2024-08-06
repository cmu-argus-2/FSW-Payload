#include "spdlog/spdlog.h"

#include "queues.hpp"
#include <iostream>




// RX_Queue 

RX_Queue::RX_Queue() 
: paused(false)
{
}


void RX_Queue::AddTask(const Task& task) {
    if (!paused) {
        task_queue.push(task);
    } else
    {
        SPDLOG_INFO("INFO: Queue is paused. Task not added.");
    }
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
        copy.pop();
        std::cout << "Task ID: " << task.GetID()
                  << ", Priority: " << task.GetPriority()
                  << ", Argument Present(s): " << (false ? "Yes" : "No") << std::endl; // TODO: Implement this
    }
}

// TX_Queue 

TX_Queue::TX_Queue() 
{}