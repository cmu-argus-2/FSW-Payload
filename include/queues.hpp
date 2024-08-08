#ifndef QUEUES_HPP
#define QUEUES_HPP

#include <queue>
#include <mutex>
#include <atomic>
#include "task.hpp"

class RX_Queue
{   

public:
    RX_Queue();


    void AddTask(const Task& task);
    Task GetNextTask();


    void Pause();
    void Resume();
    bool IsEmpty() const;
    size_t Size() const;
    void Clear();
    void PrintAllTasks() const;



private:

    struct TaskComparator {
        bool operator()(const Task& lhs, const Task& rhs) const 
        {
            // Higher priority first then earlier creation time as a tiebreaker
            if (lhs.GetPriority() == rhs.GetPriority()) {
                return lhs.GetCreationTime() > rhs.GetCreationTime(); // Earlier tasks have higher priority
            }
            return lhs.GetPriority() < rhs.GetPriority(); // Higher priority value means higher priority
        }
    };

    std::atomic<bool> paused;
    std::priority_queue<Task, std::vector<Task>, TaskComparator> task_queue;
    std::mutex queue_mutex;

};

class TX_Queue
{
public:
    TX_Queue();

    // void AddMsg();
    void Pause();
    void Resume();
    bool IsEmpty() const;
    size_t Size() const;
    void Clear();

};



#endif // QUEUES_HPP