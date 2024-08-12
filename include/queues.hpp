#ifndef QUEUES_HPP
#define QUEUES_HPP

#include <queue>
#include <mutex>
#include <atomic>
#include "task.hpp"
#include "messages.hpp"

#define RX_PRIORITY_1 1
#define RX_PRIORITY_2 2
#define RX_PRIORITY_3 3

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
                return lhs.GetCreationTime() > rhs.GetCreationTime();
            }
            return lhs.GetPriority() < rhs.GetPriority(); 
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

    void AddMsg(std::shared_ptr<Message> msg);
    std::shared_ptr<Message> GetNextMsg();

    void Pause();
    void Resume();
    bool IsEmpty() const;
    size_t Size() const;
    void Clear();


private:

    struct MsgComparator {
        bool operator()(const std::shared_ptr<Message>& lhs, const std::shared_ptr<Message>& rhs) const 
        {
            // Higher priority first, then earlier creation time as a tiebreaker
            if (lhs->priority == rhs->priority) {
                return lhs->created_at > rhs->created_at; 
            }
            return lhs->priority < rhs->priority; 
        }
    };



    std::atomic<bool> paused;
    std::priority_queue<std::shared_ptr<Message>, std::vector<std::shared_ptr<Message>>, MsgComparator> msg_queue;
    std::mutex queue_mutex;

};



#endif // QUEUES_HPP