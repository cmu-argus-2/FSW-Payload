#include <stdexcept>
#include "task.hpp"


Task::Task(int task_id, std::function<void(Payload*)> func, Payload* payload, int priority)
: 
task_id(task_id),
priority(0),
payload(payload),
func(func),
attempts(0)
{
    created_at = std::chrono::system_clock::now();
}

void Task::Execute() {
    attempts++;
    try {
        func(payload); // Try to execute the task
    } catch (const std::exception& e) {
        if (attempts < MAX_ATTEMPTS) {
            Execute();  // Retry logic
        } else {
            // Task exceeded retry limit
            // TODO: Log error or handle the exceeded retry case
        }
    }
}

int Task::GetAttempts() const 
{ 
    return attempts; 
}

int Task::GetPriority() const {
    return priority;
}

int Task::GetID() const {
    return task_id;
}

std::chrono::system_clock::time_point Task::GetCreationTime() const {
    return created_at;
}