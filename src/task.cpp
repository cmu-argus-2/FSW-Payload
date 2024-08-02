#include <stdexcept>
#include "task.hpp"


Task::Task(int task_id, std::function<void(Payload*)> func, Payload* payload)
: 
task_id(task_id),
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