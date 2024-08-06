#include <stdexcept>
#include "task.hpp"


Task::Task(int task_id, CommandFunction func, std::vector<uint8_t> data, Payload* payload, int priority)
: 
task_id(task_id),
priority(0),
data(data),
payload(payload),
func(func),
attempts(0)
{
    created_at = std::chrono::system_clock::now();

    // TODO check task_id mapping again
}

void Task::Execute() {
    attempts++;
    try {
        func(payload, data); // Try to execute the task
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