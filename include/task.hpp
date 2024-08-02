#ifndef TASK_HPP
#define TASK_HPP

#include <functional>
#include <chrono>

#define MAX_ATTEMPTS 3

// Forward declaration of Payload class
class Payload;

class Task
{

public:
    Task(int task_id, std::function<void(Payload*)> func, Payload* payload);
    ~Task(); // TODO

    void Execute();
    int GetAttempts() const;

private:
    
    int task_id; // Unique identifier for the task.
    Payload* payload; // Payload object to be passed to the task's function.
    std::function<void(Payload*)> func; // task's function to be executed.
    int attempts; // number of execution attempts.
    std::chrono::system_clock::time_point created_at; // Timestamp for when the task was created.

};



#endif // TASK_HPP