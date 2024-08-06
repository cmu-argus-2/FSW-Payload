#ifndef TASK_HPP
#define TASK_HPP

#include <functional>
#include <chrono>
#include <vector>

#define MAX_ATTEMPTS 3
#define PRIORITY_LEVEL_1 1
#define PRIORITY_LEVEL_2 2
#define PRIORITY_LEVEL_3 3

// Forward declaration of Payload class
class Payload;


typedef std::function<void(Payload*, std::vector<uint8_t>)> CommandFunction;


class Task
{

public:
    Task(int task_id, CommandFunction func, std::vector<uint8_t> data, Payload* payload, int priority=0);
    // ~Task() = default; // TODO

    void Execute();
    int GetAttempts() const;
    int GetPriority() const;
    int GetID() const;
    std::chrono::system_clock::time_point GetCreationTime() const;

private:
    
    int task_id; // Unique identifier for the task.
    int priority; // Priority of the task.
    std::vector<uint8_t> data; // Data to be passed to the task's function.
    Payload* payload; // Payload object to be passed to the task's function.
    CommandFunction func; // task's function to be executed.
    int attempts; // number of execution attempts.
    std::chrono::system_clock::time_point created_at; // Timestamp for when the task was created.

};



#endif // TASK_HPP