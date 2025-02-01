#ifndef TASK_HPP
#define TASK_HPP

#include <functional>
#include <chrono>
#include <vector>
#include <string_view>

#define MAX_ATTEMPTS 3

typedef std::function<void(std::vector<uint8_t>&)> CommandFunction;


class Task
{

public:
    Task(uint8_t task_id, CommandFunction func, std::vector<uint8_t>& data, uint8_t priority=0, std::string_view name="");
    // ~Task() = default; // TODO

    void Execute();
    int GetAttempts() const;
    uint8_t GetPriority() const;
    uint8_t GetID() const;
    size_t GetDataSize() const;
    std::chrono::system_clock::time_point GetCreationTime() const;

private:


    uint8_t task_id; // Unique identifier for the task.
    uint8_t priority; // Priority of the task.
    std::vector<uint8_t> data; // Data to be passed to the task's function.
    CommandFunction func; // task's function to be executed.
    std::chrono::system_clock::time_point created_at; // Timestamp for when the task was created.
    std::string name; // Name of the task.

};



#endif // TASK_HPP