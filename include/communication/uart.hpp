#include "comms.hpp"

#include <termios.h> // POSIX terminal control definitions
#include <unistd.h> 




class UART : public Communication
{

public:

    UART();

    bool Connect() override;
    void Disconnect() override;
    bool Receive(uint8_t& cmd_id, std::vector<uint8_t>& data) override;
    bool Send(const std::vector<uint8_t>& data) override;
    void RunLoop() override;
    void StopLoop() override;

private:

    struct termios _tty;
    int serial_port_fd = -1;


    bool port_opened = false;
    int failed_open_counter = 0;



    void OpenPort();
    void ConfigurePort();
    void ClearUpLink(); // flushes data received but not read and data written but not transmitted


}