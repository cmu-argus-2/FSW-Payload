#include <gtest/gtest.h>
#include <core/errors.hpp>

// Jsut common sense tests

// Test that the circular buffer does not exceed its maximum size
TEST(ErrorLogTest, CircularBufferSizeLimit) 
{
    ClearErrors();
    // Fill beyond capacity
    for (std::size_t i = 0; i < GetMaxErrorBufferSize() + 5; ++i) 
    {
        LogError(static_cast<EC>(static_cast<int>(EC::UNDEFINED)-1)); // get just the latest defined error
    }
    EXPECT_EQ(GetCurrentErrorCount(), GetMaxErrorBufferSize());
}

TEST(ErrorLogTest, LogAndRetrieve) 
{
    ClearErrors();
    EC err = static_cast<EC>(static_cast<int>(EC::UNDEFINED) - 1); // get just the latest defined error
    LogError(err);
    EXPECT_EQ(GetLastError(), err);
}

TEST(ErrorLogTest, OKNotLogged) 
{
    ASSERT_DEATH(LogError(EC::OK), "Can't log OK as an error.");
}


TEST(ErrorLogTest, ErrorCounting) 
{
    ClearErrors();
    EC err = static_cast<EC>(static_cast<int>(EC::UNDEFINED) - 1); // get just the latest defined error
    LogError(err);
    LogError(err);
    EXPECT_EQ(GetCurrentErrorCount(), 2);
}

// Test invalid error codes trigger assertion
TEST(ErrorLogTest, InvalidErrorCode) 
{
    ClearErrors();
    EC invalid_error = static_cast<EC>(static_cast<int>(EC::UNDEFINED) + 1);
    ASSERT_DEATH(LogError(invalid_error), "Error code doesn't exist.");
}