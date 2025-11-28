#include "communication/tilepack.hpp"
#include "core/data_handling.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace {

std::string tmp_path(const std::string& filename) {
    return (std::filesystem::temp_directory_path() / filename).string();
}

} // namespace

TEST(DataHandling, FixedPacketRoundTrip) {
    std::vector<std::vector<uint8_t>> payloads = {
        {0x01, 0x02, 0x03},
        {0xAA, 0xBB, 0xCC, 0xDD, 0xEE}
    };

    const std::string path = tmp_path("dh_fixed_packet_roundtrip.bin");
    std::filesystem::remove(path);

    ASSERT_TRUE(DH::WriteFixedPacketFile(path, payloads));

    std::vector<std::vector<uint8_t>> read_payloads;
    ASSERT_TRUE(DH::ReadFixedPacketFile(path, read_payloads));

    EXPECT_EQ(read_payloads, payloads);

    const auto file_size = std::filesystem::file_size(path);
    const auto expected_size = static_cast<std::uintmax_t>(
        DH::DH_FILE_HEADER_SIZE + payloads.size() * DH::DH_FIXED_PACKET_SIZE);
    EXPECT_EQ(file_size, expected_size);

    std::filesystem::remove(path);
}

TEST(Tilepack, EncodeImageMatchesMainboardRadioFile) {
    using namespace tilepack;

    const auto base_path = std::filesystem::path(__FILE__).parent_path() / "tilepack_test";
    const auto image_path = base_path / "test_image.jpg";
    const auto reference_path = base_path / "image_radio_file_mainboard_dh.bin";
    const auto output_path = base_path / "image_radio_file_cpp.bin";

    ASSERT_TRUE(std::filesystem::exists(image_path)) << "Missing test image: " << image_path;
    ASSERT_TRUE(std::filesystem::exists(reference_path)) << "Missing reference radio file: " << reference_path;

    if (std::filesystem::exists(output_path)) {
        std::filesystem::remove(output_path);
    }

    TilepackEncoder encoder;
    ASSERT_TRUE(encoder.load_image(image_path.string()));

    const auto& packets = encoder.get_packets();
    ASSERT_FALSE(packets.empty());

    ASSERT_TRUE(encoder.write_radio_file(output_path.string()));

    std::vector<Packet> decoded;
    ASSERT_TRUE(tilepack::ReadPacketsFromDataHandlerFile(output_path.string(), decoded));

    ASSERT_EQ(decoded.size(), packets.size());
    for (std::size_t i = 0; i < packets.size(); ++i) {
        EXPECT_EQ(decoded[i].header.payload_size_bytes, packets[i].header.payload_size_bytes);
        EXPECT_EQ(decoded[i].header.page_id, packets[i].header.page_id);
        EXPECT_EQ(decoded[i].header.tile_idx, packets[i].header.tile_idx);
        EXPECT_EQ(decoded[i].header.frag_idx, packets[i].header.frag_idx);
        EXPECT_EQ(decoded[i].payload, packets[i].payload);
    }

    const auto file_size = std::filesystem::file_size(output_path);
    const auto expected_size = static_cast<std::uintmax_t>(
        DH::DH_FILE_HEADER_SIZE + packets.size() * DH::DH_FIXED_PACKET_SIZE);
    EXPECT_EQ(file_size, expected_size);

    ASSERT_EQ(file_size, std::filesystem::file_size(reference_path));

    auto read_file = [](const std::filesystem::path& p) {
        std::ifstream f(p, std::ios::binary);
        return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    };

    const auto generated_bytes = read_file(output_path);
    const auto reference_bytes = read_file(reference_path);
    ASSERT_EQ(generated_bytes.size(), reference_bytes.size());
    EXPECT_EQ(generated_bytes, reference_bytes);
}
