#include <vision/ld.hpp>

float ComputeIoU(const Landmark& a, const Landmark& b)
{
    const float ax1 = a.x - 0.5f * a.width;
    const float ay1 = a.y - 0.5f * a.height;
    const float ax2 = a.x + 0.5f * a.width;
    const float ay2 = a.y + 0.5f * a.height;

    const float bx1 = b.x - 0.5f * b.width;
    const float by1 = b.y - 0.5f * b.height;
    const float bx2 = b.x + 0.5f * b.width;
    const float by2 = b.y + 0.5f * b.height;

    const float inter_x1 = std::max(ax1, bx1);
    const float inter_y1 = std::max(ay1, by1);
    const float inter_x2 = std::min(ax2, bx2);
    const float inter_y2 = std::min(ay2, by2);

    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;

    const float area_a = std::max(0.0f, a.width) * std::max(0.0f, a.height);
    const float area_b = std::max(0.0f, b.width) * std::max(0.0f, b.height);
    const float union_area = area_a + area_b - inter_area;

    return inter_area / (union_area + 1e-6f);
}