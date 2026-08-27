// Unit tests for corner ordering and temporal matching (ROC convention:
// GoogleTest for C++ concerns).
#include <gtest/gtest.h>

#include "ibvs_perception/corner_matching.hpp"

using ibvs_perception::matchCornersTemporal;
using ibvs_perception::orderCorners;

TEST(OrderCorners, ProducesTlTrBrBl) {
  // Scrambled square: BR, TL, BL, TR
  cv::Point2f raw[4] = {{100, 100}, {0, 0}, {0, 100}, {100, 0}};
  auto c = orderCorners(raw);
  EXPECT_EQ(c[0], cv::Point2f(0, 0));      // TL
  EXPECT_EQ(c[1], cv::Point2f(100, 0));    // TR
  EXPECT_EQ(c[2], cv::Point2f(100, 100));  // BR
  EXPECT_EQ(c[3], cv::Point2f(0, 100));    // BL
}

TEST(MatchCornersTemporal, RecoversCyclicRotation) {
  std::vector<cv::Point2f> prev = {{0, 0}, {100, 0}, {100, 100}, {0, 100}};
  // Same square, rotated order by 1
  std::vector<cv::Point2f> cur = {{100, 0}, {100, 100}, {0, 100}, {0, 0}};
  auto matched = matchCornersTemporal(cur, prev, 40000.0);
  for (int i = 0; i < 4; i++) EXPECT_EQ(matched[i], prev[i]);
}

TEST(MatchCornersTemporal, RecoversReflection) {
  std::vector<cv::Point2f> prev = {{0, 0}, {100, 0}, {100, 100}, {0, 100}};
  // Reversed winding (mirror), which cyclic rotations alone cannot fix
  std::vector<cv::Point2f> cur = {{0, 0}, {0, 100}, {100, 100}, {100, 0}};
  auto matched = matchCornersTemporal(cur, prev, 40000.0);
  for (int i = 0; i < 4; i++) EXPECT_EQ(matched[i], prev[i]);
}

TEST(MatchCornersTemporal, LargeJumpKeepsBestMatch) {
  std::vector<cv::Point2f> prev = {{0, 0}, {100, 0}, {100, 100}, {0, 100}};
  std::vector<cv::Point2f> cur = {{500, 500}, {600, 500}, {600, 600}, {500, 600}};
  bool large_jump = false;
  auto matched = matchCornersTemporal(cur, prev, 40000.0, &large_jump);
  EXPECT_TRUE(large_jump);
  ASSERT_EQ(matched.size(), 4u);  // still returns a valid best match, no reset
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
