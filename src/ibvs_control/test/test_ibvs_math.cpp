// Unit tests for the shared IBVS math (ROC convention: GoogleTest).
#include <gtest/gtest.h>

#include "ibvs_control/ibvs_math.hpp"

using namespace ibvs_control;

TEST(TsMemberships, ConvexSumAndBoundaries) {
  double hf, hc;
  // At Z_max the scheduling variable a = 1/Z is minimal -> h_far = 1.
  tsMemberships(1.0, 0.2, 1.0, hf, hc);
  EXPECT_NEAR(hf, 1.0, 1e-12);
  EXPECT_NEAR(hc, 0.0, 1e-12);
  tsMemberships(0.2, 0.2, 1.0, hf, hc);
  EXPECT_NEAR(hf, 0.0, 1e-12);
  EXPECT_NEAR(hc, 1.0, 1e-12);
  // Convex sum everywhere, including outside the clamped range.
  for (double Z : {0.1, 0.3, 0.55, 0.8, 1.5}) {
    tsMemberships(Z, 0.2, 1.0, hf, hc);
    EXPECT_NEAR(hf + hc, 1.0, 1e-12);
    EXPECT_GE(hf, 0.0);
    EXPECT_GE(hc, 0.0);
  }
}

TEST(InteractionMatrix, PointAtOpticalAxis) {
  // At x=y=0 only the depth column and the (de)stabilising rotation terms remain.
  const auto L = pointInteractionMatrix(0.0, 0.0, 0.5);
  EXPECT_DOUBLE_EQ(L(0, 0), -2.0);   // -1/Z
  EXPECT_DOUBLE_EQ(L(1, 1), -2.0);
  EXPECT_DOUBLE_EQ(L(0, 2), 0.0);    // x/Z
  EXPECT_DOUBLE_EQ(L(0, 4), -1.0);   // -(1+x^2)
  EXPECT_DOUBLE_EQ(L(1, 3), 1.0);    // 1+y^2
}

TEST(InteractionMatrix, StackedShapeAndRows) {
  Eigen::MatrixXd pts(4, 2);
  pts << -0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1;
  const auto L = stackedInteractionMatrix(pts, 0.4);
  ASSERT_EQ(L.rows(), 8);
  ASSERT_EQ(L.cols(), 6);
  // Row block i must equal the single-point matrix at that feature.
  const auto L0 = pointInteractionMatrix(-0.1, -0.1, 0.4);
  const bool first_block_matches = L.block<2, 6>(0, 0).isApprox(L0);
  EXPECT_TRUE(first_block_matches);
}

TEST(PixelJacobian, MatchesNormalizedScaling) {
  // At the principal point the pixel Jacobian is the normalized one scaled by f.
  const double fx = 500, fy = 500, cx = 320, cy = 240, Z = 0.5;
  const auto J = pixelJacobian(cx, cy, Z, fx, fy, cx, cy);
  EXPECT_DOUBLE_EQ(J(0, 0), -fx / Z);
  EXPECT_DOUBLE_EQ(J(1, 1), -fy / Z);
  EXPECT_DOUBLE_EQ(J(0, 4), -fx);  // -(fx + uc^2/fx) with uc=0
  EXPECT_DOUBLE_EQ(J(1, 3), fy);
}

TEST(EdgeFactor, DampsNearBorder) {
  EXPECT_DOUBLE_EQ(edgeFactor(320, 240, 640, 480, 50.0), 1.0);   // center
  EXPECT_DOUBLE_EQ(edgeFactor(0, 240, 640, 480, 50.0), 0.3);     // on border
  EXPECT_NEAR(edgeFactor(25, 240, 640, 480, 50.0), 0.65, 1e-12); // halfway
}

TEST(ClampAbs, SymmetricLimits) {
  EXPECT_DOUBLE_EQ(clampAbs(0.5, 0.1), 0.1);
  EXPECT_DOUBLE_EQ(clampAbs(-0.5, 0.1), -0.1);
  EXPECT_DOUBLE_EQ(clampAbs(0.05, 0.1), 0.05);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
