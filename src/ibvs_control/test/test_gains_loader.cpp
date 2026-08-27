// Unit tests for the YAML gains loader.
#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>

#include "ibvs_control/gains_loader.hpp"

using ibvs_control::loadTsGains;
using ibvs_control::TsGains;

namespace {

std::string writeTempYaml(const std::string& content) {
  std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                     "/test_gains.yaml";
  std::ofstream f(path);
  f << content;
  return path;
}

std::string identityGainsYaml() {
  auto mat66 = [](double diag) {
    std::string s = "[";
    for (int i = 0; i < 36; i++) {
      s += (i % 7 == 0) ? std::to_string(diag) : "0.0";
      if (i != 35) s += ", ";
    }
    return s + "]";
  };
  return "F1: " + mat66(0.5) + "\nF2: " + mat66(1.5) + "\nP: " + mat66(2.0) + "\n";
}

}  // namespace

TEST(GainsLoader, ParsesFlowStyleAndNormalizesP) {
  const auto path = writeTempYaml(identityGainsYaml());
  TsGains g;
  ASSERT_TRUE(loadTsGains(path, g));
  EXPECT_TRUE(g.F1.isApprox(0.5 * Eigen::MatrixXd::Identity(6, 6)));
  EXPECT_TRUE(g.F2.isApprox(1.5 * Eigen::MatrixXd::Identity(6, 6)));
  // P is normalised by its max eigenvalue -> identity here.
  EXPECT_TRUE(g.P.isApprox(Eigen::MatrixXd::Identity(6, 6)));
  EXPECT_EQ(g.L_hat_pinv.size(), 0);  // absent from file
  std::remove(path.c_str());
}

TEST(GainsLoader, FailsOnMissingKey) {
  const auto path = writeTempYaml("F1: [1.0]\n");
  TsGains g;
  EXPECT_FALSE(loadTsGains(path, g));
  std::remove(path.c_str());
}

TEST(GainsLoader, FailsOnMissingFile) {
  TsGains g;
  EXPECT_FALSE(loadTsGains("/nonexistent/gains.yaml", g));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
