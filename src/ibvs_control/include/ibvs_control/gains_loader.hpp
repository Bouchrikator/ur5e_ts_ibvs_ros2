// Minimal YAML gains loader for the LMI-designed TS-PDC gains (F1, F2, P and
// optionally L_hat_pinv). Supports both bracketed flow lists and block lists,
// matching the files produced by solve_ts_pdc_lmi_discrete.py /
// solve_ts_lmi_reduced.py.
#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace ibvs_control {

inline bool parseYamlMatrix(const std::string& content, const std::string& key,
                            Eigen::MatrixXd& mat, int rows, int cols) {
  size_t pos = content.find(key + ":");
  if (pos == std::string::npos) return false;

  std::vector<double> vals;

  // Flow style: key: [a, b, c, ...]
  size_t bs = content.find('[', pos);
  size_t nl = content.find('\n', pos);
  if (bs != std::string::npos && (nl == std::string::npos || bs < nl)) {
    size_t be = content.find(']', bs);
    if (be != std::string::npos) {
      std::stringstream ss(content.substr(bs + 1, be - bs - 1));
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        tok.erase(std::remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
        if (!tok.empty()) vals.push_back(std::stod(tok));
      }
    }
  }

  // Block style: key:\n  - a\n  - b ...
  if (vals.empty()) {
    size_t le = content.find('\n', pos);
    if (le == std::string::npos) return false;
    size_t cur = le + 1;
    while (cur < content.size()) {
      while (cur < content.size() && (content[cur] == ' ' || content[cur] == '\t')) cur++;
      if (cur < content.size() - 1 && content[cur] == '-' && content[cur + 1] == ' ') {
        cur += 2;
        size_t ve = content.find('\n', cur);
        std::string vs = (ve != std::string::npos) ? content.substr(cur, ve - cur)
                                                   : content.substr(cur);
        vs.erase(std::remove_if(vs.begin(), vs.end(), ::isspace), vs.end());
        if (!vs.empty()) {
          try {
            vals.push_back(std::stod(vs));
          } catch (...) {
            break;
          }
        }
        cur = (ve != std::string::npos) ? ve + 1 : content.size();
      } else {
        break;
      }
    }
  }

  if (static_cast<int>(vals.size()) != rows * cols) return false;
  mat.resize(rows, cols);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) mat(i, j) = vals[i * cols + j];
  return true;
}

struct TsGains {
  Eigen::MatrixXd F1, F2, P;
  Eigen::MatrixXd L_hat_pinv;  // optional; empty if absent from file
};

// Loads F1, F2, P (6x6). P is normalised by its max eigenvalue so that
// Lyapunov values stay in a comparable numeric range across gain files.
// L_hat_pinv (6x8) is optional: when present it overrides the analytically
// computed pseudo-inverse so run-time matches the offline LMI synthesis.
inline bool loadTsGains(const std::string& filepath, TsGains& gains) {
  std::ifstream file(filepath);
  if (!file.is_open()) return false;
  std::stringstream buf;
  buf << file.rdbuf();
  const std::string content = buf.str();

  bool ok = true;
  ok &= parseYamlMatrix(content, "F1", gains.F1, 6, 6);
  ok &= parseYamlMatrix(content, "F2", gains.F2, 6, 6);
  ok &= parseYamlMatrix(content, "P", gains.P, 6, 6);
  if (ok) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(gains.P);
    const double mx = es.eigenvalues().maxCoeff();
    if (mx > 1e-6) gains.P /= mx;
  }
  parseYamlMatrix(content, "L_hat_pinv", gains.L_hat_pinv, 6, 8);
  return ok;
}

}  // namespace ibvs_control
