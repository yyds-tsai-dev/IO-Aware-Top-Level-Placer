#include "flute.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <vector>

namespace {
std::once_flag lut_once;
int lut_initializations = 0;

int round_ties_even(double value) {
  const double lower = std::floor(value);
  const double fraction = value - lower;
  if (fraction < 0.5) return static_cast<int>(lower);
  if (fraction > 0.5) return static_cast<int>(lower + 1.0);
  return static_cast<int>(std::fmod(lower, 2.0) == 0.0 ? lower : lower + 1.0);
}
}  // namespace

extern "C" void ioplace_flute_batch_init(const char* powv, const char* post) {
  std::call_once(lut_once, [=]() {
    flute::readLUT(powv, post);
    ++lut_initializations;
  });
}

extern "C" int ioplace_flute_batch_init_count() {
  return lut_initializations;
}

extern "C" int ioplace_flute_batch(
    int64_t net_count, const double* pins, const int64_t* starts,
    double scale, int accuracy, int threads, const int64_t* branch_starts,
    double* positions, int64_t* parents, int64_t* terminal_nodes,
    int64_t* error_net) {
  *error_net = -1;
  if (net_count < 0 || !(scale > 0.0) || !std::isfinite(scale) ||
      accuracy < 1 || accuracy > 10 || threads < 1) {
    return -1;
  }

  // Validate before entering FLUTE so no worker can leave a partially failed
  // tree that looks usable to the caller.
  for (int64_t net = 0; net < net_count; ++net) {
    const int64_t degree64 = starts[net + 1] - starts[net];
    if (degree64 < 2 || degree64 > 256) {
      *error_net = net;
      return -2;
    }
    double xmin = pins[2 * starts[net]], xmax = xmin;
    double ymin = pins[2 * starts[net] + 1], ymax = ymin;
    for (int64_t pin = starts[net]; pin < starts[net + 1]; ++pin) {
      const double x = pins[2 * pin], y = pins[2 * pin + 1];
      if (!std::isfinite(x) || !std::isfinite(y)) {
        *error_net = net;
        return -3;
      }
      xmin = std::min(xmin, x); xmax = std::max(xmax, x);
      ymin = std::min(ymin, y); ymax = std::max(ymax, y);
    }
    const double span = std::max(xmax - xmin, ymax - ymin);
    const double budget = span * scale * std::max<int64_t>(2 * degree64, 2);
    if (!std::isfinite(budget) || budget >= std::numeric_limits<int32_t>::max()) {
      *error_net = net;
      return -4;
    }
  }

#pragma omp parallel for schedule(static) num_threads(threads)
  for (int64_t net = 0; net < net_count; ++net) {
    const int64_t pin_start = starts[net];
    const int degree = static_cast<int>(starts[net + 1] - pin_start);
    double origin_x = pins[2 * pin_start], origin_y = pins[2 * pin_start + 1];
    for (int i = 1; i < degree; ++i) {
      origin_x = std::min(origin_x, pins[2 * (pin_start + i)]);
      origin_y = std::min(origin_y, pins[2 * (pin_start + i) + 1]);
    }
    std::vector<int> x(degree), y(degree);
    for (int i = 0; i < degree; ++i) {
      x[i] = round_ties_even(
          (pins[2 * (pin_start + i)] - origin_x) * scale);
      y[i] = round_ties_even(
          (pins[2 * (pin_start + i) + 1] - origin_y) * scale);
    }
    flute::Tree tree = flute::flute(degree, x.data(), y.data(), accuracy);
    const int64_t row_start = branch_starts[net];
    const int rows = 2 * degree - 2;
    for (int row = 0; row < rows; ++row) {
      positions[2 * (row_start + row)] = tree.branch[row].x / scale + origin_x;
      positions[2 * (row_start + row) + 1] = tree.branch[row].y / scale + origin_y;
      parents[row_start + row] = row_start + tree.branch[row].n;
    }
    // FLUTE sorts terminal rows. Match coordinate occurrences one-to-one so
    // duplicate physical pins remain distinct raw terminals.
    bool used[256] = {};
    for (int pin = 0; pin < degree; ++pin) {
      int slot = -1;
      for (int row = 0; row < degree; ++row) {
        if (!used[row] && tree.branch[row].x == x[pin] &&
            tree.branch[row].y == y[pin]) {
          slot = row;
          used[row] = true;
          break;
        }
      }
      terminal_nodes[pin_start + pin] = slot < 0 ? -1 : row_start + slot;
    }
    std::free(tree.branch);
  }
  return 0;
}
