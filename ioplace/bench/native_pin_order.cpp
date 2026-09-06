// Match DREAMPlace's libstdc++ std::sort/unique policy, including tied pins.
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

extern "C" int64_t select_native_pins(
    int64_t nets, const int64_t* starts, const int64_t* nodes,
    const int64_t* node_ranks, const int64_t* pin_ranks,
    int64_t* selected, int64_t* selected_starts) {
  std::vector<int64_t> ids;
  int64_t count = 0;
  selected_starts[0] = 0;
  for (int64_t net = 0; net < nets; ++net) {
    ids.resize(starts[net + 1] - starts[net]);
    std::iota(ids.begin(), ids.end(), starts[net]);
    std::sort(ids.begin(), ids.end(), [&](int64_t a, int64_t b) {
      auto na = node_ranks[nodes[a]], nb = node_ranks[nodes[b]];
      return na < nb || (na == nb && pin_ranks[a] < pin_ranks[b]);
    });
    int64_t previous = -1;
    for (auto pin : ids) {
      if (nodes[pin] != previous) {
        selected[count++] = pin;
        previous = nodes[pin];
      }
    }
    selected_starts[net + 1] = count;
  }
  return count;
}
