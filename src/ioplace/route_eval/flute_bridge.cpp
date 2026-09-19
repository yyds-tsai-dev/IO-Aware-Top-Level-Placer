#include "flute.hpp"
#include <cstdlib>

extern "C" void ioplace_flute_init(const char* powv, const char* post) {
    flute::readLUT(powv, post);
}
extern "C" int ioplace_flute(int degree, int* x, int* y, int accuracy, int* branches) {
    if (degree < 2 || degree > 256 || accuracy < 1 || accuracy > 10) return -1;
    auto tree = flute::flute(degree, x, y, accuracy);
    if (!tree.branch) return -2;
    for (int i = 0; i < 2 * degree - 2; ++i) {
        branches[3*i] = tree.branch[i].x;
        branches[3*i+1] = tree.branch[i].y;
        branches[3*i+2] = tree.branch[i].n;
    }
    std::free(tree.branch);
    return 0;
}
