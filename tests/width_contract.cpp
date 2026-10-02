#include "model/Model.hpp"
#include <cstdlib>
#include <iostream>
int main() {
  static_assert(splash::model::ExecutionLimits::maximumBatchWidth == 4);
  if (splash::model::decodeWidthLimit() > 4) return 1;
  std::cout << "maximum=4 effective=" << splash::model::decodeWidthLimit() << '\n';
}
