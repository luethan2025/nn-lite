//===----------------------------------------------------------------------===//
//
//               Foundational Operations for Convolutions (FOCUS)
//
// float_tensor.h
//
// Identification: src/include/type/float_tensor.cpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>

namespace focus {

class FloatTensor {
public:
  FloatTensor(float *data, size_t *size, size_t ndim);
  ~FloatTensor();
  /** @brief Input data. */
  float *data_;

  /** @brief Gradient with respect to the input data. */
  float *grad_;

  /** @brief Size. */
  size_t *size_;

  /** @brief Number of dimensions. */
  size_t ndim_;

  /** @brief Total number of elements. */
  size_t numel_;
};

} // namespace focus
