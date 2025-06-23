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

  /**
   * @brief Adds input `other` to the stored data.
   *
   * This method will perform addition as an in-place operation.
   *
   * @param other The tensor to add by.
   */
  void add_(FloatTensor &other);

  /**
   * @brief Adds input `value` to each element of the stored data.
   *
   * This method will perform addition as an in-place operation.
   *
   * @param value The value to add by.
   */
  void add_(float value);

  /**
   * @brief Subtracts input `other` from the stored data.
   *
   * This method will perform subtraction as an in-place operation.
   *
   * @param other The tensor to subtract by.
   */
  void sub_(FloatTensor &other);

  /**
   * @brief Subtracts input `value` from each element of the stored data.
   *
   * This method will perform subtraction as an in-place operation.
   *
   * @param value The value to subtract by.
   */
  void sub_(float value);

  /**
   * @brief Multiplies input `value` to each element of the stored data.
   *
   * This method will perform multiplication as an in-place operation.
   *
   * @param value The value to multiply by.
   */
  void mul_(float value);

  /**
   * @brief Divides each element of the stored data by input `value`.
   *
   * This method will perform division as an in-place operation.
   *
   * @param value The value to divide by.
   */
  void div_(float value);

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
