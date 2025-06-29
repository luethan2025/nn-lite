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
  FloatTensor(float *data, size_t *size, size_t ndim,
              bool requires_grad = false, bool requires_allocation = false);
  ~FloatTensor();

  /**
   * @brief Zeros every element in the stored gradient.
   */
  void zero_grad_();

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
   * @brief Adds input `value` to each element of the stored data.
   *
   * This method will perform addition as an in-place operation.
   *
   * @param value The value to add by.
   */
  FloatTensor &operator+=(float value);

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
   * @brief Subtracts input `value` from each element of the stored data.
   *
   * This method will perform subtraction as an in-place operation.
   *
   * @param value The value to subtract by.
   */
  FloatTensor &operator-=(float value);

  /**
   * @brief Multiplies input `value` to each element of the stored data.
   *
   * This method will perform multiplication as an in-place operation.
   *
   * @param value The value to multiply by.
   */
  void mul_(float value);

  /**
   * @brief Multiplies input `value` to each element of the stored data.
   *
   * This method will perform multiplication as an in-place operation.
   *
   * @param value The value to multiply by.
   */
  FloatTensor &operator*=(float value);

  /**
   * @brief Divides each element of the stored data by input `value`.
   *
   * This method will perform division as an in-place operation.
   *
   * @param value The value to divide by.
   */
  void div_(float value);

  /**
   * @brief Divides each element of the stored data by input `value`.
   *
   * This method will perform division as an in-place operation.
   *
   * @param value The value to divide by.
   */
  FloatTensor &operator/=(float value);

  /**
   * @brief Returns the sum of all the elements in the stored data.
   *
   * @return float
   */
  float sum_();

  /** @brief Input data. */
  float *data_;

  /** @brief `true` if input `requires_grad` was `true`, `false` otherwise. */
  bool requires_grad_;

  /** @brief Gradient with respect to the input data. */
  float *grad_;

  /**
   * @brief `true` if input `requires_allocation` was `true`, `false` otherwise.
   */
  bool requires_allocation_;

  /** @brief Size. */
  size_t *size_;

  /** @brief Number of dimensions. */
  size_t ndim_;

  /** @brief Total number of elements. */
  size_t numel_;
};

} // namespace focus
