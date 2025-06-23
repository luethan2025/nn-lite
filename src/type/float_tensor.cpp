//===----------------------------------------------------------------------===//
//
//               Foundational Operations for Convolutions (FOCUS)
//
// float_tensor.cpp
//
// Identification: src/type/float_tensor.cpp
//
//===----------------------------------------------------------------------===//

#include "type/float_tensor.h"

namespace focus {

FloatTensor::FloatTensor(float *data, size_t *size, size_t ndim)
    : data_(data), size_(size), ndim_(ndim) {

  // Calculate the number of elements based on the provided size.
  numel_ = 1;
  for (size_t dim = 0; dim < ndim; ++dim) {
    numel_ *= size_[dim];
  }
  grad_ = new float[numel_]();
}

FloatTensor::~FloatTensor() { delete[] grad_; }

/**
 * @brief Adds input `other` to the stored data.
 *
 * This method will perform addition as an in-place operation.
 *
 * @param other The tensor to add by.
 */
void FloatTensor::add_(FloatTensor &other) {
  for (size_t i = 0; i < numel_; ++i) {
    data_[i] += other.data_[i];
  }
}

/**
 * @brief Adds input `value` to each element of the stored data.
 *
 * This method will perform addition as an in-place operation.
 *
 * @param value The value to add by.
 */
void FloatTensor::add_(float value) {
  for (size_t i = 0; i < numel_; ++i) {
    data_[i] += value;
  }
}

/**
 * @brief Subtracts input `other` from the stored data.
 *
 * This method will perform subtraction as an in-place operation.
 *
 * @param other The tensor to subtract by.
 */
void FloatTensor::sub_(FloatTensor &other) {
  for (size_t i = 0; i < numel_; ++i) {
    data_[i] -= other.data_[i];
  }
}

/**
 * @brief Subtracts input `value` from each element of the stored data.
 *
 * This method will perform subtraction as an in-place operation.
 *
 * @param value The value to subtract by.
 */
void FloatTensor::sub_(float value) {
  for (size_t i = 0; i < numel_; ++i) {
    data_[i] -= value;
  }
}

/**
 * @brief Multiplies input `value` to each element of the stored data.
 *
 * This method will perform multiplication as an in-place operation.
 *
 * @param value The value to multiply by.
 */
void FloatTensor::mul_(float value) {
  for (size_t i = 0; i < numel_; ++i) {
    data_[i] *= value;
  }
}

/**
 * @brief Divides each element of the stored data by input `value`.
 *
 * This method will perform division as an in-place operation.
 *
 * @param value The value to divide by.
 */
void FloatTensor::div_(float value) {
  for (size_t i = 0; i < numel_; ++i) {
    data_[i] /= value;
  }
}

} // namespace focus
