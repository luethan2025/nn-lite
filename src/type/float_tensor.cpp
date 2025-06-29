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

FloatTensor::FloatTensor(float *data, size_t *size, size_t ndim,
                         bool requires_grad, bool requires_allocation)
    : data_(data), size_(size), ndim_(ndim), requires_grad_(requires_grad),
      requires_allocation_(requires_allocation) {

  // Calculate the number of elements based on the provided size.
  numel_ = 1;
  for (size_t dim = 0; dim < ndim; ++dim) {
    numel_ *= size_[dim];
  }

  if (requires_allocation_) {
    data_ = new float[numel_];
    for (size_t i = 0; i < numel_; ++i) {
      data_[i] = data[i];
    }
    size_ = new size_t[numel_];
    for (size_t dim = 0; dim < ndim; ++dim) {
      size_[dim] = size[dim];
    }
  }

  if (requires_grad_) {
    grad_ = new float[numel_]();
  }
}

FloatTensor::~FloatTensor() {
  if (requires_allocation_) {
    delete[] data_;
    delete[] size_;
  }

  if (requires_grad_) {
    delete[] grad_;
  }
}

/**
 * @brief Zeros every element in the stored gradient.
 */
void FloatTensor::zero_grad_() {
  for (size_t i = 0; i < numel_; ++i) {
    grad_[i] = 0;
  }
}

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
 * @brief Adds input `value` to each element of the stored data.
 *
 * This method will perform addition as an in-place operation.
 *
 * @param value The value to add by.
 */
FloatTensor &FloatTensor::operator+=(float value) {
  add_(value);
  return *this;
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
 * @brief Subtracts input `value` from each element of the stored data.
 *
 * This method will perform subtraction as an in-place operation.
 *
 * @param value The value to subtract by.
 */
FloatTensor &FloatTensor::operator-=(float value) {
  sub_(value);
  return *this;
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
 * @brief Multiplies input `value` to each element of the stored data.
 *
 * This method will perform multiplication as an in-place operation.
 *
 * @param value The value to multiply by.
 */
FloatTensor &FloatTensor::operator*=(float value) {
  mul_(value);
  return *this;
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

/**
 * @brief Divides each element of the stored data by input `value`.
 *
 * This method will perform division as an in-place operation.
 *
 * @param value The value to divide by.
 */
FloatTensor &FloatTensor::operator/=(float value) {
  div_(value);
  return *this;
}

/**
 * @brief Returns the sum of all the elements in the stored data.
 *
 * @return float
 */
float FloatTensor::sum_() {
  float out = 0;
  for (size_t i = 0; i < numel_; ++i) {
    out += data_[i];
  }
  return out;
}

} // namespace focus
