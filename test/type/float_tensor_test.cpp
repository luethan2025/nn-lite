//===----------------------------------------------------------------------===//
//
//               Foundational Operations for Convolutions (FOCUS)
//
// float_tensor_test.cpp
//
// Identification: test/type/float_tensor_test.cpp
//
//===----------------------------------------------------------------------===//

#include "type/float_tensor.h"
#include "gtest/gtest.h"

namespace focus {
float A[2][2] = {{1, 2}, {3, 4}};

float B[3][2] = {{1, 2}, {3, 4}, {5, 6}};

float C[2][3][2] = {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}};

float D[2][2][3][2] = {{{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}},
                       {{
                            {13, 14},
                            {15, 16},
                            {17, 18},
                        },
                        {
                            {19, 20},
                            {21, 22},
                            {23, 24},
                        }}};

TEST(FloatTensorTest, FloatTensorInitialization) {
  {
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    float expected_values[4] = {1, 2, 3, 4};
    auto x = FloatTensor(&A[0][0], size, ndim);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    float expected_values[6] = {1, 2, 3, 4, 5, 6};
    auto x = FloatTensor(&B[0][0], size, ndim);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    float expected_values[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto x = FloatTensor(&C[0][0][0], size, ndim);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    size_t size[4] = {2, 2, 3, 2};
    size_t ndim = 4;
    float expected_values[24] = {1,  2,  3,  4,  5,  6,  7,  8,
                                 9,  10, 11, 12, 13, 14, 15, 16,
                                 17, 18, 19, 20, 21, 22, 23, 24};
    auto x = FloatTensor(&D[0][0][0][0], size, ndim);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }
}

TEST(FloatTensorTest, FloatTensorNumel) {
  {
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    auto x = FloatTensor(&A[0][0], size, ndim);
    EXPECT_EQ(x.numel_, 4);
  }

  {
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    auto x = FloatTensor(&B[0][0], size, ndim);
    EXPECT_EQ(x.numel_, 6);
  }

  {
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    auto x = FloatTensor(&C[0][0][0], size, ndim);
    EXPECT_EQ(x.numel_, 12);
  }

  {
    size_t size[4] = {2, 2, 3, 2};
    size_t ndim = 4;
    auto x = FloatTensor(&D[0][0][0][0], size, ndim);
    EXPECT_EQ(x.numel_, 24);
  }
}

TEST(FloatTensorTest, FloatTensorGradInitialization) {
  {
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    auto x = FloatTensor(&A[0][0], size, ndim);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.grad_[i], 0);
    }
  }

  {
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    auto x = FloatTensor(&B[0][0], size, ndim);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.grad_[i], 0);
    }
  }

  {
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    auto x = FloatTensor(&C[0][0][0], size, ndim);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.grad_[i], 0);
    }
  }

  {
    size_t size[4] = {2, 2, 3, 2};
    size_t ndim = 4;
    auto x = FloatTensor(&D[0][0][0][0], size, ndim);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.grad_[i], 0);
    }
  }
}

} // namespace focus
