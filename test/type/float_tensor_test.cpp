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

// clang-format off
float A[2][2] = {
  {1, 2},
  {3, 4}
};

float B[3][2] = {
  {1, 2},
  {3, 4},
  {5, 6}
};

float C[2][3][2] = {
  {
    {1,  2},
    {3,  4},
    {5,  6}
  },
  {
    {7,  8},
    {9,  10},
    {11, 12}
  }
};

float D[2][2][3][2] = {
  {
    {
      {1,  2},
      {3,  4},
      {5,  6}
    },
    {
      {7,  8},
      {9,  10}, 
      {11, 12}
    }
  },
  {
    {
      {13, 14},
      {15, 16},
      {17, 18}
    },
    {
      {19, 20},
      {21, 22},
      {23, 24}
    }
  }
};
// clang-format on

void set_memory_data(float *addr, float *values, size_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    addr[i] = values[i];
  }
}

void reset_A() {
  float values[2][2] = {{1, 2}, {3, 4}};
  size_t numel = 4;
  set_memory_data(&A[0][0], &values[0][0], numel);
}

void zero_A() {
  float values[2][2] = {};
  size_t numel = 4;
  set_memory_data(&A[0][0], &values[0][0], numel);
}

void reset_B() {
  float values[3][2] = {{1, 2}, {3, 4}, {5, 6}};
  size_t numel = 6;
  set_memory_data(&B[0][0], &values[0][0], numel);
}

void zero_B() {
  float values[3][2] = {};
  size_t numel = 6;
  set_memory_data(&B[0][0], &values[0][0], numel);
}

void reset_C() {
  float values[2][3][2] = {{{1, 2}, {3, 4}, {5, 6}},
                           {{7, 8}, {9, 10}, {11, 12}}};
  size_t numel = 12;
  set_memory_data(&C[0][0][0], &values[0][0][0], numel);
}

void zero_C() {
  float values[2][3][2] = {};
  size_t numel = 12;
  set_memory_data(&C[0][0][0], &values[0][0][0], numel);
}

void reset_D() {
  float values[2][2][3][2] = {
      {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}},
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
  size_t numel = 24;
  set_memory_data(&D[0][0][0][0], &values[0][0][0][0], numel);
}

void zero_D() {
  float values[2][2][3][2] = {};
  size_t numel = 24;
  set_memory_data(&D[0][0][0][0], &values[0][0][0][0], numel);
}

TEST(FloatTensorTest, FloatTensorInitialization) {
  {
    reset_A();
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    float expected_values[4] = {1, 2, 3, 4};
    auto x = FloatTensor(&A[0][0], size, ndim);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_B();
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    float expected_values[6] = {1, 2, 3, 4, 5, 6};
    auto x = FloatTensor(&B[0][0], size, ndim);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_C();
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    float expected_values[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto x = FloatTensor(&C[0][0][0], size, ndim);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_D();
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
    reset_A();
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    auto x = FloatTensor(&A[0][0], size, ndim);
    EXPECT_EQ(x.numel_, 4);
  }

  {
    reset_B();
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    auto x = FloatTensor(&B[0][0], size, ndim);
    EXPECT_EQ(x.numel_, 6);
  }

  {
    reset_C();
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    auto x = FloatTensor(&C[0][0][0], size, ndim);
    EXPECT_EQ(x.numel_, 12);
  }

  {
    reset_D();
    size_t size[4] = {2, 2, 3, 2};
    size_t ndim = 4;
    auto x = FloatTensor(&D[0][0][0][0], size, ndim);
    EXPECT_EQ(x.numel_, 24);
  }
}

TEST(FloatTensorTest, FloatTensorGradInitialization) {
  {
    reset_A();
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    auto x = FloatTensor(&A[0][0], size, ndim, true);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.grad_[i], 0);
    }
  }

  {
    reset_B();
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    auto x = FloatTensor(&B[0][0], size, ndim, true);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.grad_[i], 0);
    }
  }

  {
    reset_C();
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    auto x = FloatTensor(&C[0][0][0], size, ndim, true);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.grad_[i], 0);
    }
  }

  {
    reset_D();
    size_t size[4] = {2, 2, 3, 2};
    size_t ndim = 4;
    auto x = FloatTensor(&D[0][0][0][0], size, ndim, true);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.grad_[i], 0);
    }
  }
}

TEST(FloatTensorTest, FloatTensorRequiresAllocation) {
  {
    reset_A();
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    float expected_values[4] = {1, 2, 3, 4};
    auto x = FloatTensor(&A[0][0], size, ndim, false, true);
    zero_A();
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_B();
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    float expected_values[6] = {1, 2, 3, 4, 5, 6};
    auto x = FloatTensor(&B[0][0], size, ndim, false, true);
    zero_B();
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_C();
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    float expected_values[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto x = FloatTensor(&C[0][0][0], size, ndim, false, true);
    zero_C();
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_D();
    size_t size[4] = {2, 2, 3, 2};
    size_t ndim = 4;
    float expected_values[24] = {1,  2,  3,  4,  5,  6,  7,  8,
                                 9,  10, 11, 12, 13, 14, 15, 16,
                                 17, 18, 19, 20, 21, 22, 23, 24};
    auto x = FloatTensor(&D[0][0][0][0], size, ndim, false, true);
    zero_D();
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }
}

TEST(FloatTensorTest, FloatTensorZeroGrad) {
  {
    reset_A();
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    auto x = FloatTensor(&A[0][0], size, ndim, true);
    for (size_t i = 0; i < x.numel_; ++i) {
      x.grad_[i] = 1;
    }
    x.zero_grad_();
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.grad_[i], 0);
    }
  }

  {
    reset_B();
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    auto x = FloatTensor(&B[0][0], size, ndim, true);
    for (size_t i = 0; i < x.numel_; ++i) {
      x.grad_[i] = 1;
    }
    x.zero_grad_();
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.grad_[i], 0);
    }
  }

  {
    reset_C();
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    auto x = FloatTensor(&C[0][0][0], size, ndim, true);
    for (size_t i = 0; i < x.numel_; ++i) {
      x.grad_[i] = 1;
    }
    x.zero_grad_();
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.grad_[i], 0);
    }
  }

  {
    reset_D();
    size_t size[4] = {2, 2, 3, 2};
    size_t ndim = 4;
    auto x = FloatTensor(&D[0][0][0][0], size, ndim, true);
    for (size_t i = 0; i < x.numel_; ++i) {
      x.grad_[i] = 1;
    }
    x.zero_grad_();
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.grad_[i], 0);
    }
  }
}

TEST(FloatTensorTest, FloatTensorInPlaceConstantAddition) {
  {
    reset_A();
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    float expected_values[4] = {2, 3, 4, 5};
    auto x = FloatTensor(&A[0][0], size, ndim);
    int value = 1;
    x.add_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_B();
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    float expected_values[6] = {2, 3, 4, 5, 6, 7};
    auto x = FloatTensor(&B[0][0], size, ndim);
    int value = 1;
    x.add_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_C();
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    float expected_values[12] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    auto x = FloatTensor(&C[0][0][0], size, ndim);
    int value = 1;
    x.add_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_D();
    size_t size[4] = {2, 2, 3, 2};
    size_t ndim = 4;
    float expected_values[24] = {2,  3,  4,  5,  6,  7,  8,  9,
                                 10, 11, 12, 13, 14, 15, 16, 17,
                                 18, 19, 20, 21, 22, 23, 24, 25};
    auto x = FloatTensor(&D[0][0][0][0], size, ndim);
    int value = 1;
    x.add_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }
}

TEST(FloatTensorTest, FloatTensorInPlaceTensorAddition) {
  {
    reset_A();
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    float expected_values[4] = {2, 4, 6, 8};
    auto x = FloatTensor(&A[0][0], size, ndim);
    float other_data[2][2] = {{1, 2}, {3, 4}};
    auto other = FloatTensor(&other_data[0][0], size, ndim);
    x.add_(other);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_B();
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    float expected_values[6] = {2, 4, 6, 8, 10, 12};
    auto x = FloatTensor(&B[0][0], size, ndim);
    float other_data[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    auto other = FloatTensor(&other_data[0][0], size, ndim);
    x.add_(other);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_C();
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    float expected_values[12] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24};
    auto x = FloatTensor(&C[0][0][0], size, ndim);
    float other_data[2][3][2] = {{{1, 2}, {3, 4}, {5, 6}},
                                 {{7, 8}, {9, 10}, {11, 12}}};
    auto other = FloatTensor(&other_data[0][0][0], size, ndim);
    x.add_(other);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_D();
    size_t size[4] = {2, 2, 3, 2};
    size_t ndim = 4;
    float expected_values[24] = {2,  4,  6,  8,  10, 12, 14, 16,
                                 18, 20, 22, 24, 26, 28, 30, 32,
                                 34, 36, 38, 40, 42, 44, 46, 48};
    auto x = FloatTensor(&D[0][0][0][0], size, ndim);
    float other_data[2][2][3][2] = {
        {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}},
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
    auto other = FloatTensor(&other_data[0][0][0][0], size, ndim);
    x.add_(other);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }
}

TEST(FloatTensorTest, FloatTensorInPlaceConstantSubtraction) {
  {
    reset_A();
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    float expected_values[4] = {0, 1, 2, 3};
    auto x = FloatTensor(&A[0][0], size, ndim);
    int value = 1;
    x.sub_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_B();
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    float expected_values[6] = {0, 1, 2, 3, 4, 5};
    auto x = FloatTensor(&B[0][0], size, ndim);
    int value = 1;
    x.sub_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_C();
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    float expected_values[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto x = FloatTensor(&C[0][0][0], size, ndim);
    int value = 1;
    x.sub_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_D();
    size_t size[4] = {2, 2, 3, 2};
    size_t ndim = 4;
    float expected_values[24] = {0,  1,  2,  3,  4,  5,  6,  7,
                                 8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23};
    auto x = FloatTensor(&D[0][0][0][0], size, ndim);
    int value = 1;
    x.sub_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }
}

TEST(FloatTensorTest, FloatTensorInPlaceTensorSubtraction) {
  {
    reset_A();
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    float expected_values[4] = {0};
    auto x = FloatTensor(&A[0][0], size, ndim);
    float other_data[2][2] = {{1, 2}, {3, 4}};
    auto other = FloatTensor(&other_data[0][0], size, ndim);
    x.sub_(other);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_B();
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    float expected_values[6] = {0};
    auto x = FloatTensor(&B[0][0], size, ndim);
    float other_data[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    auto other = FloatTensor(&other_data[0][0], size, ndim);
    x.sub_(other);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_C();
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    float expected_values[12] = {0};
    auto x = FloatTensor(&C[0][0][0], size, ndim);
    float other_data[2][3][2] = {{{1, 2}, {3, 4}, {5, 6}},
                                 {{7, 8}, {9, 10}, {11, 12}}};
    auto other = FloatTensor(&other_data[0][0][0], size, ndim);
    x.sub_(other);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_D();
    size_t size[4] = {2, 2, 3, 2};
    size_t ndim = 4;
    float expected_values[24] = {0};
    auto x = FloatTensor(&D[0][0][0][0], size, ndim);
    float other_data[2][2][3][2] = {
        {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}},
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
    auto other = FloatTensor(&other_data[0][0][0][0], size, ndim);
    x.sub_(other);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }
}

TEST(FloatTensorTest, FloatTensorInPlaceConstantMultplication) {
  {
    reset_A();
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    float expected_values[4] = {0};
    auto x = FloatTensor(&A[0][0], size, ndim);
    int value = 0;
    x.mul_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_B();
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    float expected_values[6] = {0};
    auto x = FloatTensor(&B[0][0], size, ndim);
    int value = 0;
    x.mul_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_C();
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    float expected_values[12] = {0};
    auto x = FloatTensor(&C[0][0][0], size, ndim);
    int value = 0;
    x.mul_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_D();
    size_t size[4] = {2, 2, 3, 2};
    size_t ndim = 4;
    float expected_values[24] = {0};
    auto x = FloatTensor(&D[0][0][0][0], size, ndim);
    int value = 0;
    x.mul_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }
}

TEST(FloatTensorTest, FloatTensorInPlaceDivision) {
  {
    reset_A();
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    float expected_values[4] = {0.5f, 1.0f, 1.5f, 2.0f};
    auto x = FloatTensor(&A[0][0], size, ndim);
    int value = 2;
    x.div_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_B();
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    float expected_values[6] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f};
    auto x = FloatTensor(&B[0][0], size, ndim);
    int value = 2;
    x.div_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_C();
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    float expected_values[12] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f,
                                 3.5f, 4.0f, 4.5f, 5.0f, 5.5f, 6.0f};
    auto x = FloatTensor(&C[0][0][0], size, ndim);
    int value = 2;
    x.div_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }

  {
    reset_D();
    size_t size[4] = {2, 2, 3, 2};
    size_t ndim = 4;
    float expected_values[24] = {0.5f, 1.0f,  1.5f,  2.0f,  2.5f,  3.0f,
                                 3.5f, 4.0f,  4.5f,  5.0f,  5.5f,  6.0f,
                                 6.5f, 7.0f,  7.5f,  8.0f,  8.5f,  9.0f,
                                 9.5f, 10.0f, 10.5f, 11.0f, 11.5f, 12.0f};
    auto x = FloatTensor(&D[0][0][0][0], size, ndim);
    int value = 2;
    x.div_(value);
    for (size_t i = 0; i < x.numel_; ++i) {
      EXPECT_EQ(x.data_[i], expected_values[i]);
    }
  }
}

TEST(FloatTensorTest, FloatTensorSum) {
  {
    reset_A();
    size_t size[2] = {2, 2};
    size_t ndim = 2;
    float expected_value = 10;
    auto x = FloatTensor(&A[0][0], size, ndim);
    auto out = x.sum_();
    EXPECT_EQ(out, expected_value);
  }

  {
    reset_B();
    size_t size[2] = {3, 2};
    size_t ndim = 2;
    float expected_value = 21;
    auto x = FloatTensor(&B[0][0], size, ndim);
    auto out = x.sum_();
    EXPECT_EQ(out, expected_value);
  }

  {
    reset_C();
    size_t size[3] = {2, 3, 2};
    size_t ndim = 3;
    float expected_value = 78;
    auto x = FloatTensor(&C[0][0][0], size, ndim);
    auto out = x.sum_();
    EXPECT_EQ(out, expected_value);
  }

  {
    reset_D();
    size_t size[4] = {2, 2, 3, 2};
    size_t ndim = 4;
    float expected_value = 300;
    auto x = FloatTensor(&D[0][0][0][0], size, ndim);
    auto out = x.sum_();
    EXPECT_EQ(out, expected_value);
  }
}

} // namespace focus
