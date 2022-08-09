#include "simple_nn.h"

#include <cmath>
#include <iostream>

#include "accuracy_evaluation/evaluation.h"

SimpleNeuralNet::SimpleNeuralNet(int epochs, int rows_m1, int columns_m1,
                                 int columns_m2)
    : m_epochs{epochs},  // number of learning epochs
      m_columns_m1{columns_m1},  // number of features
      m_columns_m2{columns_m2},  // number of columns in weight matrix
      m_rows_m1{rows_m1} {  // number of rows (training example size)
  /* the entire dataset is divided into training examples ("batches") of size 4x4,
   for example:
   X (batch = 4x4)  Weights = 4x1
   5.1 3.5 1.4 0.2    0.5
   4.9 3.0 1.4 0.2    0.5
   6.2 3.4 5.4 2.3    0.5
   5.9 3.0 5.1 1.8    0.5
  */
  for (int i = 0; i != rows_m1; ++i) {
    m_weights.push_back(0.5);
  }
}

std::vector<float> operator+(const std::vector<float>& m1,
                             const std::vector<float>& m2) {
  /*  Returns the element-wise sum of two vectors
      Input data:
          m1: vector
          m2: vector
      Output: vector, sum of the vectors m1 and m2.
  */

  const unsigned long vec_size = m1.size();
  std::vector<float> sum(vec_size);

  for (unsigned i = 0; i != vec_size; ++i) {
    sum[i] = m1[i] + m2[i];
  };

  return sum;
}

std::vector<float> operator-(const std::vector<float>& m1,
                             const std::vector<float>& m2) {
  /*  Returns the element-wise difference of two vectors.
      Input data:
          m1: vector
          m2: vector
      Output: vector, m1 - m2, difference between two vectors m1 and
     m2.
  */

  const unsigned long vec_size = m1.size();
  std::vector<float> difference(vec_size);

  for (unsigned i = 0; i != vec_size; ++i) {
    difference[i] = m1[i] - m2[i];
  };

  return difference;
}

std::vector<float> operator*(const std::vector<float>& m1,
                             const std::vector<float>& m2) {
  /*  Returns the product of two vectors (element-wise multiplication).
      Input data:
          m1: vector
          m2: vector
      Output: вектор, m1 * m2, product of two vectors m1 and m2
  */

  const unsigned long vec_size = m1.size();
  std::vector<float> product(vec_size);

  for (unsigned i = 0; i != vec_size; ++i) {
    product[i] = m1[i] * m2[i];
  };

  return product;
}

void SimpleNeuralNet::train(const std::vector<float>& x,
                            const std::vector<float>& y) {
  /*  Neural network training (creation of a matrix of weight coefficients).
      Input data:
         x: vector (signs)
         y: vector (classes)
  */
  std::vector<int> y_int{};
  for (const auto& item_y : y) {
    y_int.push_back(round(item_y));
  }
  int batches = x.size() / m_xbatch_size;
  std::cout << "\nTrain..." << std::endl;

  for (unsigned e = 0; e != m_epochs; ++e) {
    int kx = 0, ky = 0;
    std::vector<int> pred_total{};
    for (unsigned j = 0; j != batches; ++j) {
      std::vector<float> xbatch{};
      std::vector<float> ybatch{};
      for (unsigned b = 0; b != m_xbatch_size; ++b) {
        xbatch.push_back(x.at(b + kx));
      }
      for (unsigned b = 0; b != m_ybatch_size; ++b) {
        ybatch.push_back(y.at(b + ky));
      }
      kx += m_xbatch_size;
      ky += m_ybatch_size;

      std::vector<float> pred = sigmoid(
          mul(xbatch, m_weights, m_rows_m1, m_columns_m1, m_columns_m2));
      std::vector<float> pred_error = ybatch - pred;
      std::vector<float> pred_delta = pred_error * sigmoid_d(pred);
      std::vector<float> W_delta =
          mul(transpose(&xbatch[0], m_columns_m1, m_rows_m1), pred_delta,
              m_rows_m1, m_columns_m1, m_columns_m2);
      m_weights = m_weights + W_delta;

      for (unsigned b = 0; b != m_ybatch_size; ++b) {
        pred_total.push_back(round(pred.at(b)));
      }
    }

    Confusion confusion = Confusion(y_int, pred_total);
    Evaluation evaluation = Evaluation(confusion);
    std::cout << "Epoch " << e + 1 << ":" << std::endl;
    evaluation.print();
  }
}

void SimpleNeuralNet::predict(const std::vector<float>& t,
                              const std::vector<int>& t_y) {
  /*  prediction function.
      Input data:
          t: vector with test data
          t_y: vector with "correct" answers
  */
  int kt = 0;
  std::vector<int> pred_total{};
  std::vector<int> t_int{};
  int batches = t.size() / m_xbatch_size;

  for (unsigned j = 0; j != batches; ++j) {
    std::vector<float> tbatch{};
    for (unsigned b = 0; b != m_xbatch_size; ++b) {
      tbatch.push_back(t.at(b + kt));
    }
    kt += m_xbatch_size;

    std::vector<float> pred =
        sigmoid(mul(tbatch, m_weights, m_rows_m1, m_columns_m1, m_columns_m2));

    for (unsigned b = 0; b != m_ybatch_size; ++b) {
      pred_total.push_back(round(pred.at(b)));
    }
  }

  for (const auto& item_t : t) {
    t_int.push_back(round(item_t));
  }

  std::cout << "\n************* Prediction: *************" << std::endl;
  std::cout << "Test dataset:" << std::endl;
  for (const auto& item_ty : t_y) {
    std::cout << item_ty << " ";
  }

  std::cout << "\nPrediction result:" << std::endl;
  for (const auto& pred : pred_total) {
    std::cout << pred << " ";
  }
  std::cout << std::endl;
}

std::vector<float> SimpleNeuralNet::sigmoid_d(const std::vector<float>& m1) {
  /*  Returns the value of the derivative of the sigmoid function f'(x) = f(x)(1 -
     f(x)), где f(x) - this is a sigmoid function
     Input data:
         m1: vector.
     Output:
         x(1 - x) for each element of the input matrix m1.
  */

  const unsigned long vec_size = m1.size();
  std::vector<float> output(vec_size);

  for (unsigned i = 0; i != vec_size; ++i) {
    output[i] = m1[i] * (1 - m1[i]);
  }

  return output;
}

std::vector<float> SimpleNeuralNet::sigmoid(const std::vector<float>& m1) {
  /*  Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
      Input data:
          m1, vector.
      Output:
          1/(1 + e^-x) for each element of the input matrix m1.
  */

  const unsigned long vec_size = m1.size();
  std::vector<float> output(vec_size);

  for (unsigned i = 0; i != vec_size; ++i) {
    output[i] = 1 / (1 + exp(-m1[i]));
  }

  return output;
}

std::vector<float> SimpleNeuralNet::transpose(const float* m, const int C,
                                              const int R) {
  /*  Returns the transposed matrix of the input matrix.
      Input data:
          m: vector, input matrix
          C: number of columns in input matrix
          R: number of rows in the input matrix
      Output:
          vector, transposed matrix mT of the input matrix m
  */

  std::vector<float> mT(C * R);

  for (unsigned n = 0; n != C * R; n++) {
    unsigned i = n / C;
    unsigned j = n % C;
    mT[n] = m[R * j + i];
  }

  return mT;
}

std::vector<float> SimpleNeuralNet::mul(const std::vector<float>& m1,
                                        const std::vector<float>& m2,
                                        const int m1_rows, const int m1_columns,
                                        const int m2_columns) {
  /*  Returns the product of two matrices: m1 x m2.
      Input data:
          m1: vector, left matrix with m1 rows x m1 columns
          m2: vector, right matrix with m2 columns x m1 rows
     (the number of rows in the right matrix must be equal to the number of columns in the left)
     m1_rows: number of rows in the left matrix m1
     m1_columns: number of columns in the left matrix m1
     m2_columns: number of columns in the right matrix m2

     Output: vector, m1 * m2, product of two vectors m1 and m2, matrix of m1 rows x m2 columns
  */

  std::vector<float> output(m1_rows * m2_columns);

  for (int row = 0; row != m1_rows; ++row) {
    for (int col = 0; col != m2_columns; ++col) {
      output[row * m2_columns + col] = 0.f;
      for (int k = 0; k != m1_columns; ++k) {
        output[row * m2_columns + col] +=
            m1[row * m1_columns + k] * m2[k * m2_columns + col];
      }
    }
  }

  return output;
}
