#include <vector>

class SimpleNeuralNet {
 public:
  SimpleNeuralNet(int epochs, int rows_m1, int columns_m1, int columns_m2);
  void train(const std::vector<float>& x, const std::vector<float>& y);
  void predict(const std::vector<float>& t, const std::vector<int>& t_y);

 private:
  std::vector<float> sigmoid_d(const std::vector<float>& m1);
  std::vector<float> sigmoid(const std::vector<float>& m1);
  std::vector<float> transpose(const float* m, const int C, const int R);
  std::vector<float> mul(const std::vector<float>& m1,
                         const std::vector<float>& m2, const int m1_rows,
                         const int m1_columns, const int m2_columns);

  int m_epochs;
  int m_columns_m1;
  int m_columns_m2;
  int m_rows_m1;
  const int m_xbatch_size = 16;
  const int m_ybatch_size = 4;
  std::vector<float> m_weights;
};
