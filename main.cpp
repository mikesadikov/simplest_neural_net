#include "read_csv.h"
#include "simple_nn.h"

std::vector<float> X{};
std::vector<float> y{};
std::vector<float> t{};
std::vector<int> t_y{};

int main(int argc, const char* argv[]) {
  int ret =
      readCsv("dataset/iris-train.csv", "dataset/iris-test.csv", X, y, t, t_y);
  if (ret < 0) {
    return ret;
  }
  SimpleNeuralNet net = SimpleNeuralNet{5, 4, 4, 1};
  net.train(X, y);
  net.predict(t, t_y);
  return 0;
}