#include <bits/stdc++.h>

using namespace std;

class Timer {
 public:
  Timer(const std::string& n) : name(n), start(std::clock()) {}

  ~Timer() {
    double elapsed = (double(std::clock() - start)) / double(CLOCKS_PER_SEC);

    std::cout << name << ": " << int(elapsed * 1000) << "ms" << std::endl;
  }

 private:
  std::string name;
  std::clock_t start;
};

#define Timer(n) Timer timer(n)

double compute_pi_naive(size_t dt) {
  double pi = 0.0;
  double delta = 1.0 / dt;
  for (size_t i = 0; i < dt; i++) {
    double x = (double)i / dt;
    pi += delta / (1.0 + x * x);
  }
  return pi * 4.0;
}

double compute_pi_omp_avx(size_t dt){
  double pi = 0.0;
  double delta = 1.0 / dt;
  __m256d ymm0, ymm1, ymm2, ymm3, ymm4;
  ymm0 = _mm256_set1_pd(1.0);
  ymm1 = _mm256_set1_pd(delta);
  ymm2 = _mm256_set_pd(delta * 3, delta * 2, delta, 0.0);
  ymm4 = _mm256_setzero_pd();
  for (int i = 0; i <= dt-4; i += 4) {
    ymm3 = _mm256_set1_pd(i * delta);
    ymm3 = _mm256_add_pd(ymm3, ymm2);
    ymm3 = _mm256_mul_pd(ymm3, ymm3);
    ymm3 = _mm256_add_pd(ymm0, ymm3);
    ymm3 = _mm256_div_pd(ymm1, ymm3);
    ymm4 = _mm256_add_pd(ymm4, ymm3);
  }
  double tmp[4] __attribute__((aligned(32)));
  _mm256_store_pd(tmp, ymm4);
  pi += tmp[0] + tmp[1] + tmp[2] + tmp[3];
  return pi * 4.0;
}

double compute_pi_omp_avx_v2(size_t dt){
  double pi = 0.0;
  double delta = 1.0 / dt;
  __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6;
  ymm0 = _mm256_set1_pd(1.0);
  ymm1 = _mm256_set1_pd(delta);
  ymm2 = _mm256_set_pd(delta * 3, delta * 2, delta, 0.0);
  ymm4 = _mm256_setzero_pd();
  ymm5 = _mm256_setzero_pd();
  for (int i = 0; i <= dt - 8; i += 8){
    ymm3 = _mm256_set1_pd(i * delta);
    ymm3 = _mm256_add_pd(ymm3, ymm2);
    ymm3 = _mm256_mul_pd(ymm3, ymm3);
    ymm3 = _mm256_add_pd(ymm0, ymm3);
    ymm3 = _mm256_div_pd(ymm1, ymm3);
    ymm4 = _mm256_add_pd(ymm4, ymm3);

    ymm6 = _mm256_set1_pd((i+4) * delta);
    ymm6 = _mm256_add_pd(ymm6, ymm2);
    ymm6 = _mm256_mul_pd(ymm6, ymm6);
    ymm6 = _mm256_add_pd(ymm0, ymm6);
    ymm6 = _mm256_div_pd(ymm1, ymm6);
    ymm5 = _mm256_add_pd(ymm5, ymm6);
  }
  ymm4 = _mm256_add_pd(ymm4, ymm5);

  double tmp[4] __attribute__((aligned(32)));
  _mm256_store_pd(tmp, ymm4);
  pi += tmp[0] + tmp[1] + tmp[2] + tmp[3];
  return pi * 4.0;
}

int main() {
  double pi;

  {
    Timer("compute_pi_omp_avx_v2");
    pi = compute_pi_omp_avx_v2(10000000);
  }
  cout << "pi=" << fixed << setprecision(8) << pi << endl;

  {
    Timer("compute_pi_naive");
    pi = compute_pi_naive(10000000);
  }
  cout << "pi=" << fixed << setprecision(8) << pi << endl;

  {
    Timer("compute_pi_omp_avx");
    pi = compute_pi_omp_avx(10000000);
  }
  cout << "pi=" << fixed << setprecision(8) << pi << endl;

  return 0;
}
