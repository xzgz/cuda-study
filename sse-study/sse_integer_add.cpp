#include <bits/stdc++.h>

using namespace std;

// refer to: https://www.jianshu.com/p/d718c1ea5f22

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

int normal_add(int *a, size_t n) {
  assert(a);
  int sum = 0;

  for(size_t i = 0; i < n; ++i) {
    sum += a[i];
  }
  return sum;
}

int normal_add_loop4(int *a, size_t n) {
  assert(a);

  int sum = 0;
  size_t block = n / 4;    // 等价于n >> 2
  size_t reserve = n % 4;  // 等价于 n & 0x3
  int *p = a;

  for(size_t i = 0; i < block; ++i) {
    sum += *p;
    sum += *(p+1);
    sum += *(p+2);
    sum += *(p+3);
    p += 4;
  }

  // 剩余的不足4字节
  for(size_t i = 0; i < reserve; ++i) {
    sum += p[i];
  }
  return sum;
}

int sse_add(int *a, size_t n) {
  assert(a);

  int sum = 0;
  __m128i sse_sum = _mm_setzero_si128();
  __m128i sse_load;
  __m128i *p = (__m128i*)a;

  size_t block = n / 4;     // SSE寄存器能一次处理4个32位的整数
  size_t reserve = n % 4;  // 剩余的不足16字节

  for(size_t i = 0; i < block; ++i) {
    sse_load = _mm_load_si128(p);
    sse_sum  = _mm_add_epi32(sse_sum, sse_load); // 带符号32位紧缩加法
    ++p;
  }

  // 剩余的不足16字节
  int *q = (int *)p;
  for(size_t i = 0; i < reserve; ++i) {
    sum += q[i];
  }

  // 将累加值合并
  sse_sum = _mm_hadd_epi32(sse_sum, sse_sum);  // 带符号32位水平加法
  sse_sum = _mm_hadd_epi32(sse_sum, sse_sum);

  sum += _mm_cvtsi128_si32(sse_sum);  // 返回低32位
  return sum;
}

int sse_add_loop4(int *a, int n) {
  assert(a);

  int sum = 0;
  size_t block = n / 16;    // SSE寄存器能一次处理4个32位的整数
  size_t reserve = n % 16; // 剩余的字节

  __m128i sse_sum0 = _mm_setzero_si128();
  __m128i sse_sum1 = _mm_setzero_si128();
  __m128i sse_sum2 = _mm_setzero_si128();
  __m128i sse_sum3 = _mm_setzero_si128();
  __m128i sse_load0;
  __m128i sse_load1;
  __m128i sse_load2;
  __m128i sse_load3;
  __m128i *p = (__m128i*)a;

  for(size_t i = 0; i < block; ++i) {
    sse_load0 = _mm_load_si128(p);
    sse_load1 = _mm_load_si128(p+1);
    sse_load2 = _mm_load_si128(p+2);
    sse_load3 = _mm_load_si128(p+3);

    sse_sum0 = _mm_add_epi32(sse_sum0, sse_load0);
    sse_sum1 = _mm_add_epi32(sse_sum1, sse_load1);
    sse_sum2 = _mm_add_epi32(sse_sum2, sse_load2);
    sse_sum3 = _mm_add_epi32(sse_sum3, sse_load3);

    p += 4;
  }

  // 剩余的不足16字节
  int *q = (int *)p;
  for(size_t i = 0; i < reserve; ++i) {
    sum += q[i];
  }

  // 将累加值两两合并
  sse_sum0 = _mm_add_epi32(sse_sum0, sse_sum1);
  sse_sum2 = _mm_hadd_epi32(sse_sum2, sse_sum3);
  sse_sum0 = _mm_add_epi32(sse_sum0, sse_sum2);

  sse_sum0 = _mm_hadd_epi32(sse_sum0, sse_sum0);
  sse_sum0 = _mm_hadd_epi32(sse_sum0, sse_sum0);

  sum += _mm_cvtsi128_si32(sse_sum0); // 取低32位

  return sum;
}

int main(int argc, char *argv[]) {
//  if (argc != 2) {
//    std::cout << "error argument" << std::endl;
//    return -1;
//  }

  argv[1] = "100000000";
  size_t n = atoi(argv[1]);

  int *a = NULL;
  posix_memalign((void **) &a, 16, sizeof(int) * n);

  for (size_t i = 0; i < n; ++i) {
    a[i] = 5;
  }
  int sum = 0;

  {
    Timer("sse_add_loop4");
    sum = sse_add_loop4(a, n);
  }
  {
    Timer("normal_add");
    sum = normal_add(a, n);
  }
  {
    Timer("normal_add_loop4");
    sum = normal_add_loop4(a, n);
  }
  {
    Timer("sse_add");
    sum = sse_add(a, n);
  }

  cout << endl << "sum=" << sum << endl;
  free(a);

  return 0;
}
