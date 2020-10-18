#include <bits/stdc++.h>
#include <sys/time.h>
#include <unistd.h>

using namespace std;

// refer to: https://blog.csdn.net/guotianqing/article/details/80958281

static double what_time_is_it_now() {
  struct timeval time;
  if (gettimeofday(&time,NULL)){
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

static __inline__ double GetCurrentTime(double cpu_frequency = 1.80e9) {
  unsigned long long time;
#if defined (__i386__)
  unsigned long long int x;
  __asm__ volatile("rdtsc":"=A"(x));
  time = x;
#elif defined (__x86_64__)
  unsigned hi, lo;
  __asm__ volatile("rdtsc":"=a"(lo), "=d"(hi));
  time = ((unsigned long long) lo) | (((unsigned long long) hi) << 32);
#endif
  return ((double)time) / cpu_frequency;
}

class Timer {
 public:
//  Timer(const std::string& n, double *t = nullptr) : name(n), start(std::clock()), t(t) {}
  Timer(const std::string& n, double *t = nullptr) : name(n), start(what_time_is_it_now()), t(t) {}
//  Timer(const std::string& n, double *t = nullptr) : name(n), start(GetCurrentTime()), t(t) {}

  ~Timer() {
//    double elapsed = (double(std::clock() - start)) / double(CLOCKS_PER_SEC);
    double elapsed = what_time_is_it_now() - start;
//    double elapsed = GetCurrentTime() - start;

    std::cout << name << ": " << int(elapsed * 1000) << "ms" << std::endl;
    if (t != nullptr) *t = elapsed;
  }

 private:
  std::string name;
//  std::clock_t start;
  double start;
  double *t;
};

#define Timer(n) Timer timer(n)
#define TimerRecord(n, pt) Timer timer(n, pt)

void measure_cpu_peak_performance() {
  __asm__ __volatile__(
          "mov $0x10000000, %rax\n\t"
          "vxorps %ymm0, %ymm0, %ymm0\n\t"
          "vxorps %ymm1, %ymm1, %ymm1\n\t"
          "vxorps %ymm2, %ymm2, %ymm2\n\t"
          "vxorps %ymm3, %ymm3, %ymm3\n\t"
          "rs:\n\t"
          "vmulps %ymm1, %ymm1, %ymm0\n\t"
          "vaddps %ymm3, %ymm3, %ymm2\n\t"
          "subq $0x1, %rax\n\t"
          "jne rs"
          );
}

#define THREAD_NUM 4
int num = 0;
double t[THREAD_NUM];

void *threadFun(void *arg) {
  int *a = (int *)arg;
  {
    TimerRecord("measure_cpu_peak_performance", &t[*a]);
    measure_cpu_peak_performance();
  }
}

#define handle_error_en(en, msg) \
        do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

int main(int argc, char *argv[]) {
  num = sysconf(_SC_NPROCESSORS_CONF);
  cpu_set_t cpuset;
  pthread_t thread[THREAD_NUM];
  cout << "processor num: " << num << endl;
  int tid[THREAD_NUM];
  int thread_num = THREAD_NUM;

  CPU_ZERO(&cpuset);
  CPU_SET(0, &cpuset);  // use cpu0
  CPU_SET(1, &cpuset);  // use cpu1
  CPU_SET(2, &cpuset);  // use cpu2
  CPU_SET(3, &cpuset);  // use cpu3
  int s = sched_setaffinity(0, sizeof(cpuset), &cpuset);
  if (s == -1) {
    cout << "this process could not set cpu affinity\n";
  }
  CPU_ZERO(&cpuset);
  s = sched_getaffinity(0, sizeof(cpuset), &cpuset);
  if (s == -1) {
    cout << "this process could not get cpu affinity\n";
  }
  for (int i = 0; i < CPU_SETSIZE; ++i)
    if (CPU_ISSET(i, &cpuset)) cout << "this process run in cpu " << i << endl;

  for (int i = 0; i < thread_num; ++i) {
    tid[i] = i;
    pthread_create(&thread[i], NULL, threadFun, (void *)&tid[i]);
    CPU_ZERO(&cpuset);
    CPU_SET(i, &cpuset);
    int s = pthread_setaffinity_np(thread[i], sizeof(cpu_set_t), &cpuset);
    if (s != 0)
      handle_error_en(s, "pthread_setaffinity_np");
  }
  for (int i = 0; i < thread_num; ++i) {
    CPU_ZERO(&cpuset);
    int s = pthread_getaffinity_np(thread[i], sizeof(cpu_set_t), &cpuset);
    if (s != 0)
      handle_error_en(s, "pthread_getaffinity_np");
    for (int j = 0; j < CPU_SETSIZE; ++j)
      if (CPU_ISSET(j, &cpuset)) cout << "thread " << i << " run in cpu " << j << endl;
  }
  for (int i = 0; i < thread_num; ++i) {
    pthread_join(thread[i], NULL);
  }

  double sum = 0;
  for (int i = 0; i < thread_num; ++i) {
    sum += t[i];
  }
  double avg_time = sum / THREAD_NUM;
  double circle_count = 0x10000000/1e9;
  cout << int(1000 * avg_time) << endl;
  cout << 16 * 4 * circle_count / avg_time << endl;
  cout << 16 * 4 * 1.80 << endl;

  return 0;
}
