#include "common_function.h"
#include <vector>
#include <iostream>
#include <random>
#include <thread>
#include <iomanip>

#ifdef _WIN32
#include "windows.h"
#else
#include <unistd.h>
#endif // _WIN32

using namespace std;
using namespace chrono;

template<typename dtype>
void Print1DVector(const vector<dtype>& array) {
    for (dtype val : array) cout << val << '\t';
    cout << endl;
}

void PrintNowTime() {
    time_t now = time(0);
    tm *ltm = localtime(&now);

    cout << "second from year 1970: " << now << endl;
    cout << "year.month.day hour:minute:second --> "
         << 1900 + ltm->tm_year << "."
         << 1 + ltm->tm_mon << "."
         << ltm->tm_mday << " "
         << ltm->tm_hour << ":"
         << ltm->tm_min << ":"
         << ltm->tm_sec << endl;
}

TimerClock::TimerClock(bool use_destruct_print_time) {
    use_destruct_print_time_ = use_destruct_print_time;
    update();
}

TimerClock::~TimerClock() {
    if (use_destruct_print_time_) {
        double used_milli_second = get_milli_second();
        cout << "used " << fixed << setprecision(6) << used_milli_second << " ms" << endl;
    }
}

void TimerClock::update() {
    start_ = high_resolution_clock::now();
}

double TimerClock::get_second() {
    return double(get_micro_second()) / 1000000;
}

double TimerClock::get_milli_second() {
    return double(get_micro_second()) / 1000;
}

long int TimerClock::get_micro_second() {
    return duration_cast<microseconds>(high_resolution_clock::now() - start_).count();
}


template <typename dtype>
vector<dtype> GenerateRandom1DIntegerArray(unsigned int seed, int min, int max, int length) {
    vector<dtype> array(length);
//    static default_random_engine e(seed);
//    static uniform_int_distribution<dtype> u(min, max);
    default_random_engine e(seed);
    uniform_int_distribution<int> u(min, max);
    for (int i = 0; i < length; ++i) {
        array[i] = u(e);
    }
    return array;
}
template <typename dtype>
vector<dtype> GenerateRandom1DIntegerArray(int min, int max, int length) {
    vector<dtype> array(length);
    std::random_device rd;
    default_random_engine e(rd());
    uniform_int_distribution<dtype> u(min, max);
    for (int i = 0; i < length; ++i) {
        array[i] = u(e);
    }
    return array;
}

template <typename dtype>
vector<dtype> GenerateRandom1DRealArray(unsigned int seed, int min, int max, int length) {
    vector<dtype> array(length);
    static default_random_engine e(seed);
    static uniform_real_distribution<dtype> u(min, max);
    for (int i = 0; i < length; ++i) {
        array[i] = u(e);
    }
    return array;
}
template <typename dtype>
vector<dtype> GenerateRandom1DRealArray(int min, int max, int length) {
    vector<dtype> array(length);
    std::random_device rd;
    default_random_engine e(rd());
    uniform_real_distribution<dtype> u(min, max);
    for (int i = 0; i < length; ++i) {
        array[i] = u(e);
    }
    return array;
}

vector<int> GenerateRandom1DIntegerArrayUseRandSideOpen(unsigned int seed, int min, int max, int length) {
    vector<int> array(length, 0);
    if (max - min < 2) return array;

    srand(seed);
    for (int i = 0; i < length; ++i) {
        array[i] = rand() % (max - min - 1) + min + 1;
    }
    return array;
}
vector<int> GenerateRandom1DIntegerArrayUseRandSideOpen(int min, int max, int length) {
    vector<int> array(length, 0);
    if (max - min < 2) return array;

    unsigned int seed = time(nullptr);
    srand(seed);
    for (int i = 0; i < length; ++i) {
        array[i] = rand() % (max - min - 1) + min + 1;
    }
    return array;
}

vector<int> GenerateRandom1DIntegerArrayUseRandSideClose(unsigned int seed, int min, int max, int length) {
    vector<int> array(length, 0);
    if (max - min < 0) return array;

    srand(seed);
    for (int i = 0; i < length; ++i) {
        array[i] = rand() % (max - min + 1) + min;
    }
    return array;
}
vector<int> GenerateRandom1DIntegerArrayUseRandSideClose(int min, int max, int length) {
    vector<int> array(length, 0);
    if (max - min < 0) return array;

    unsigned int seed = time(nullptr);
    srand(seed);
    for (int i = 0; i < length; ++i) {
        array[i] = rand() % (max - min + 1) + min;
    }
    return array;
}

vector<int> GenerateRandom1DIntegerArrayUseRandLeftClose(unsigned int seed, int min, int max, int length) {
    vector<int> array(length, 0);
    if (max - min < 1) return array;

    srand(seed);
    for (int i = 0; i < length; ++i) {
        array[i] = rand() % (max - min) + min;
    }
    return array;
}
vector<int> GenerateRandom1DIntegerArrayUseRandLeftClose(int min, int max, int length) {
    vector<int> array(length, 0);
    if (max - min < 1) return array;

    unsigned int seed = time(nullptr);
    srand(seed);
    for (int i = 0; i < length; ++i) {
        array[i] = rand() % (max - min) + min;
    }
    return array;
}

vector<int> GenerateRandom1DIntegerArrayUseRandRightClose(unsigned int seed, int min, int max, int length) {
    vector<int> array(length, 0);
    if (max - min < 1) return array;

    srand(seed);
    for (int i = 0; i < length; ++i) {
        array[i] = rand() % (max - min) + min + 1;
    }
    return array;
}
vector<int> GenerateRandom1DIntegerArrayUseRandRightClose(int min, int max, int length) {
    vector<int> array(length, 0);
    if (max - min < 1) return array;

    unsigned int seed = time(nullptr);
    srand(seed);
    for (int i = 0; i < length; ++i) {
        array[i] = rand() % (max - min) + min + 1;
    }
    return array;
}

template <typename dtype>
vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandSideOpen(unsigned int seed, int length) {
    vector<dtype> array(length, 0);

    srand(seed);
    int random_int;
    dtype val;
    for (int i = 0; i < length; ++i) {
        do {
            random_int = rand();
            val = (random_int - 1) / dtype(RAND_MAX);
        } while (random_int <= 1);
        array[i] = val;
    }
    return array;
}
template <typename dtype>
vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandSideOpen(int length) {
    vector<dtype> array(length, 0);

    unsigned int seed = time(nullptr);
    srand(seed);
    int random_int;
    dtype val;
    for (int i = 0; i < length; ++i) {
        do {
            random_int = rand();
            val = (random_int - 1) / dtype(RAND_MAX);
        } while (random_int <= 1);
        array[i] = val;
    }
    return array;
}

template <typename dtype>
vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandSideClose(unsigned int seed, int length) {
    vector<dtype> array(length, 0);

    srand(seed);
    dtype val;
    for (int i = 0; i < length; ++i) {
        val = rand() / dtype(RAND_MAX);
        array[i] = val;
    }
    return array;
}
template <typename dtype>
vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandSideClose(int length) {
    vector<dtype> array(length, 0);

    unsigned int seed = time(nullptr);
    srand(seed);
    dtype val;
    for (int i = 0; i < length; ++i) {
        val = rand() / dtype(RAND_MAX);
        array[i] = val;
    }
    return array;
}

template <typename dtype>
vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandLeftClose(unsigned int seed, int length) {
    vector<dtype> array(length, 0);

    srand(seed);
    int random_int;
    dtype val;
    for (int i = 0; i < length; ++i) {
        do {
            random_int = rand();
            val = (random_int - 1) / dtype(RAND_MAX);
        } while (random_int < 1);
        array[i] = val;
    }
    return array;
}
template <typename dtype>
vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandLeftClose(int length) {
    vector<dtype> array(length, 0);

    unsigned int seed = time(nullptr);
    srand(seed);
    int random_int;
    dtype val;
    for (int i = 0; i < length; ++i) {
        do {
            random_int = rand();
            val = (random_int - 1) / dtype(RAND_MAX);
        } while (random_int < 1);
        array[i] = val;
    }
    return array;
}

template <typename dtype>
vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandRightClose(unsigned int seed, int length) {
    vector<dtype> array(length, 0);

    srand(seed);
    int random_int;
    dtype val;
    for (int i = 0; i < length; ++i) {
        do {
            random_int = rand();
            val = random_int / dtype(RAND_MAX);
        } while (random_int < 1);
        array[i] = val;
    }
    return array;
}
template <typename dtype>
vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandRightClose(int length) {
    vector<dtype> array(length, 0);

    unsigned int seed = time(nullptr);
    srand(seed);
    int random_int;
    dtype val;
    for (int i = 0; i < length; ++i) {
        do {
            random_int = rand();
            val = random_int / dtype(RAND_MAX);
        } while (random_int < 1);
        array[i] = val;
    }
    return array;
}

template <typename dtype>
vector<dtype> GenerateRandom1DRealArrayUseRandSideOpen(unsigned int seed, dtype min, dtype max, int length) {
    vector<dtype> array(length, 0);
    if (max - min < 0) return array;

    array = GenerateRandom1DRealZeroOneArrayUseRandSideOpen<dtype>(seed, length);
    for (int i = 0; i < array.size(); ++i) {
        array[i] = min + array[i] * (max - min);
    }
    return array;
}
template <typename dtype>
vector<dtype> GenerateRandom1DRealArrayUseRandSideOpen(dtype min, dtype max, int length) {
    vector<dtype> array(length, 0);
    if (max - min < 0) return array;

    array = GenerateRandom1DRealZeroOneArrayUseRandSideOpen<dtype>(length);
    for (int i = 0; i < array.size(); ++i) {
        array[i] = min + array[i] * (max - min);
    }
    return array;
}

template <typename dtype>
vector<dtype> GenerateRandom1DRealArrayUseRandSideClose(unsigned int seed, dtype min, dtype max, int length) {
    vector<dtype> array(length, 0);
    if (max - min < 0) return array;

    array = GenerateRandom1DRealZeroOneArrayUseRandSideClose<dtype>(seed, length);
    for (int i = 0; i < array.size(); ++i) {
        array[i] = min + array[i] * (max - min);
    }
    return array;
}
template <typename dtype>
vector<dtype> GenerateRandom1DRealArrayUseRandSideClose(dtype min, dtype max, int length) {
    vector<dtype> array(length, 0);
    if (max - min < 0) return array;

    array = GenerateRandom1DRealZeroOneArrayUseRandSideClose<dtype>(length);
    for (int i = 0; i < array.size(); ++i) {
        array[i] = min + array[i] * (max - min);
    }
    return array;
}

template <typename dtype>
vector<dtype> GenerateRandom1DRealArrayUseRandLeftClose(unsigned int seed, dtype min, dtype max, int length) {
    vector<dtype> array(length, 0);
    if (max - min < 0) return array;

    array = GenerateRandom1DRealZeroOneArrayUseRandLeftClose<dtype>(seed, length);
    for (int i = 0; i < array.size(); ++i) {
        array[i] = min + array[i] * (max - min);
    }
    return array;
}
template <typename dtype>
vector<dtype> GenerateRandom1DRealArrayUseRandLeftClose(dtype min, dtype max, int length) {
    vector<dtype> array(length, 0);
    if (max - min < 0) return array;

    array = GenerateRandom1DRealZeroOneArrayUseRandLeftClose<dtype>(length);
    for (int i = 0; i < array.size(); ++i) {
        array[i] = min + array[i] * (max - min);
    }
    return array;
}

template <typename dtype>
vector<dtype> GenerateRandom1DRealArrayUseRandRightClose(unsigned int seed, dtype min, dtype max, int length) {
    vector<dtype> array(length, 0);
    if (max - min < 0) return array;

    array = GenerateRandom1DRealZeroOneArrayUseRandRightClose<dtype>(seed, length);
    for (int i = 0; i < array.size(); ++i) {
        array[i] = min + array[i] * (max - min);
    }
    return array;
}
template <typename dtype>
vector<dtype> GenerateRandom1DRealArrayUseRandRightClose(dtype min, dtype max, int length) {
    vector<dtype> array(length, 0);
    if (max - min < 0) return array;

    array = GenerateRandom1DRealZeroOneArrayUseRandRightClose<dtype>(length);
    for (int i = 0; i < array.size(); ++i) {
        array[i] = min + array[i] * (max - min);
    }
    return array;
}

void SleepCrossPlatform(unsigned int millisecond) {
#ifdef _WIN32
    Sleep(millisecond);
#else
    usleep(millisecond * 1000);
#endif // _WIN32
}

//void SleepCrossPlatformUseSTL(unsigned int millisecond) {
//    this_thread::sleep_for(chrono::milliseconds(millisecond));
//}


template void Print1DVector(const vector<int>& array);
template void Print1DVector(const vector<unsigned int>& array);
template void Print1DVector(const vector<long int>& array);
template void Print1DVector(const vector<unsigned long int>& array);
template void Print1DVector(const vector<long long int>& array);
template void Print1DVector(const vector<unsigned long long int>& array);
template void Print1DVector(const vector<float>& array);
template void Print1DVector(const vector<double>& array);

template vector<int> GenerateRandom1DIntegerArray(unsigned int seed, int min, int max, int length);
template vector<unsigned int> GenerateRandom1DIntegerArray(unsigned int seed, int min, int max, int length);
template vector<long int> GenerateRandom1DIntegerArray(unsigned int seed, int min, int max, int length);
template vector<unsigned long int> GenerateRandom1DIntegerArray(unsigned int seed, int min, int max, int length);
template vector<long long int> GenerateRandom1DIntegerArray(unsigned int seed, int min, int max, int length);
template vector<unsigned long long int> GenerateRandom1DIntegerArray(unsigned int seed, int min, int max, int length);
template vector<int> GenerateRandom1DIntegerArray(int min, int max, int length);
template vector<unsigned int> GenerateRandom1DIntegerArray(int min, int max, int length);
template vector<long int> GenerateRandom1DIntegerArray(int min, int max, int length);
template vector<unsigned long int> GenerateRandom1DIntegerArray(int min, int max, int length);
template vector<long long int> GenerateRandom1DIntegerArray(int min, int max, int length);
template vector<unsigned long long int> GenerateRandom1DIntegerArray(int min, int max, int length);

template vector<float> GenerateRandom1DRealArray(unsigned int seed, int min, int max, int length);
template vector<double> GenerateRandom1DRealArray(unsigned int seed, int min, int max, int length);
template vector<float> GenerateRandom1DRealArray(int min, int max, int length);
template vector<double> GenerateRandom1DRealArray(int min, int max, int length);

template vector<float> GenerateRandom1DRealZeroOneArrayUseRandSideOpen(unsigned int seed, int length);
template vector<double> GenerateRandom1DRealZeroOneArrayUseRandSideOpen(unsigned int seed, int length);
template vector<float> GenerateRandom1DRealZeroOneArrayUseRandSideOpen(int length);
template vector<double> GenerateRandom1DRealZeroOneArrayUseRandSideOpen(int length);

template vector<float> GenerateRandom1DRealZeroOneArrayUseRandSideClose(unsigned int seed, int length);
template vector<double> GenerateRandom1DRealZeroOneArrayUseRandSideClose(unsigned int seed, int length);
template vector<float> GenerateRandom1DRealZeroOneArrayUseRandSideClose(int length);
template vector<double> GenerateRandom1DRealZeroOneArrayUseRandSideClose(int length);

template vector<float> GenerateRandom1DRealZeroOneArrayUseRandLeftClose(unsigned int seed, int length);
template vector<double> GenerateRandom1DRealZeroOneArrayUseRandLeftClose(unsigned int seed, int length);
template vector<float> GenerateRandom1DRealZeroOneArrayUseRandLeftClose(int length);
template vector<double> GenerateRandom1DRealZeroOneArrayUseRandLeftClose(int length);

template vector<float> GenerateRandom1DRealZeroOneArrayUseRandRightClose(unsigned int seed, int length);
template vector<double> GenerateRandom1DRealZeroOneArrayUseRandRightClose(unsigned int seed, int length);
template vector<float> GenerateRandom1DRealZeroOneArrayUseRandRightClose(int length);
template vector<double> GenerateRandom1DRealZeroOneArrayUseRandRightClose(int length);

template vector<float> GenerateRandom1DRealArrayUseRandSideOpen(unsigned int seed, float min, float max, int length);
template vector<double> GenerateRandom1DRealArrayUseRandSideOpen(unsigned int seed, double min, double max, int length);
template vector<float> GenerateRandom1DRealArrayUseRandSideOpen(float min, float max, int length);
template vector<double> GenerateRandom1DRealArrayUseRandSideOpen(double min, double max, int length);

template vector<float> GenerateRandom1DRealArrayUseRandSideClose(unsigned int seed, float min, float max, int length);
template vector<double> GenerateRandom1DRealArrayUseRandSideClose(unsigned int seed, double min, double max, int length);
template vector<float> GenerateRandom1DRealArrayUseRandSideClose(float min, float max, int length);
template vector<double> GenerateRandom1DRealArrayUseRandSideClose(double min, double max, int length);

template vector<float> GenerateRandom1DRealArrayUseRandLeftClose(unsigned int seed, float min, float max, int length);
template vector<double> GenerateRandom1DRealArrayUseRandLeftClose(unsigned int seed, double min, double max, int length);
template vector<float> GenerateRandom1DRealArrayUseRandLeftClose(float min, float max, int length);
template vector<double> GenerateRandom1DRealArrayUseRandLeftClose(double min, double max, int length);

template vector<float> GenerateRandom1DRealArrayUseRandRightClose(unsigned int seed, float min, float max, int length);
template vector<double> GenerateRandom1DRealArrayUseRandRightClose(unsigned int seed, double min, double max, int length);
template vector<float> GenerateRandom1DRealArrayUseRandRightClose(float min, float max, int length);
template vector<double> GenerateRandom1DRealArrayUseRandRightClose(double min, double max, int length);
