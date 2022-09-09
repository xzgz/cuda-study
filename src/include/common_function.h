#ifndef C_PLUS_STUDY_INCLUDE_COMMON_FUNCTION_H
#define C_PLUS_STUDY_INCLUDE_COMMON_FUNCTION_H

#include <vector>
#include <chrono>
//#include <iostream>
//#include <iomanip>

//using namespace std;
//using namespace chrono;

template<typename dtype>
void Print1DVector(const std::vector<dtype>& array);

void PrintNowTime();

class TimerClock {
public:
    TimerClock(bool use_destruct_print_time);

    ~TimerClock();

    void update();

    double get_second();
    double get_milli_second();
    long int get_micro_second();
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    bool use_destruct_print_time_;
};

//class TimerClock {
//public:
//    TimerClock(bool use_destruct_print_time) {
//        use_destruct_print_time_ = use_destruct_print_time;
//        update();
//    }
//
//    ~TimerClock() {
//        if (use_destruct_print_time_) {
//            double used_milli_second = get_milli_second();
//            cout << "used " << fixed << setprecision(6) << used_milli_second << " ms" << endl;
//        }
//    }
//
//    void update() {
//        start_ = high_resolution_clock::now();
//    }
//
//    double get_second() {
//        return double(get_micro_second()) / 1000000;
//    }
//    double get_milli_second() {
//        return double(get_micro_second()) / 1000;
//    }
//    long int get_micro_second() {
//        return duration_cast<microseconds>(high_resolution_clock::now() - start_).count();
//    }
//private:
//    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
//    bool use_destruct_print_time_;
//};

template <typename dtype>
std::vector<dtype> GenerateRandom1DIntegerArray(unsigned int seed, int min, int max, int length);
template <typename dtype>
std::vector<dtype> GenerateRandom1DIntegerArray(int min, int max, int length);

template <typename dtype>
std::vector<dtype> GenerateRandom1DRealArray(unsigned int seed, int min, int max, int length);
template <typename dtype>
std::vector<dtype> GenerateRandom1DRealArray(int min, int max, int length);


std::vector<int> GenerateRandom1DIntegerArrayUseRandSideOpen(unsigned int seed, int min, int max, int length);
std::vector<int> GenerateRandom1DIntegerArrayUseRandSideOpen(int min, int max, int length);

std::vector<int> GenerateRandom1DIntegerArrayUseRandSideClose(unsigned int seed, int min, int max, int length);
std::vector<int> GenerateRandom1DIntegerArrayUseRandSideClose(int min, int max, int length);

std::vector<int> GenerateRandom1DIntegerArrayUseRandLeftClose(unsigned int seed, int min, int max, int length);
std::vector<int> GenerateRandom1DIntegerArrayUseRandLeftClose(int min, int max, int length);

std::vector<int> GenerateRandom1DIntegerArrayUseRandRightClose(unsigned int seed, int min, int max, int length);
std::vector<int> GenerateRandom1DIntegerArrayUseRandRightClose(int min, int max, int length);


template <typename dtype>
std::vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandSideOpen(unsigned int seed, int length);
template <typename dtype>
std::vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandSideOpen(int length);

template <typename dtype>
std::vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandSideClose(unsigned int seed, int length);
template <typename dtype>
std::vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandSideClose(int length);

template <typename dtype>
std::vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandLeftClose(unsigned int seed, int length);
template <typename dtype>
std::vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandLeftClose(int length);

template <typename dtype>
std::vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandRightClose(unsigned int seed, int length);
template <typename dtype>
std::vector<dtype> GenerateRandom1DRealZeroOneArrayUseRandRightClose(int length);


template <typename dtype>
std::vector<dtype> GenerateRandom1DRealArrayUseRandSideOpen(unsigned int seed, dtype min, dtype max, int length);
template <typename dtype>
std::vector<dtype> GenerateRandom1DRealArrayUseRandSideOpen(dtype min, dtype max, int length);

template <typename dtype>
std::vector<dtype> GenerateRandom1DRealArrayUseRandSideClose(unsigned int seed, dtype min, dtype max, int length);
template <typename dtype>
std::vector<dtype> GenerateRandom1DRealArrayUseRandSideClose(dtype min, dtype max, int length);

template <typename dtype>
std::vector<dtype> GenerateRandom1DRealArrayUseRandLeftClose(unsigned int seed, dtype min, dtype max, int length);
template <typename dtype>
std::vector<dtype> GenerateRandom1DRealArrayUseRandLeftClose(dtype min, dtype max, int length);

template <typename dtype>
std::vector<dtype> GenerateRandom1DRealArrayUseRandRightClose(unsigned int seed, dtype min, dtype max, int length);
template <typename dtype>
std::vector<dtype> GenerateRandom1DRealArrayUseRandRightClose(dtype min, dtype max, int length);

void SleepCrossPlatform(unsigned int millisecond);
void SleepCrossPlatformUseSTL(unsigned int millisecond);

#endif // C_PLUS_STUDY_INCLUDE_COMMON_FUNCTION_H
