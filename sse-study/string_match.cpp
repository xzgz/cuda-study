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

int normal_strstr(const char *src, const char *dest) {
  assert(src && dest);

  int src_len = strlen(src);
  int dest_len = strlen(dest);

  int i;
  int j;
  int k;

  for(i = 0; i < src_len; ++i) {
    k = i;
    for(j = 0; j < dest_len; ++j) {
      if(src[k] == dest[j]) {
        ++k;
      } else break;
    }
    if(j == dest_len) {
      return k - dest_len;
    }
  }
  return -1;
}

static void get_next(const char* dest, int next[]) {
  /* 计算回溯值 */

  int j = 0;
  int k = -1;
  next[0] = -1;
  int dest_len = strlen(dest);

  while(j < dest_len) {
    if(k == -1 || dest[j] == dest[k]) {
      ++j;
      ++k;
      if(dest[j] != dest[k]) {
        next[j] = k;
      } else {
        next[j] = next[k];
      }
      next[j] = k;
    } else {
      k = next[k];
    }
  }
}

int kmp(const char *src, const char* dest) {
  assert(src && dest);

  int i = 0;
  int j = 0;
  const int src_len = strlen(src);
  const int dest_len = strlen(dest);
  int next[dest_len];

  get_next(dest, next);

  i = 0;
  while(i < src_len && j < dest_len) {
    if(j == -1 || src[i] == dest[j]) {
      i++;
      j++;
    } else {
      j = next[j];
    }
  }
  return j >= dest_len ? i - dest_len : -1;
}

int sse_strstr(const char *src, const char *dest) {
  assert(src && dest);

  if(strlen(src) < strlen(dest)) {
    return -1;
  }

  const char *s = src;
  const char *d = dest;
  int cmp;
  int cmp_z;
  int cmp_c;
  int cmp_s;

  const char *s_16;
  const char *d_16;

  __m128i frag1;
  __m128i frag2;
  frag1 = _mm_loadu_si128((__m128i *)s);
  frag2 = _mm_loadu_si128((__m128i *)d);
  cmp_s = _mm_cmpistrs(frag2, frag1, 0xc);

  if(cmp_s) {
    /* strlen(dest) < 16 */
    do {
      frag1 = _mm_loadu_si128((__m128i *)s);
      cmp   = _mm_cmpistri(frag2, frag1, 0x0c);
      cmp_c = _mm_cmpistrc(frag2, frag1, 0x0c);
      cmp_z = _mm_cmpistrz(frag2, frag1, 0xc);

      if((!cmp) & cmp_c) break;
      s += cmp;
    } while(!cmp_z);

    if(!cmp_c) {
      return -1;
    }
    return s - src;
  } else {
    /* strlen(dest) >= 16 */
    do {
      frag1 = _mm_loadu_si128((__m128i *)s);
      frag2 = _mm_loadu_si128((__m128i *)d);
      cmp   = _mm_cmpistri(frag2, frag1, 0xc);
      cmp_z = _mm_cmpistrz(frag2, frag1, 0xc);
      cmp_s = _mm_cmpistrs(frag2, frag1, 0xc);

      if(cmp) {
        /* suffix or not match(cmp=16)*/
        s += cmp;
      } else {
        /* match */
        do {
          s_16  = s + 16;
          d_16  = d + 16;
          frag1 = _mm_loadu_si128((__m128i *)s_16);
          frag2 = _mm_loadu_si128((__m128i *)d_16);
          cmp   = _mm_cmpistri(frag2, frag1, 0xc);
          cmp_z = _mm_cmpistrz(frag2, frag1, 0xc);
          cmp_s = _mm_cmpistrs(frag2, frag1, 0xc);

          if(cmp) break;

        } while(!cmp_s && !cmp_z);
        if(!cmp) {
          return s - src;
        } else {
          s += 1;
          cmp_z = 0;
        }
      }
    } while(!cmp_z);
    return -1;
  }
}

int main(int argc, char *argv[]) {
#define BUF_SIZE (10 * 1024 * 1024)

  char *buf = (char *)malloc(sizeof(char) * BUF_SIZE);
  if(!buf) {
    std::cout << "failed malloc" << std::endl;
    return -1;
  }

  memset(buf, 'm', BUF_SIZE);

  const char *dest = "message=";

  memcpy(buf + BUF_SIZE - 1 - strlen(dest), dest, strlen(dest));

  buf[BUF_SIZE - 1] = '\0';

  const char *src = buf;
  int pos = 0;
  const char *p = nullptr;

  {
    Timer("strstr");
    for(int i = 0; i < 10; ++i) {
      p = strstr(src, dest);
    }
  }
  {
    Timer("normal_strstr");
    for(int i = 0; i < 10; ++i) {
      pos = normal_strstr(src, dest);
    }
  }
  {
    Timer("kmp");
    for(int i = 0; i < 10; ++i) {
      pos = kmp(src, dest);
    }
  }
  {
    Timer("sse_strstr");
    for(int i = 0; i < 10; ++i) {
      pos = sse_strstr(src, dest);
    }
  }

  cout << endl << "pos: " << pos << endl;
  if (pos) {
    for (int i = 0; i < 8; ++i) {
      cout << buf[pos+i];
    }
    cout << endl;
  }
  if (p) {
    for (int i = 0; i < 8; ++i) {
      cout << *(p+i);
    }
    cout << endl;
  }

  free(buf);
}
