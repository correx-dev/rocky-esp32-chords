// Estructuras de datos propias del firmware (sin librerías de estructuras).
// Son las mismas ideas que Python/estructuras.py. Se prueban en el PC: tests/test_firmware_host.cpp
//
//  RingBuffer<T,N>   buffer circular de un productor y un consumidor (tarea de adquisición -> tarea de IA)
//  VotingWindow<N,C> ventana de votación con histéresis para estabilizar el acorde
//  RunningStats      media y desviación estándar en línea (algoritmo de Welford): sin guardar las muestras
#pragma once
#include <stdint.h>
#include <atomic>

// ---------------------------------------------------------------------------------------------
// Buffer circular. N debe ser potencia de 2: el índice se obtiene con una máscara (más barato que %).
// Un solo productor (push) y un solo consumidor (last) en núcleos distintos: no necesita mutex.
// El productor escribe los datos y DESPUÉS publica el contador `total` (release); el consumidor lo lee
// con acquire, así nunca ve un contador adelantado a los datos.
//   push(x, n)  O(n)       last(dst, n)  O(n)       total()  O(1)       memoria: N * sizeof(T), fija
template <typename T, uint32_t N>
class RingBuffer {
  static_assert((N & (N - 1)) == 0, "N debe ser potencia de 2");

 public:
  void push(const T *x, uint32_t n) {
    uint32_t t = total_.load(std::memory_order_relaxed);
    for (uint32_t i = 0; i < n; i++) data_[(t + i) & (N - 1)] = x[i];
    total_.store(t + n, std::memory_order_release);
  }
  uint32_t total() const { return total_.load(std::memory_order_acquire); }
  // Copia las n muestras más recientes (orden cronológico). Devuelve false si aún no hay n.
  bool last(T *dst, uint32_t n) const {
    uint32_t t = total_.load(std::memory_order_acquire);
    if (t < n || n > N) return false;
    for (uint32_t i = 0; i < n; i++) dst[i] = data_[(t - n + i) & (N - 1)];
    return true;
  }

 private:
  T data_[N];
  std::atomic<uint32_t> total_{0};
};

// ---------------------------------------------------------------------------------------------
// Ventana de votación. Guarda las últimas N predicciones (clase, confianza) en un arreglo circular.
// Acorde estable = la clase con más votos si tiene >= minimo votos y su confianza media >= confMin.
// Si ninguna cumple, se conserva el acorde estable anterior (histéresis: evita el parpadeo).
//   add() O(1)     vote() O(N + C)     memoria fija
template <int N, int C>
class VotingWindow {
 public:
  VotingWindow(int minimo = 3, float confMin = 0.5f) : minimo_(minimo), confMin_(confMin) {
    for (int i = 0; i < N; i++) { cls_[i] = -1; conf_[i] = 0; }
  }
  void add(int clase, float conf) {
    cls_[head_] = clase;
    conf_[head_] = conf;
    head_ = (head_ + 1) % N;
    if (count_ < N) count_++;
  }
  // Devuelve la clase estable (o -1) y la fracción de votos de esa clase en la ventana.
  int vote(float *fraccion) {
    int votos[C] = {0};
    float suma[C] = {0};
    for (int i = 0; i < count_; i++) {
      int c = cls_[i];
      if (c >= 0 && c < C) { votos[c]++; suma[c] += conf_[i]; }
    }
    int mejor = 0;
    for (int k = 1; k < C; k++)
      if (votos[k] > votos[mejor] || (votos[k] == votos[mejor] && suma[k] > suma[mejor])) mejor = k;
    if (votos[mejor] >= minimo_ && suma[mejor] / votos[mejor] >= confMin_) estable_ = mejor;
    *fraccion = (estable_ >= 0 && count_ > 0) ? (float)votos[estable_] / count_ : 0.0f;
    return estable_;
  }

 private:
  int cls_[N];
  float conf_[N];
  int head_ = 0, count_ = 0, estable_ = -1;
  int minimo_;
  float confMin_;
};

// ---------------------------------------------------------------------------------------------
// Media y desviación estándar (muestral) en una sola pasada y O(1) de memoria (Welford).
// Se usa para la latencia por inferencia: >= 100 inferencias para el requisito R7.
struct RunningStats {
  uint32_t n = 0;
  double mean = 0, m2 = 0;
  void add(double x) {
    n++;
    double d = x - mean;
    mean += d / n;
    m2 += d * (x - mean);
  }
  double stddev() const { return n > 1 ? __builtin_sqrt(m2 / (n - 1)) : 0.0; }
};
