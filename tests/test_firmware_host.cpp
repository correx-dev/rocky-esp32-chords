// Prueba en el PC del código C del firmware (sin ESP32).
//   g++ -O2 -std=c++17 -I../Firmware/rocky_ia tests/test_firmware_host.cpp -o /tmp/test_fw
//   /tmp/test_fw estructuras            -> pruebas de las estructuras
//   /tmp/test_fw features in.raw out.f32 -> características de 16000 muestras int16 (para comparar con Python)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "estructuras.h"
#include "ia_dsp.h"

static int fallos = 0;
#define CHECK(c) do { if (!(c)) { printf("FALLO linea %d: %s\n", __LINE__, #c); fallos++; } } while (0)

static IaDsp dsp;

int main(int argc, char **argv) {
  if (argc >= 2 && !strcmp(argv[1], "estructuras")) {
    static RingBuffer<int16_t, 64> rb;
    std::vector<int16_t> ref;
    unsigned s = 1;
    for (int it = 0; it < 500; it++) {
      s = s * 1103515245u + 12345u;
      int n = 1 + (s >> 16) % 40;
      std::vector<int16_t> x(n);
      for (int i = 0; i < n; i++) { x[i] = (int16_t)(s >> 3) + i; ref.push_back(x[i]); }
      rb.push(x.data(), n);
      for (int m : {1, 10, 33, 64}) {
        int16_t dst[64];
        bool ok = rb.last(dst, m);
        CHECK(ok == ((int)ref.size() >= m));
        if (ok) for (int i = 0; i < m; i++) CHECK(dst[i] == ref[ref.size() - m + i]);
      }
    }
    VotingWindow<5, 11> v(3, 0.5f);
    float fr;
    for (int c : {0, 0, 0, 1, 2}) v.add(c, 0.9f);
    CHECK(v.vote(&fr) == 0);
    v.add(1, 0.9f); v.add(2, 0.9f);
    CHECK(v.vote(&fr) == 0);               // nadie llega a 3 votos: se mantiene
    v.add(1, 0.9f); v.add(1, 0.9f);
    CHECK(v.vote(&fr) == 1);
    VotingWindow<5, 11> w(3, 0.6f);
    for (int i = 0; i < 5; i++) w.add(4, 0.3f);
    CHECK(w.vote(&fr) == -1);
    RunningStats st;
    double xs[] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (double x : xs) st.add(x);
    CHECK(st.mean == 4.5);
    CHECK(__builtin_fabs(st.stddev() - 2.449489742783178) < 1e-9);
    printf(fallos ? "estructuras: %d FALLOS\n" : "estructuras: OK\n", fallos);
    return fallos != 0;
  }
  if (argc >= 4 && !strcmp(argv[1], "features")) {
    std::vector<int16_t> x(IA_MUESTRAS);
    FILE *f = fopen(argv[2], "rb");
    if (!f || fread(x.data(), 2, IA_MUESTRAS, f) != (size_t)IA_MUESTRAS) { printf("no pude leer %s\n", argv[2]); return 2; }
    fclose(f);
    dsp.init();
    static float out[IA_TRAMAS * IA_BANDAS];
    dsp.caracteristicas(x.data(), out);
    f = fopen(argv[3], "wb");
    fwrite(out, sizeof(float), IA_TRAMAS * IA_BANDAS, f);
    fclose(f);
    return 0;
  }
  printf("uso: test_fw estructuras | features in.raw out.f32\n");
  return 2;
}
