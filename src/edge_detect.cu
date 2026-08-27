#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <vector>

bool readPgm(const char* path, unsigned char** data, int* width, int* height) {
  FILE* f = fopen(path, "rb");
  if (f == nullptr) {
    return false;
  }
  char magic[3] = {0};
  if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
    fclose(f);
    return false;
  }
  int w = 0;
  int h = 0;
  int maxval = 0;
  if (fscanf(f, "%d %d %d", &w, &h, &maxval) != 3) {
    fclose(f);
    return false;
  }
  fgetc(f);
  unsigned char* buf = static_cast<unsigned char*>(malloc(w * h));
  if (fread(buf, 1, w * h, f) != static_cast<size_t>(w * h)) {
    free(buf);
    fclose(f);
    return false;
  }
  fclose(f);
  *data = buf;
  *width = w;
  *height = h;
  return true;
}

bool writePgm(const char* path, const unsigned char* data, int width,
              int height) {
  FILE* f = fopen(path, "wb");
  if (f == nullptr) {
    return false;
  }
  fprintf(f, "P5\n%d %d\n255\n", width, height);
  fwrite(data, 1, width * height, f);
  fclose(f);
  return true;
}

std::vector<std::string> listPgmFiles(const std::string& dir) {
  std::vector<std::string> files;
  DIR* d = opendir(dir.c_str());
  if (d == nullptr) {
    return files;
  }
  struct dirent* entry;
  while ((entry = readdir(d)) != nullptr) {
    std::string name = entry->d_name;
    if (name.size() > 4 && name.substr(name.size() - 4) == ".pgm") {
      files.push_back(name);
    }
  }
  closedir(d);
  std::sort(files.begin(), files.end());
  return files;
}

__global__ void sobelKernel(const unsigned char* in, unsigned char* out,
                            int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    out[y * width + x] = 0;
    return;
  }
  int gx = -in[(y - 1) * width + (x - 1)] + in[(y - 1) * width + (x + 1)]
           - 2 * in[y * width + (x - 1)] + 2 * in[y * width + (x + 1)]
           - in[(y + 1) * width + (x - 1)] + in[(y + 1) * width + (x + 1)];
  int gy = -in[(y - 1) * width + (x - 1)] - 2 * in[(y - 1) * width + x]
           - in[(y - 1) * width + (x + 1)] + in[(y + 1) * width + (x - 1)]
           + 2 * in[(y + 1) * width + x] + in[(y + 1) * width + (x + 1)];
  int mag = static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy)));
  out[y * width + x] = static_cast<unsigned char>(mag > 255 ? 255 : mag);
}

void sobelCpu(const unsigned char* in, unsigned char* out, int width,
              int height) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        out[y * width + x] = 0;
        continue;
      }
      int gx = -in[(y - 1) * width + (x - 1)] + in[(y - 1) * width + (x + 1)]
               - 2 * in[y * width + (x - 1)] + 2 * in[y * width + (x + 1)]
               - in[(y + 1) * width + (x - 1)] + in[(y + 1) * width + (x + 1)];
      int gy = -in[(y - 1) * width + (x - 1)] - 2 * in[(y - 1) * width + x]
               - in[(y - 1) * width + (x + 1)] + in[(y + 1) * width + (x - 1)]
               + 2 * in[(y + 1) * width + x] + in[(y + 1) * width + (x + 1)];
      int mag = static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy)));
      out[y * width + x] = static_cast<unsigned char>(mag > 255 ? 255 : mag);
    }
  }
}

int main(int argc, char** argv) {
  std::string input_dir = "data/input";
  std::string output_dir = "data/output";
  int max_images = 0;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-i" && i + 1 < argc) {
      input_dir = argv[++i];
    } else if (arg == "-o" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "-n" && i + 1 < argc) {
      max_images = atoi(argv[++i]);
    }
  }

  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  printf("Number of GPU devices: %d\n", device_count);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Using device 0: %s\n", prop.name);

  std::vector<std::string> files = listPgmFiles(input_dir);
  if (max_images > 0 && static_cast<int>(files.size()) > max_images) {
    files.resize(max_images);
  }
  printf("Found %d input images in %s\n", static_cast<int>(files.size()),
         input_dir.c_str());

  mkdir(output_dir.c_str(), 0755);

  double cpu_total_ms = 0.0;
  double gpu_total_ms = 0.0;
  int processed = 0;
  bool all_match = true;

  for (const std::string& name : files) {
    unsigned char* h_in = nullptr;
    int w = 0;
    int h = 0;
    std::string in_path = input_dir + "/" + name;
    if (!readPgm(in_path.c_str(), &h_in, &w, &h)) {
      printf("Skipping %s, unreadable file\n", name.c_str());
      continue;
    }
    size_t size = static_cast<size_t>(w) * h;

    unsigned char* h_cpu = static_cast<unsigned char*>(malloc(size));
    auto t0 = std::chrono::high_resolution_clock::now();
    sobelCpu(h_in, h_cpu, w, h);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    cpu_total_ms += cpu_ms;

    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sobelKernel<<<grid, block>>>(d_in, d_out, w, h);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    gpu_total_ms += gpu_ms;

    unsigned char* h_gpu = static_cast<unsigned char*>(malloc(size));
    cudaMemcpy(h_gpu, d_out, size, cudaMemcpyDeviceToHost);

    bool match = true;
    for (size_t i = 0; i < size; i++) {
      if (h_cpu[i] != h_gpu[i]) {
        match = false;
        break;
      }
    }
    if (!match) {
      all_match = false;
    }

    std::string out_path = output_dir + "/" + name;
    writePgm(out_path.c_str(), h_gpu, w, h);
    printf("Processed %s (%dx%d) cpu %.3f ms gpu %.3f ms match %s\n",
           name.c_str(), w, h, cpu_ms, gpu_ms, match ? "yes" : "no");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_cpu);
    free(h_gpu);
    processed++;
  }

  printf("Processed %d images\n", processed);
  printf("Total CPU time %.3f ms, total GPU kernel time %.3f ms\n",
         cpu_total_ms, gpu_total_ms);
  if (gpu_total_ms > 0.0) {
    printf("GPU speedup over single threaded CPU: %.2fx\n",
           cpu_total_ms / gpu_total_ms);
  }
  printf("All CPU and GPU outputs match: %s\n", all_match ? "yes" : "no");

  cudaDeviceReset();
  return 0;
}
