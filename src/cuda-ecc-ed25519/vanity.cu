#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <ctime>
#include <assert.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>

#include "curand_kernel.h"
#include "ed25519.h"
#include "fixedint.h"
#include "gpu_common.h"
#include "gpu_ctx.h"

#include "keypair.cu"
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"
#include "../config.h"

/* -- Configurazione suffisso da trovare -- */
const char suffixes[][4] = {"pump"}; // Definisci qui i suffissi desiderati

typedef struct
{
	curandState *states[8];
} config;

void vanity_setup(config &vanity);
void vanity_run(config &vanity);
void __global__ vanity_init(unsigned long long int *seed, curandState *state);
void __global__ vanity_scan(curandState *state, int *keys_found, int *gpu, int *execution_count);
bool __device__ b58enc(char *b58, size_t *b58sz, uint8_t *data, size_t binsz);

int main(int argc, char const *argv[])
{
	ed25519_set_verbose(true);
	config vanity;
	vanity_setup(vanity);
	vanity_run(vanity);
}

void vanity_setup(config &vanity)
{
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	for (int i = 0; i < gpuCount; ++i)
	{
		cudaSetDevice(i);
		int blockSize = 0, minGridSize = 0, maxActiveBlocks = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

		unsigned long long int rseed = std::chrono::system_clock::now().time_since_epoch().count();
		unsigned long long int *dev_rseed;
		cudaMalloc((void **)&dev_rseed, sizeof(unsigned long long int));
		cudaMemcpy(dev_rseed, &rseed, sizeof(unsigned long long int), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&(vanity.states[i]), maxActiveBlocks * blockSize * sizeof(curandState));
		vanity_init<<<maxActiveBlocks, blockSize>>>(dev_rseed, vanity.states[i]);
	}
}

void vanity_run(config &vanity)
{
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	int keys_found_total = 0;
	int keys_found_this_iteration;
	int *dev_keys_found[100];

	for (int g = 0; g < gpuCount; ++g)
	{
		cudaSetDevice(g);
		int blockSize = 0, minGridSize = 0, maxActiveBlocks = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

		cudaMalloc((void **)&dev_keys_found[g], sizeof(int));
		vanity_scan<<<maxActiveBlocks, blockSize>>>(vanity.states[g], dev_keys_found[g], &g, nullptr);
	}

	cudaDeviceSynchronize();

	for (int g = 0; g < gpuCount; ++g)
	{
		cudaMemcpy(&keys_found_this_iteration, dev_keys_found[g], sizeof(int), cudaMemcpyDeviceToHost);
		keys_found_total += keys_found_this_iteration;
	}

	printf("Chiavi trovate: %d\n", keys_found_total);
}

/* -- Funzioni CUDA ------------------------------------------------------- */

void __global__ vanity_init(unsigned long long int *rseed, curandState *state)
{
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	curand_init(*rseed + id, id, 0, &state[id]);
}

void __global__ vanity_scan(curandState *state, int *keys_found, int *gpu, int *exec_count)
{
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	atomicAdd(exec_count, 1);

	curandState localState = state[id];
	unsigned char publick[32] = {0};
	char key[256] = {0};
	size_t keysize = 256;

	b58enc(key, &keysize, publick, 32);

	for (int i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); ++i)
	{
		int match = 1;
		for (int j = 0; suffixes[i][j] != '\0'; ++j)
		{
			if (key[j] != suffixes[i][j])
			{
				match = 0;
				break;
			}
		}
		if (match)
		{
			atomicAdd(keys_found, 1);
			printf("GPU %d: Trovato suffisso %s\n", *gpu, key);
			break;
		}
	}
	state[id] = localState;
}

bool __device__ b58enc(char *b58, size_t *b58sz, uint8_t *data, size_t binsz)
{
	const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
	const uint8_t *bin = data;
	int carry;
	size_t i, j, high, zcount = 0;
	size_t size;

	while (zcount < binsz && !bin[zcount])
		++zcount;

	size = (binsz - zcount) * 138 / 100 + 1;
	uint8_t buf[256] = {0};

	for (i = zcount, high = size - 1; i < binsz; ++i, high = j)
	{
		for (carry = bin[i], j = size - 1; (j > high) || carry; --j)
		{
			carry += 256 * buf[j];
			buf[j] = carry % 58;
			carry /= 58;
			if (!j)
				break;
		}
	}

	for (j = 0; j < size && !buf[j]; ++j)
		;

	if (*b58sz <= zcount + size - j)
	{
		*b58sz = zcount + size - j + 1;
		return false;
	}

	if (zcount)
		memset(b58, '1', zcount);
	for (i = zcount; j < size; ++i, ++j)
		b58[i] = b58digits_ordered[buf[j]];

	b58[i] = '\0';
	*b58sz = i + 1;
	return true;
}
