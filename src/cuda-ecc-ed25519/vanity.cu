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

typedef struct
{
	curandState *states[8];
} config;

#define MAX_KEYS 1000
__device__ char found_keys[MAX_KEYS][128];
__device__ int found_key_count = 0;

void vanity_setup(config &vanity);
void vanity_run(config &vanity);
__device__ const char suffix[] = "pump";

void __global__ vanity_init(unsigned long long int *seed, curandState *state);
void __global__ vanity_scan(curandState *state, int *keys_found, int *gpu, int *execution_count);
bool __device__ b58enc(char *b58, size_t *b58sz, uint8_t *data, size_t binsz);

// Definizione della macro RND per la compressione SHA512
#define RND(a, b, c, d, e, f, g, h, i)              \
	t0 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i]; \
	t1 = Sigma0(a) + Maj(a, b, c);                  \
	d += t0;                                        \
	h = t0 + t1;

__device__ void convert_to_hex(const unsigned char *input, char *output, int length)
{
	const char hex_chars[] = "0123456789abcdef";
	for (int i = 0; i < length; i++)
	{
		output[i * 2] = hex_chars[(input[i] >> 4) & 0x0F];
		output[i * 2 + 1] = hex_chars[input[i] & 0x0F];
	}
}

// Funzione device_strlen per calcolare la lunghezza di una stringa in ambiente GPU
__device__ int device_strlen(const char *str)
{
	int len = 0;
	while (str[len] != '\0')
	{
		len++;
	}
	return len;
}

// Funzione device_strcmp per confrontare due stringhe in ambiente GPU
__device__ int device_strcmp(const char *str1, const char *str2)
{
	while (*str1 && (*str1 == *str2))
	{
		str1++;
		str2++;
	}
	return *(const unsigned char *)str1 - *(const unsigned char *)str2;
}

int main(int argc, char const *argv[])
{
	ed25519_set_verbose(true);
	config vanity;
	vanity_setup(vanity);
	vanity_run(vanity);
}

std::string getTimeStr()
{
	std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	std::string s(30, '\0');
	std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
	return s;
}

unsigned long long int makeSeed()
{
	unsigned long long int seed = 0;
	char *pseed = (char *)&seed;
	std::random_device rd;
	for (unsigned int b = 0; b < sizeof(seed); b++)
	{
		auto r = rd();
		char *entropy = (char *)&r;
		pseed[b] = entropy[0];
	}
	return seed;
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
		unsigned long long int rseed = makeSeed();
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
	unsigned long long int executions_total = 0;
	unsigned long long int executions_this_iteration;
	int executions_this_gpu;
	int *dev_executions_this_gpu[100];
	int keys_found_total = 0;
	int keys_found_this_iteration;
	int *dev_keys_found[100];
	for (int i = 0; i < MAX_ITERATIONS; ++i)
	{
		auto start = std::chrono::high_resolution_clock::now();
		executions_this_iteration = 0;
		for (int g = 0; g < gpuCount; ++g)
		{
			cudaSetDevice(g);
			int blockSize = 0, minGridSize = 0, maxActiveBlocks = 0;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);
			int *dev_g;
			cudaMalloc((void **)&dev_g, sizeof(int));
			cudaMemcpy(dev_g, &g, sizeof(int), cudaMemcpyHostToDevice);
			cudaMalloc((void **)&dev_keys_found[g], sizeof(int));
			cudaMalloc((void **)&dev_executions_this_gpu[g], sizeof(int));
			vanity_scan<<<maxActiveBlocks, blockSize>>>(vanity.states[g], dev_keys_found[g], dev_g, dev_executions_this_gpu[g]);
		}
		cudaDeviceSynchronize();
		int host_key_count;
		cudaMemcpyFromSymbol(&host_key_count, found_key_count, sizeof(int), 0, cudaMemcpyDeviceToHost);
		char host_keys[MAX_KEYS][512];
		cudaMemcpyFromSymbol(host_keys, found_keys, sizeof(char) * MAX_KEYS * 512, 0, cudaMemcpyDeviceToHost);
		FILE *outputFile = fopen("found_keys.txt", "w");
		if (outputFile)
		{
			for (int i = 0; i < host_key_count; ++i)
			{
				fprintf(outputFile, "Key %d: Privata: %.*s Pubblica: %s\n", i + 1, 64, host_keys[i], host_keys[i] + 65);
			}
			fclose(outputFile);
		}
		else
		{
			printf("Errore: impossibile aprire il file per scrivere\n");
		}
		auto finish = std::chrono::high_resolution_clock::now();
		for (int g = 0; g < gpuCount; ++g)
		{
			cudaMemcpy(&keys_found_this_iteration, dev_keys_found[g], sizeof(int), cudaMemcpyDeviceToHost);
			keys_found_total += keys_found_this_iteration;
			cudaMemcpy(&executions_this_gpu, dev_executions_this_gpu[g], sizeof(int), cudaMemcpyDeviceToHost);
			executions_this_iteration += executions_this_gpu * ATTEMPTS_PER_EXECUTION;
			executions_total += executions_this_gpu * ATTEMPTS_PER_EXECUTION;
		}
		std::chrono::duration<double> elapsed = finish - start;
		printf("%s Iteration %d Attempts: %llu in %f at %fcps - Total Attempts %llu - keys found %d\n",
			   getTimeStr().c_str(), i + 1, executions_this_iteration, elapsed.count(),
			   executions_this_iteration / elapsed.count(), executions_total, keys_found_total);
		if (keys_found_total >= STOP_AFTER_KEYS_FOUND)
		{
			printf("Enough keys found, Done! \n");
			exit(0);
		}
	}
}

void __global__ vanity_init(unsigned long long int *rseed, curandState *state)
{
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	curand_init(*rseed + id, id, 0, &state[id]);
}

void __global__ vanity_scan(curandState *state, int *keys_found, int *gpu, int *exec_count)
{
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	atomicAdd(exec_count, 1);

	ge_p3 A;
	curandState localState = state[id];
	unsigned char seed[32] = {0};
	unsigned char publick[32] = {0};
	unsigned char privatek[64] = {0};
	char key[256] = {0};

	for (int i = 0; i < 32; ++i)
	{
		seed[i] = (uint8_t)(curand_uniform(&localState) * 255);
	}

	sha512_context md;
	md.curlen = 0;
	md.length = 0;
	md.state[0] = UINT64_C(0x6a09e667f3bcc908);
	md.state[1] = UINT64_C(0xbb67ae8584caa73b);
	md.state[2] = UINT64_C(0x3c6ef372fe94f82b);
	md.state[3] = UINT64_C(0xa54ff53a5f1d36f1);
	md.state[4] = UINT64_C(0x510e527fade682d1);
	md.state[5] = UINT64_C(0x9b05688c2b3e6c1f);
	md.state[6] = UINT64_C(0x1f83d9abfb41bd6b);
	md.state[7] = UINT64_C(0x5be0cd19137e2179);

	for (int i = 0; i < 32; i++)
	{
		md.buf[i] = seed[i];
	}
	md.curlen += 32;

	md.length += md.curlen * 8;
	md.buf[md.curlen++] = 0x80;
	while (md.curlen < 120)
	{
		md.buf[md.curlen++] = 0;
	}
	STORE64H(md.length, md.buf + 120);

	uint64_t S[8], W[80], t0, t1;
	for (int i = 0; i < 8; i++)
		S[i] = md.state[i];
	for (int i = 0; i < 16; i++)
		LOAD64H(W[i], md.buf + (8 * i));
	for (int i = 16; i < 80; i++)
	{
		W[i] = Gamma1(W[i - 2]) + W[i - 7] + Gamma0(W[i - 15]) + W[i - 16];
	}

	for (int i = 0; i < 80; i += 8)
	{
		RND(S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], i + 0);
		RND(S[7], S[0], S[1], S[2], S[3], S[4], S[5], S[6], i + 1);
		RND(S[6], S[7], S[0], S[1], S[2], S[3], S[4], S[5], i + 2);
		RND(S[5], S[6], S[7], S[0], S[1], S[2], S[3], S[4], i + 3);
		RND(S[4], S[5], S[6], S[7], S[0], S[1], S[2], S[3], i + 4);
		RND(S[3], S[4], S[5], S[6], S[7], S[0], S[1], S[2], i + 5);
		RND(S[2], S[3], S[4], S[5], S[6], S[7], S[0], S[1], i + 6);
		RND(S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[0], i + 7);
	}
	for (int i = 0; i < 8; i++)
		md.state[i] = md.state[i] + S[i];
	for (int i = 0; i < 8; i++)
		STORE64H(md.state[i], privatek + (8 * i));

	privatek[0] &= 248;
	privatek[31] &= 63;
	privatek[31] |= 64;

	ge_scalarmult_base(&A, privatek);
	ge_p3_tobytes(publick, &A);

	size_t keysize = 256;
	b58enc(key, &keysize, publick, 32);

	int len_key = device_strlen(key);
	if (device_strlen(suffix) <= len_key &&
		device_strcmp(&key[len_key - 4], suffix) == 0)
	{
		int index = atomicAdd(&found_key_count, 1); // Ottiene l'indice per memorizzare la chiave
		if (index < MAX_KEYS)
		{
			// Converte la chiave privata in esadecimale e salva la chiave pubblica
			convert_to_hex(privatek, found_keys[index], 32); // Chiave privata in esadecimale
			int pub_key_offset = 64;						 // Offset dopo i 64 caratteri della chiave privata
			for (int j = 0; j < keysize && j < 64; j++)
			{
				found_keys[index][pub_key_offset + j] = key[j];
			}
			found_keys[index][pub_key_offset + keysize] = '\0'; // Terminazione della stringa
		}
	}
	state[id] = localState;
}