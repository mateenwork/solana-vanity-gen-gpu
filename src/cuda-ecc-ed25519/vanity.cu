#include <vector>
#include <random>
#include <chrono>
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

/* -- Types ----------------------------------------------------------------- */
typedef struct
{
	// CUDA Random States.
	curandState *states[8];
} config;

/* -- Prototypes ------------------------------------------------------------ */
void vanity_setup(config &vanity);
void vanity_run(config &vanity);
void __global__ vanity_init(curandState *state);
void __global__ vanity_scan(curandState *state, int key_length);
bool __host__ __device__ b58enc(char *b58, size_t *b58sz, uint8_t *data, size_t binsz);

/* -- Entry Point ----------------------------------------------------------- */
int main(int argc, char const *argv[])
{
	ed25519_set_verbose(true);

	config vanity;
	vanity_setup(vanity);
	vanity_run(vanity);
}

/* -- Vanity Step Functions ------------------------------------------------- */
void vanity_setup(config &vanity)
{
	printf("GPU: Initializing Memory\n");
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	// Create random states so kernels have access to random generators
	// while running in the GPU.
	for (int i = 0; i < gpuCount; ++i)
	{
		cudaSetDevice(i);

		// Fetch Device Properties
		cudaDeviceProp device;
		cudaGetDeviceProperties(&device, i);

		// Calculate Occupancy
		int blockSize = 0, minGridSize = 0, maxActiveBlocks = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

		printf("GPU: (%s <%d, %d, %d>) -- W: %d, P: %d, TPB: %d, MTD: (%dx, %dy, %dz), MGS: (%dx, %dy, %dz)\n",
			   device.name,
			   blockSize,
			   minGridSize,
			   maxActiveBlocks,
			   device.warpSize,
			   device.multiProcessorCount,
			   device.maxThreadsPerBlock,
			   device.maxThreadsDim[0],
			   device.maxThreadsDim[1],
			   device.maxThreadsDim[2],
			   device.maxGridSize[0],
			   device.maxGridSize[1],
			   device.maxGridSize[2]);

		cudaMalloc((void **)&(vanity.states[i]), maxActiveBlocks * blockSize * sizeof(curandState));
		vanity_init<<<maxActiveBlocks, blockSize>>>(vanity.states[i]);
	}

	printf("END: Initializing Memory\n");
}

void vanity_run(config &vanity)
{
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	for (int i = 0; i < 1024; ++i)
	{
		auto start = std::chrono::high_resolution_clock::now();

		// Run on all GPUs
		for (int i = 0; i < gpuCount; ++i)
		{
			cudaSetDevice(i);

			int blockSize = 0, minGridSize = 0, maxActiveBlocks = 0;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0);

			// Genera la chiave `key` e calcola `key_length`
			char key[256] = {0};
			unsigned char publick[32] = {0};
			size_t keysize = 256;

			b58enc(key, &keysize, publick, 32); // Genera l’indirizzo
			int key_length = strlen(key);		// Calcola `key_length` sul lato host

			// Lancia il kernel con `key_length`
			vanity_scan<<<maxActiveBlocks, blockSize>>>(vanity.states[i], key_length);
		}

		cudaDeviceSynchronize();
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		printf("Attempts: %d in %f seconds\n", (8 * 8 * 256 * 100000), elapsed.count());
	}
}

/* -- CUDA Vanity Functions ------------------------------------------------- */
void __global__ vanity_init(curandState *state)
{
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	curand_init(580000 + id, id, 0, &state[id]);
}

void __global__ vanity_scan(curandState *state, int key_length)
{
	int id = threadIdx.x + (blockIdx.x * blockDim.x);

	ge_p3 A;
	curandState localState = state[id];
	unsigned char seed[32] = {0};
	unsigned char publick[32] = {0};
	unsigned char privatek[64] = {0};
	char key[256] = {0};

	for (int i = 0; i < 32; ++i)
	{
		float random = curand_uniform(&localState);
		uint8_t keybyte = (uint8_t)(random * 255);
		seed[i] = keybyte;
	}

	sha512_context md;

	for (int attempts = 0; attempts < 100000; ++attempts)
	{
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

		const unsigned char *in = seed;
		for (size_t i = 0; i < 32; i++)
		{
			md.buf[i + md.curlen] = in[i];
		}
		md.curlen += 32;

		md.length += md.curlen * UINT64_C(8);
		md.buf[md.curlen++] = (unsigned char)0x80;

		while (md.curlen < 120)
		{
			md.buf[md.curlen++] = (unsigned char)0;
		}

		STORE64H(md.length, md.buf + 120);

		uint64_t S[8], W[80], t0, t1;
		int i;

		for (i = 0; i < 8; i++)
		{
			S[i] = md.state[i];
		}

		for (i = 0; i < 16; i++)
		{
			LOAD64H(W[i], md.buf + (8 * i));
		}

		for (i = 16; i < 80; i++)
		{
			W[i] = Gamma1(W[i - 2]) + W[i - 7] + Gamma0(W[i - 15]) + W[i - 16];
		}

#define RND(a, b, c, d, e, f, g, h, i)              \
	t0 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i]; \
	t1 = Sigma0(a) + Maj(a, b, c);                  \
	d += t0;                                        \
	h = t0 + t1;

		for (i = 0; i < 80; i += 8)
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

#undef RND

		for (i = 0; i < 8; i++)
		{
			md.state[i] = md.state[i] + S[i];
		}

		for (i = 0; i < 8; i++)
		{
			STORE64H(md.state[i], privatek + (8 * i));
		}

		privatek[0] &= 248;
		privatek[31] &= 63;
		privatek[31] |= 64;

		ge_scalarmult_base(&A, privatek);
		ge_p3_tobytes(publick, &A);

		size_t keysize = 256;
		b58enc(key, &keysize, publick, 32);

		// Controllo del suffisso "pump"
		bool has_suffix = (key[key_length - 4] == 'p' &&
						   key[key_length - 3] == 'u' &&
						   key[key_length - 2] == 'm' &&
						   key[key_length - 1] == 'p');

		if (has_suffix)
		{
			printf("Key Found: (%lu): %s\n", keysize, key); // Stampa la chiave trovata									// Esci dal ciclo se trovi una corrispondenza
		}
	}

	state[id] = localState;
}

bool __host__ __device__ b58enc(char *b58, size_t *b58sz, uint8_t *data, size_t binsz)
{
	const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

	const uint8_t *bin = data;
	int carry;
	size_t i, j, high, zcount = 0;
	size_t size;

	while (zcount < binsz && !bin[zcount])
		++zcount;

	size = (binsz - zcount) * 138 / 100 + 1;
	uint8_t buf[256];
	memset(buf, 0, size);

	for (i = zcount, high = size - 1; i < binsz; ++i, high = j)
	{
		for (carry = bin[i], j = size - 1; (j > high) || carry; --j)
		{
			carry += 256 * buf[j];
			buf[j] = carry % 58;
			carry /= 58;
			if (!j)
			{
				break;
			}
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
