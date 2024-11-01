#include <vector> // Libreria per gestire vettori dinamici
#include <random> // Libreria per la generazione di numeri casuali
#include <chrono> // Libreria per la misurazione del tempo di esecuzione

#include <assert.h>	  // Libreria per le asserzioni e il controllo degli errori
#include <inttypes.h> // Libreria per definire tipi di interi con dimensioni specifiche
#include <pthread.h>  // Libreria per la gestione dei thread
#include <stdio.h>	  // Libreria per funzioni di input/output

#include "curand_kernel.h" // Libreria CUDA per la generazione di numeri casuali sui dispositivi GPU
#include "ed25519.h"	   // Libreria per la crittografia basata su curve ellittiche Ed25519
#include "fixedint.h"	   // Libreria per gestire interi di dimensioni fisse
#include "gpu_common.h"	   // Libreria di funzioni comuni per il supporto GPU
#include "gpu_ctx.h"	   // Libreria per la gestione del contesto GPU

#include "keypair.cu"  // File di implementazione per la generazione di coppie di chiavi
#include "sc.cu"	   // File di implementazione per operazioni scalari
#include "fe.cu"	   // File di implementazione per operazioni di campo
#include "ge.cu"	   // File di implementazione per operazioni geometriche
#include "sha512.cu"   // File di implementazione per il hashing SHA512
#include "../config.h" // File di configurazione per l'applicazione

/* -- Tipi ------------------------------------------------------------------- */

typedef struct
{
	curandState *states[8]; // Array di stati casuali per CUDA, uno per ogni GPU
} config;

/* -- Prototipi delle Funzioni ------------------------------------------------*/

void vanity_setup(config &vanity);											   // Funzione per inizializzare la configurazione GPU
void vanity_run(config &vanity);											   // Funzione per avviare il processo di generazione vanity
void __global__ vanity_init(curandState *state);							   // Kernel CUDA per inizializzare gli stati casuali
void __global__ vanity_scan(curandState *state);							   // Kernel CUDA per generare chiavi vanity
bool __device__ b58enc(char *b58, size_t *b58sz, uint8_t *data, size_t binsz); // Funzione per l’encoding in Base58

/* -- Punto di Ingresso Principale ------------------------------------------- */

int main(int argc, char const *argv[])
{
	ed25519_set_verbose(true); // Attiva la modalità di output verbose

	config vanity;		  // Struttura per la configurazione del generatore vanity
	vanity_setup(vanity); // Inizializza la configurazione della GPU
	vanity_run(vanity);	  // Avvia il processo di generazione vanity
}

/* -- Funzioni per la Configurazione GPU ------------------------------------- */

void vanity_setup(config &vanity)
{
	printf("GPU: Inizializzazione della memoria\n"); // Messaggio per l’inizio dell’inizializzazione GPU
	int gpuCount = 0;								 // Variabile per memorizzare il numero di GPU
	cudaGetDeviceCount(&gpuCount);					 // Ottiene il numero di dispositivi GPU disponibili

	for (int i = 0; i < gpuCount; ++i) // Itera su ciascuna GPU
	{
		cudaSetDevice(i); // Seleziona la GPU corrente

		cudaDeviceProp device;				 // Struttura per le proprietà del dispositivo
		cudaGetDeviceProperties(&device, i); // Ottiene le proprietà della GPU corrente

		int blockSize = 0, minGridSize = 0, maxActiveBlocks = 0;									// Variabili per calcolare l’occupazione
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);			// Calcola block size ottimale
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0); // Calcola max block attivi

		printf("GPU: (%s <%d, %d, %d>) -- W: %d, P: %d, TPB: %d, MTD: (%dx, %dy, %dz), MGS: (%dx, %dy, %dz)\n",
			   device.name, blockSize, minGridSize, maxActiveBlocks, device.warpSize,
			   device.multiProcessorCount, device.maxThreadsPerBlock, device.maxThreadsDim[0],
			   device.maxThreadsDim[1], device.maxThreadsDim[2], device.maxGridSize[0],
			   device.maxGridSize[1], device.maxGridSize[2]); // Stampa le proprietà della GPU

		cudaMalloc((void **)&(vanity.states[i]), maxActiveBlocks * blockSize * sizeof(curandState)); // Alloca memoria per stati casuali
		vanity_init<<<maxActiveBlocks, blockSize>>>(vanity.states[i]);								 // Avvia il kernel per inizializzare gli stati casuali
	}

	printf("Fine: Inizializzazione della memoria\n"); // Messaggio di fine inizializzazione
}

void vanity_run(config &vanity)
{
	int gpuCount = 0;			   // Variabile per memorizzare il numero di GPU
	cudaGetDeviceCount(&gpuCount); // Ottiene il numero di GPU disponibili

	for (int i = 0; i < 1024; ++i) // Loop per tentativi di generazione
	{
		auto start = std::chrono::high_resolution_clock::now(); // Inizio misurazione tempo

		for (int i = 0; i < gpuCount; ++i) // Esegue il kernel su tutte le GPU
		{
			cudaSetDevice(i);																			// Seleziona la GPU corrente
			int blockSize = 0, minGridSize = 0, maxActiveBlocks = 0;									// Variabili per calcolare l'occupazione
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vanity_scan, 0, 0);			// Calcola la block size ottimale
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, vanity_scan, blockSize, 0); // Calcola max blocchi attivi
			vanity_scan<<<maxActiveBlocks, blockSize>>>(vanity.states[i]);								// Esegue il kernel per generare chiavi vanity
		}

		cudaDeviceSynchronize();								 // Sincronizza tutte le GPU per attendere la fine dei kernel
		auto finish = std::chrono::high_resolution_clock::now(); // Fine misurazione tempo

		std::chrono::duration<double> elapsed = finish - start; // Calcola il tempo trascorso
		printf("Tentativi: %d in %f s a %fcps\n",				// Stampa il resoconto delle prestazioni
			   (8 * 8 * 256 * 100000), elapsed.count(),
			   (8 * 8 * 256 * 100000) / elapsed.count());
	}
}

/* -- Kernel CUDA per Inizializzare Stati Casuali ---------------------------- */

void __global__ vanity_init(curandState *state)
{
	int id = threadIdx.x + (blockIdx.x * blockDim.x); // Calcola l'ID del thread corrente
	curand_init(580000 + id, id, 0, &state[id]);	  // Inizializza lo stato casuale del thread
}

/* -- Kernel CUDA per la Scansione Vanity ------------------------------------ */

void __global__ vanity_scan(curandState *state)
{
	int id = threadIdx.x + (blockIdx.x * blockDim.x); // Calcola l'ID del thread

	ge_p3 A;							// Struttura per l'operazione di curva ellittica
	curandState localState = state[id]; // Copia lo stato casuale per evitare modifiche simultanee
	unsigned char seed[32] = {0};		// Buffer per il seed casuale
	unsigned char publick[32] = {0};	// Buffer per la chiave pubblica generata
	unsigned char privatek[64] = {0};	// Buffer per la chiave privata generata
	char key[256] = {0};				// Buffer per la chiave vanity generata in Base58
	char pkey[256] = {0};				// Buffer per l'output finale della chiave

	for (int i = 0; i < 32; ++i) // Genera un seed casuale
	{
		float random = curand_uniform(&localState); // Numero casuale tra 0 e 1
		uint8_t keybyte = (uint8_t)(random * 255);	// Converte in byte tra 0 e 255
		seed[i] = keybyte;							// Assegna il byte al seed
	}

	size_t keys_found = 0; // Contatore per le chiavi trovate
	sha512_context md;	   // Struttura di contesto per SHA512

	for (int attempts = 0; attempts < 100000; ++attempts)
	{
		// sha512_init Inlined
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

		// Inline sha512_compress
		uint64_t S[8], W[80], t0, t1;
		int i;

		/* Copy state into S */
		for (i = 0; i < 8; i++)
		{
			S[i] = md.state[i];
		}

		/* Copy the state into 1024-bits into W[0..15] */
		for (i = 0; i < 16; i++)
		{
			LOAD64H(W[i], md.buf + (8 * i));
		}

		/* Fill W[16..79] */
		for (i = 16; i < 80; i++)
		{
			W[i] = Gamma1(W[i - 2]) + W[i - 7] + Gamma0(W[i - 15]) + W[i - 16];
		}

/* Compress */
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

		/* Feedback */
		for (i = 0; i < 8; i++)
		{
			md.state[i] = md.state[i] + S[i];
		}

		// We can now output our finalized bytes into the output buffer.
		for (i = 0; i < 8; i++)
		{
			STORE64H(md.state[i], privatek + (8 * i));
		}

		// Code Until here runs at 87_000_000H/s.

		// ed25519 Hash Clamping
		privatek[0] &= 248;
		privatek[31] &= 63;
		privatek[31] |= 64;

		// ed25519 curve multiplication to extract a public key.
		ge_scalarmult_base(&A, privatek);
		ge_p3_tobytes(publick, &A);

		size_t keysize = 256;
		b58enc(key, &keysize, publick, 32);

		for (int i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); ++i)
		{
			size_t found = 0;
			for (int j = 0; prefixes[i][j] != 0; ++j)
			{
				char lowered = (key[j] >= 65 && key[j] <= 90)
								   ? key[j] + 32
								   : key[j];

				if (prefixes[i][found] == '?' || prefixes[i][found] == lowered)
					found++;
				else
					found = 0;

				if (found == 6)
					break;
			}

			if (found == 6)
			{
				keys_found += 1;
				size_t pkeysize = 256;
				b58enc(pkey, &pkeysize, seed, 32);
				printf("(%d): %s - %s\n", keysize, key, pkey);
			}
		}

		for (int i = 0; i < 32; ++i)
		{
			if (seed[i] == 255)
			{
				seed[i] = 0;
			}
			else
			{
				seed[i] += 1;
				break;
			}
		}
	}

	state[id] = localState; // Salva lo stato aggiornato
}

/* -- Funzione per Encoding in Base58 ---------------------------------------- */

bool __device__ b58enc(
	char *b58,
	size_t *b58sz,
	uint8_t *data,
	size_t binsz)
{
	const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"; // Tabella Base58
	const uint8_t *bin = data;																	   // Puntatore ai dati binari

	int carry;
	size_t i, j, high, zcount = 0;
	size_t size;

	while (zcount < binsz && !bin[zcount]) // Conta i byte iniziali zero
		++zcount;

	size = (binsz - zcount) * 138 / 100 + 1; // Calcola la lunghezza necessaria per Base58
	uint8_t buf[256];
	memset(buf, 0, size); // Inizializza il buffer a zero

	for (i = zcount, high = size - 1; i < binsz; ++i, high = j) // Esegui la divisione binaria
	{
		for (carry = bin[i], j = size - 1; (j > high) || carry; --j)
		{
			carry += 256 * buf[j];
			buf[j] = carry % 58;
			carry /= 58;
			if (!j)
				break; // Fermati quando il buffer è vuoto
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
