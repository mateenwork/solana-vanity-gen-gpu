const solanaWeb3 = require('@solana/web3.js');

// Array completo di 64 valori decimali che rappresenta la chiave privata
const privateKeyArray = [
   33, 148, 18, 96, 164, 180, 194, 18, 135, 220, 182, 133, 34, 223, 12, 88,
   151, 181, 143, 32, 178, 96, 100, 110, 191, 169, 171, 72, 171, 214, 64, 57,
   120, 12, 34, 56, 78, 90, 12, 34, 56, 78, 90, 12, 34, 56, 78, 90,
   12, 34, 56, 78, 90, 12, 34, 56, 78, 90, 12, 34, 56, 78, 90, 99
];

// Converti l'array in un Uint8Array
const secretKey = Uint8Array.from(privateKeyArray);

// Crea un oggetto Keypair dalla chiave privata
const keypair = solanaWeb3.Keypair.fromSecretKey(secretKey);

// Verifica l'indirizzo pubblico associato alla chiave privata
console.log('Indirizzo pubblico:', keypair.publicKey.toBase58());
console.log('Chiave privata (base58):', bs58.encode(keypair.secretKey));
