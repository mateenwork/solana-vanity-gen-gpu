const { Keypair } = require('@solana/web3.js');

// Array JSON che contiene la chiave privata e pubblica da 64 byte
const secretKeyArray = JSON.parse("[120,234,41,158,113,131,135,119,27,168,1,102,223,211,138,25,193,0,60,167,99,91,62,180,179,58,117,187,4,56,52,75,24,148,81,73,192,155,213,58,47,110,160,115,151,161,206,194,192,96,192,26,127,99,22,210,65,22,23,97,178,101,135,223]");

// Converti in un Uint8Array
const secretKey = Uint8Array.from(secretKeyArray);

try {
    // Importa la chiave privata
    const keypair = Keypair.fromSecretKey(secretKey);
    console.log("Public Key:", keypair.publicKey.toBase58());
} catch (error) {
    console.error("Errore durante l'importazione della chiave:", error.message);
}
