// Array esadecimale della chiave privata
const hexKeyArray = [
  '24', 'c2', '39', 'a5', '69', 'c7', 'b0', '29', '80', 'e0', 'e7', 'c1', '3f', '5e', 'b4', '2d',
  '22', 'b5', 'ba', 'fa', 'f9', '28', '55', 'f4', '52', '8c', 'ed', '4c', '6b', '8d', '76', 'd7'
];

// Conversione dell'array esadecimale in un array di byte decimale
const byteKeyArray = hexKeyArray.map(byte => parseInt(byte, 16));

console.log("Chiave in formato byte:", byteKeyArray);
