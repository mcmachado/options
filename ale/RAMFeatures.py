import numpy as np

NUM_BITS_RAM = 1024

def bitfield(n):
    return [1 if digit=='1' else 0 for digit in bin(n)[2:].zfill(8)]

def getRAMVector(ale):
  ramFeatures = np.zeros(NUM_BITS_RAM)
  ram_size = ale.getRAMSize()
  ram = np.zeros((ram_size,), dtype=np.uint8)
  ale.getRAM(ram)

  ### Essa parte do codigo que esta muito lenta:
  for i, byteInRAM in enumerate(ram):
    byte = bitfield(byteInRAM)
    for j, bit in enumerate(byte):
      ramFeatures[(i * 8) + (7 - j)] = bit

  return ramFeatures