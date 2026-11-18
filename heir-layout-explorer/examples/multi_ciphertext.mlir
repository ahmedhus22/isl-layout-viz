#two_ciphertexts = #tensor_ext.layout<"{ [i0] -> [ct, slot] : 0 <= i0 <= 4999 and ct = floor(i0 / 4096) and slot = i0 mod 4096 }">
func.func @example(%arg0: tensor<2x!ct>) { return }
