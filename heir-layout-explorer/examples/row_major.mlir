#row_major = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = 64i0 + i1 and 0 <= i0 <= 63 and 0 <= i1 <= 63 and 0 <= slot <= 4095 }">
func.func @example(%arg0: tensor<1x!ct>) { return }
