import numpy as np

path = "./mobilenetv3_quant_eval/weights_npy.npy"

w = dict(np.load(path, allow_pickle=True).tolist())

for key in w.keys():
	print(key)
	print(w[key])

print(w.keys())
