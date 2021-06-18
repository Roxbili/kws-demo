import numpy as np
import os, sys

# 设置信息全打印
np.set_printoptions(threshold=np.inf)

log_dir = 'get_kernel_feature_area/log'

def savetxt(src_path, dst_path, add_num):
    data = np.load(src_path)
    data += add_num
    np.savetxt(dst_path, data, fmt='%4d')

# padding层需要手动保存
# savetxt(os.path.join(log_dir, 'npy/stem_conv.npy'), os.path.join(log_dir, 'txt/stem_conv.txt'), add_num=221.)
savetxt(os.path.join(log_dir, 'npy/inverted_residual_1_expansion.npy'), os.path.join(log_dir, 'txt/inverted_residual_1_expansion.txt'), add_num=128.)
# savetxt(os.path.join(log_dir, 'npy/inverted_residual_1_depthwise.npy'), os.path.join(log_dir, 'txt/inverted_residual_1_depthwise.txt'), add_num=128.)
savetxt(os.path.join(log_dir, 'npy/inverted_residual_1_projection.npy'), os.path.join(log_dir, 'txt/inverted_residual_1_projection.txt'), add_num=128.)
savetxt(os.path.join(log_dir, 'npy/inverted_residual_2_expansion.npy'), os.path.join(log_dir, 'txt/inverted_residual_2_expansion.txt'), add_num=128.)
# savetxt(os.path.join(log_dir, 'npy/inverted_residual_2_depthwise.npy'), os.path.join(log_dir, 'txt/inverted_residual_2_depthwise.txt'), add_num=128.)
savetxt(os.path.join(log_dir, 'npy/inverted_residual_2_projection.npy'), os.path.join(log_dir, 'txt/inverted_residual_2_projection.txt'), add_num=128.)
savetxt(os.path.join(log_dir, 'npy/inverted_residual_3_expansion.npy'), os.path.join(log_dir, 'txt/inverted_residual_3_expansion.txt'), add_num=128.)
# savetxt(os.path.join(log_dir, 'npy/inverted_residual_3_depthwise.npy'), os.path.join(log_dir, 'txt/inverted_residual_3_depthwise.txt'), add_num=128.)
savetxt(os.path.join(log_dir, 'npy/inverted_residual_3_projection.npy'), os.path.join(log_dir, 'txt/inverted_residual_3_projection.txt'), add_num=128.)
savetxt(os.path.join(log_dir, 'npy/Conv2D.npy'), os.path.join(log_dir, 'txt/Conv2D.txt'), add_num=128.)

print('See txt directory')