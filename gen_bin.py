import numpy as np
import os, sys

# 设置信息全打印
np.set_printoptions(threshold=np.inf)

def save_bin(data, binary_path):
    print(binary_path)
    print('file shape: {}, dtype: {}'.format(data.shape, data.dtype))

    data.tofile(binary_path)
    # read bin file use fromfile
    print('Done\n')

def save_txt(data, txt_path):
    print(txt_path)
    print('file shape: {}, dtype: {}'.format(data.shape, data.dtype))

    # 没有括号的版本
    data = data.squeeze()
    if len(data.shape) == 3:    # savetxt最多存储二维
        data = data.reshape(-1, data.shape[-1])
    np.savetxt(txt_path, data, fmt='%4d')

    # 有括号的版本
    # with open(txt_path, 'w') as f:
    #     print(data, file=f)
    print('Done\n')

def load_save(npy_path, dst_path, transpose=None):
    ''' load and save numpy array to bin
        data will be reshape to 3 dimensions before save to bin

        Args:
            transpose: list [0,3,1,2], and fill the last dimension to the times of 16 with the value 0
    '''
    npy_data = np.load(npy_path)
    if transpose != None:
        npy_data = npy_data.transpose(*transpose)
        npy_data = npy_data.reshape(npy_data.shape[0], npy_data.shape[1], -1)

        # only depthwise layers need to be filled
        fill_num = 16 - npy_data.shape[-1] % 16
        npy_data = np.pad(npy_data, ((0,0), (0,0), (0,fill_num)), 'constant', constant_values=((0,0), (0,0), (0,128)))

    # save_bin(npy_data, dst_path)    # 保存成bin格式
    save_txt(npy_data, dst_path)


if __name__ == '__main__':

    model_dir = 'test_log/mobilenetv3_quant_mfcc_gen'
    weight_dir = os.path.join(model_dir, 'weight')
    bin_dir = os.path.join(model_dir, 'bin')
    txt_dir = os.path.join(model_dir, 'txt')
    if os.path.exists(bin_dir) == False:
        os.mkdir(bin_dir)
    if os.path.exists(txt_dir) == False:
        os.mkdir(txt_dir)

    ################## stem conv ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_stem_conv_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        dst_path=os.path.join(txt_dir, 'stem_conv_weights.txt'),
        transpose=[0,3,1,2]
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_stem_conv_conv_Conv2D_Fold_bias.npy'),
        dst_path=os.path.join(txt_dir, 'stem_conv_bias.txt')
    )

    ################## inverted residual 1 expansion ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_1_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_1_expansion_weights.txt')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_1_expansion_conv_Conv2D_Fold_bias.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_1_expansion_bias.txt')
    )

    ################## inverted residual 1 depthwise ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_1_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_1_depthwise_weights.txt'),
        transpose=[0,3,1,2]
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_1_depthwise_depthwise_conv_Fold_bias.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_1_depthwise_bias.txt')
    )

    ################## inverted residual 1 projection ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_1_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_1_projection_weights.txt')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_1_projection_conv_Conv2D_Fold_bias.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_1_projection_bias.txt')
    )

    ################## inverted residual 2 expansion ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_2_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_2_expansion_weights.txt')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_2_expansion_conv_Conv2D_Fold_bias.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_2_expansion_bias.txt')
    )

    ################## inverted residual 2 depthwise ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_2_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_2_depthwise_weights.txt'),
        transpose=[0,3,1,2]
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_2_depthwise_depthwise_conv_Fold_bias.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_2_depthwise_bias.txt')
    )

    ################## inverted residual 2 projection ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_2_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_2_projection_weights.txt')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_2_projection_conv_Conv2D_Fold_bias.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_2_projection_bias.txt')
    )

    ################## inverted residual 3 expansion ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_3_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_3_expansion_weights.txt')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_3_expansion_conv_Conv2D_Fold_bias.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_3_expansion_bias.txt')
    )

    ################## inverted residual 3 depthwise ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_3_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_3_depthwise_weights.txt'),
        transpose=[0,3,1,2]
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_3_depthwise_depthwise_conv_Fold_bias.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_3_depthwise_bias.txt')
    )

    ################## inverted residual 3 projection ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_3_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_3_projection_weights.txt')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_3_projection_conv_Conv2D_Fold_bias.npy'),
        dst_path=os.path.join(txt_dir, 'inverted_residual_3_projection_bias.txt')
    )

    ################## Conv2D ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_fc_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        dst_path=os.path.join(txt_dir, 'Conv2D_weights.txt')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_fc_conv_Conv2D_bias.npy'),
        dst_path=os.path.join(txt_dir, 'Conv2D_bias.txt')
    )


    ################## Structure info ##################
    struct_info = np.array([0b00001010, 
                            0b00011000, 0b00001000, 0b00010100,
                            0b00011000, 0b00001000, 0b00010010, 
                            0b00011000, 0b00001000, 0b00010101,
                            0b00010000], dtype=np.uint8)
    save_txt(struct_info, os.path.join(txt_dir, 'structure_info.txt'))

    ################## Weight info ##################
    weight_info = np.array([
        [1,10,4,16], 
        [32,1,1,16], [1,3,3,16], [16,1,1,32],
        [32,1,1,16], [1,3,3,32], [32,1,1,32],
        [64,1,1,32], [1,5,5,64], [32,1,1,64],
        [12,1,1,32]
    ], dtype=np.uint8)

    save_txt(weight_info, os.path.join(txt_dir, 'weight_info.txt'))

    ################## Feature info ##################
    feature_info = np.array([
        [1,49,10,1], [1,25,5,16],
        [1,25,5,32], [1,25,5,32], [1,25,5,16], 
        [1,25,5,32], [1,25,5,32], [1,25,5,32], 
        [1,25,5,64], [1,25,5,64], [1,25,5,32], 
        [1,1,1,12]
    ], dtype=np.uint8)

    save_txt(feature_info, os.path.join(txt_dir, 'feature_info.txt'))

    ################## Scale int ##################
    bias_scale = np.array([0.0006772454944439232, 0.0019126507686451077, 0.004039060324430466, 0.0009780717082321644, 0.0011637755669653416, 0.002527922624722123, 0.000784197065513581, 0.00036984056350775063, 0.0027576638385653496, 0.0018317087087780237, 0.003179859137162566])
    result_sacale = np.array([0.15135173499584198, 0.20287899672985077, 0.1442921757698059, 0.11213209480047226, 0.1550600677728653, 0.0902664065361023, 0.07894150912761688, 0.0978255569934845, 0.08960756659507751, 0.1850544661283493, 0.19603444635868073])
    add_scale = np.array([0.11213209480047226, 0.15135173499584198, 0.16829396784305573, 0.1850544661283493, 0.07894150912761688, 0.1915309578180313])

    scale = bias_scale / result_sacale
    scale = np.round(scale * 2**10).astype(np.uint16)
    add_scale = np.round(add_scale * 2**10).astype(np.uint16)
    # change division to multiplication
    add_scale[2] = np.floor(1 / add_scale[2] * 2**15).astype(np.int32)
    add_scale[5] = np.floor(1 / add_scale[5] * 2**15).astype(np.int32)

    scale_int = np.concatenate((scale, add_scale), axis=0)
    save_txt(scale_int, os.path.join(txt_dir, 'scale_int.txt'))

    ################## input data ##################
    load_save(
        npy_path=os.path.join(model_dir, 'input_data/0_no.npy'), 
        dst_path=os.path.join(txt_dir, '0_no.txt')
    )