import numpy as np
import os


def save_bin(data, binary_path):
    print(binary_path)
    print('file shape: {}, dtype: {}'.format(data.shape, data.dtype))

    data.tofile(binary_path)
    # read bin file use fromfile
    print('Done\n')

def load_save(npy_path, bin_path, transpose=None):
    '''
        transpose: list [0,3,1,2]
    '''
    npy_data = np.load(npy_path)
    if transpose != None:
        npy_data = npy_data.transpose(*transpose)

    save_bin(npy_data, bin_path)


if __name__ == '__main__':

    model_dir = 'test_log/mobilenetv3_quant_mfcc_gen'
    weight_dir = os.path.join(model_dir, 'weight')
    bin_dir = os.path.join(model_dir, 'bin')
    if os.path.exists(bin_dir) == False:
        os.mkdir(bin_dir)

    ################## stem conv ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_stem_conv_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        bin_path=os.path.join(bin_dir, 'stem_conv_weights.bin')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_stem_conv_conv_Conv2D_Fold_bias.npy'),
        bin_path=os.path.join(bin_dir, 'stem_conv_bias.bin')
    )

    ################## inverted residual 1 expansion ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_1_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_1_expansion_weights.bin')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_1_expansion_conv_Conv2D_Fold_bias.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_1_expansion_bias.bin')
    )

    ################## inverted residual 1 depthwise ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_1_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_1_depthwise_weights.bin')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_1_depthwise_depthwise_conv_Fold_bias.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_1_depthwise_bias.bin')
    )

    ################## inverted residual 1 projection ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_1_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_1_projection_weights.bin')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_1_projection_conv_Conv2D_Fold_bias.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_1_projection_bias.bin')
    )

    ################## inverted residual 2 expansion ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_2_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_2_expansion_weights.bin')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_2_expansion_conv_Conv2D_Fold_bias.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_2_expansion_bias.bin')
    )

    ################## inverted residual 2 depthwise ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_2_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_2_depthwise_weights.bin')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_2_depthwise_depthwise_conv_Fold_bias.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_2_depthwise_bias.bin')
    )

    ################## inverted residual 2 projection ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_2_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_2_projection_weights.bin')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_2_projection_conv_Conv2D_Fold_bias.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_2_projection_bias.bin')
    )

    ################## inverted residual 3 expansion ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_3_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_3_expansion_weights.bin')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_3_expansion_conv_Conv2D_Fold_bias.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_3_expansion_bias.bin')
    )

    ################## inverted residual 3 depthwise ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_3_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_3_depthwise_weights.bin')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_3_depthwise_depthwise_conv_Fold_bias.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_3_depthwise_bias.bin')
    )

    ################## inverted residual 3 projection ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_3_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_3_projection_weights.bin')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_inverted_residual_3_projection_conv_Conv2D_Fold_bias.npy'),
        bin_path=os.path.join(bin_dir, 'inverted_residual_3_projection_bias.bin')
    )

    ################## Conv2D ##################
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_fc_conv_weights_quant_FakeQuantWithMinMaxVars.npy'),
        bin_path=os.path.join(bin_dir, 'Conv2D_weights.bin')
    )
    load_save(
        npy_path=os.path.join(weight_dir, 'MBNetV3-CNN_fc_conv_Conv2D_bias.npy'),
        bin_path=os.path.join(bin_dir, 'Conv2D_bias.bin')
    )


    ################## Structure info ##################
    struct_info = np.array([0b00001010, 
                            0b00011000, 0b00001000, 0b00010100,
                            0b00011000, 0b00001000, 0b00010010, 
                            0b00011000, 0b00001000, 0b00010101,
                            0b00010000], dtype=np.uint8)
    save_bin(struct_info, os.path.join(bin_dir, 'structure_info.bin'))

    ################## Weight info ##################
    weight_info = np.array([
        [1,10,4,16], 
        [32,1,1,16], [1,3,3,16], [16,1,1,32],
        [32,1,1,16], [1,3,3,32], [32,1,1,32],
        [64,1,1,32], [1,5,5,64], [32,1,1,64],
        [12,1,1,32]
    ], dtype=np.uint8)

    save_bin(weight_info, os.path.join(bin_dir, 'weight_info.bin'))

    ################## Feature info ##################
    feature_info = np.array([
        [1,49,10,1], [1,25,5,16],
        [1,25,5,32], [1,25,5,32], [1,25,5,16], 
        [1,25,5,32], [1,25,5,32], [1,25,5,32], 
        [1,25,5,64], [1,25,5,64], [1,25,5,32], 
        [1,1,1,12]
    ], dtype=np.uint8)

    save_bin(feature_info, os.path.join(bin_dir, 'feature_info.bin'))

    ################## Scale int ##################
    bias_scale = np.array([0.0006772454944439232, 0.0019126507686451077, 0.004039060324430466, 0.0009780717082321644, 0.0011637755669653416, 0.002527922624722123, 0.000784197065513581, 0.00036984056350775063, 0.0027576638385653496, 0.0018317087087780237, 0.003179859137162566])
    result_sacale = np.array([0.15135173499584198, 0.20287899672985077, 0.1442921757698059, 0.11213209480047226, 0.1550600677728653, 0.0902664065361023, 0.07894150912761688, 0.0978255569934845, 0.08960756659507751, 0.1850544661283493, 0.19603444635868073])
    add_scale = np.array([0.11213209480047226, 0.15135173499584198, 0.16829396784305573, 0.1850544661283493, 0.07894150912761688, 0.1915309578180313])

    scale = bias_scale / result_sacale
    scale = np.round(scale * 2**10).astype(np.uint16)
    add_scale = np.round(add_scale * 2**10).astype(np.uint16)

    scale_int = np.concatenate((scale, add_scale), axis=0)
    save_bin(scale, os.path.join(bin_dir, 'scale_int.bin'))
