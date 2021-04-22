import numpy as np
import os


def save_bin(data, binary_path):
    print(binary_path)
    print('file shape: {}, dtype: {}'.format(data.shape, data.dtype))

    data.tofile(binary_path)
    # read bin file use fromfile
    print('Done\n')


if __name__ == '__main__':

    ################## stem conv ##################
    weights = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_stem_conv_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    save_bin(weights, 'test_log/mobilenetv3_quant_gen/bin/stem_conv_weights.bin')

    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_stem_conv_conv_Conv2D_Fold_bias.npy')
    save_bin(bias, 'test_log/mobilenetv3_quant_gen/bin/stem_conv_bias.bin')

    ################## inverted residual 1 expansion ##################
    weights = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_1_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    save_bin(weights, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_1_expansion_weights.bin')

    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_1_expansion_conv_Conv2D_Fold_bias.npy')
    save_bin(bias, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_1_expansion_bias.bin')

    ################## inverted residual 1 depthwise ##################
    weights = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_1_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy')
    save_bin(weights, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_1_depthwise_weights.bin')

    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_1_depthwise_depthwise_conv_Fold_bias.npy')
    save_bin(bias, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_1_depthwise_bias.bin')

    ################## inverted residual 1 projection ##################
    weights = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_1_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    save_bin(weights, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_1_projection_weights.bin')

    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_1_projection_conv_Conv2D_Fold_bias.npy')
    save_bin(bias, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_1_projection_bias.bin')

    ################## inverted residual 2 expansion ##################
    weights = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_2_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    save_bin(weights, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_2_expansion_weights.bin')

    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_2_expansion_conv_Conv2D_Fold_bias.npy')
    save_bin(bias, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_2_expansion_bias.bin')

    ################## inverted residual 2 depthwise ##################
    weights = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_2_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy')
    save_bin(weights, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_2_depthwise_weights.bin')

    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_2_depthwise_depthwise_conv_Fold_bias.npy')
    save_bin(bias, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_2_depthwise_bias.bin')

    ################## inverted residual 2 projection ##################
    weights = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_2_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    save_bin(weights, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_2_projection_weights.bin')

    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_2_projection_conv_Conv2D_Fold_bias.npy')
    save_bin(bias, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_2_projection_bias.bin')

    ################## inverted residual 3 expansion ##################
    weights = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_3_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    save_bin(weights, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_3_expansion_weights.bin')

    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_3_expansion_conv_Conv2D_Fold_bias.npy')
    save_bin(bias, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_3_expansion_bias.bin')

    ################## inverted residual 3 depthwise ##################
    weights = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_3_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy')
    save_bin(weights, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_3_depthwise_weights.bin')

    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_3_depthwise_depthwise_conv_Fold_bias.npy')
    save_bin(bias, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_3_depthwise_bias.bin')

    ################## inverted residual 3 projection ##################
    weights = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_3_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    save_bin(weights, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_3_projection_weights.bin')

    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_3_projection_conv_Conv2D_Fold_bias.npy')
    save_bin(bias, 'test_log/mobilenetv3_quant_gen/bin/inverted_residual_3_projection_bias.bin')

    ################## Conv2D ##################
    weights = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_fc_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    save_bin(weights, 'test_log/mobilenetv3_quant_gen/bin/Conv2D_weights.bin')

    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_fc_conv_Conv2D_bias.npy')
    save_bin(bias, 'test_log/mobilenetv3_quant_gen/bin/Conv2D_bias.bin')



    ################## Structure info ##################
    struct_info = np.array([0b00001010, 
                            0b00011000, 0b00001000, 0b00010100,
                            0b00011000, 0b00001000, 0b00010010, 
                            0b00011000, 0b00001000, 0b00010101,
                            0b00010000], dtype=np.uint8)
    save_bin(struct_info, 'test_log/mobilenetv3_quant_gen/bin/sturcture_info.bin')

    ################## Weight info ##################
    weight_info = np.array([
        [1,10,4,16], 
        [32,1,1,16], [1,3,3,16], [16,1,1,32],
        [32,1,1,16], [1,3,3,32], [32,1,1,32],
        [64,1,1,32], [1,5,5,64], [32,1,1,64],
        [12,1,1,32]
    ], dtype=np.uint8)

    save_bin(weight_info, 'test_log/mobilenetv3_quant_gen/bin/weight_info.bin')

    ################## Feature info ##################
    feature_info = np.array([
        [1,49,10,1], [1,25,5,16],
        [1,25,5,32], [1,25,5,32], [1,25,5,16], 
        [1,25,5,32], [1,25,5,32], [1,25,5,32], 
        [1,25,5,64], [1,25,5,64], [1,25,5,32], 
        [1,1,1,12]
    ], dtype=np.uint8)

    save_bin(feature_info, 'test_log/mobilenetv3_quant_gen/bin/feature_info.bin')

    ################## Scale int ##################
    bias_scale = np.array([0.0008852639002725482, 0.0035931775346398354, 0.00785899069160223, 0.0014689048985019326, 0.0015524440677836537, 0.0028435662388801575, 0.001141879241913557, 0.0007087105768732727, 0.009289528243243694, 0.0015117411967366934, 0.004092711955308914])
    result_sacale = np.array([0.20100615918636322, 0.42823609709739685, 0.23841151595115662, 0.1732778549194336, 0.21222199499607086, 0.15781369805335999, 0.12740808725357056, 0.1111915186047554, 0.11338130384683609, 0.19232141971588135, 0.17540767788887024])
    add_scale = np.array([0.1732778549194336, 0.20100615918636322, 0.26455792784690857, 0.19232141971588135, 0.12740808725357056, 0.20970593392848969])

    scale = bias_scale / result_sacale
    scale = np.round(scale * 2**10).astype(np.uint16)
    add_scale = np.round(add_scale * 2**10).astype(np.uint16)

    scale_int = np.concatenate((scale, add_scale), axis=0)
    save_bin(scale, 'test_log/mobilenetv3_quant_gen/bin/scale_int.bin')
