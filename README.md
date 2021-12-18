# KWS Demo

## 环境
tensorflow==1.13  
python_speech_features==0.6

## 模型训练及量化

### 下载训练集
在根目录下新建`speech_dataset`目录:
```bash
mkdir speech_dataset
```

下载[speech_commands数据集](https://www.tensorflow.org/datasets/catalog/speech_commands)并放在`speech_dataset`目录下:   
链接: https://pan.baidu.com/s/1UdQQxlPPwjOHFjGnM1B_jg 提取码: ehb2

### 训练和量化

1. 量化感知训练
```bash
bash script/train_quant.sh
```

2. 生成量化模型
```bash
bash script/gen_checkpoint.sh
```

### 现有的训练好的模型
下载链接：链接: https://pan.baidu.com/s/1uvnOpvy0bTUiBc0VcEovug 提取码: p33p

均存放于根目录下，最终版本为`test_log/mbnetv3_quant_8bit_mfcc`以及`test_log/mobilenetv3_quant_mfcc_gen`

-------------------------

## 关于ZYNQ推理及演示的工程
https://github.com/Roxbili/kwd-demo-zynq