## 文件功能
- layers.py: 算子实现，主要在里面有添加保存代码。（feature对应卷积核的方框行扫描，对于conv，先通道，再行扫描）
- test_sdcard_numpy.py: 使用numpy实现的算子实现神经网络，主要用于保存感受野。保存为npy格式，之后再用脚本处理成txt格式。也可以保存每层的输出结果，主要是conv和depth_conv的save_name参数设置来判断是否保存感受野数据。
- npy2txt.py: 将npy格式加上数，并保存成txt
- layers_output_view.py: 输出output文件夹的npy内容
- log:
  - npy: 主要用于生成感受野的txt，用npy2txt脚本处理，但是有padding的层不行
  - txt: 感受野txt
  - output: 每层的输出结果