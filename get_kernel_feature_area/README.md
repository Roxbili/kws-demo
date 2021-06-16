## 文件功能
- layers.py: 算子实现，主要在里面有添加保存代码。（feature对应卷积核的方框行扫描，对于conv，先通道，再行扫描）
- test_sdcard_numpy.py: 使用numpy实现的算子实现神经网络，主要用于保存感受野。保存为npy格式，之后再用脚本处理成txt格式
- npy2txt.py: 将npy格式加上数，并保存成txt