# KWS Demo

## 文件结构说明

### 根目录
- gen_bin.py: 生成初始化信息，包括net info、weight、bias、scale_int，可生成bin格式或者txt格式

### zynq
- kws_dataset_client.py: 运行测试集，将结果返回上位机
- input_data_zynq.py: 加载测试集时的预处理文件
- layers.py: 使用numpy实现的神经网络算子
- realtime_kws_client.py: 不在zynq上运行的代码，在一台linux上模拟有麦克分的板子，实时检测麦克风的输入识别后将结果返回上位机
- kws_ps_pl.py: 使用pl侧识别，将结果返回至ps，ps再将结果发送至上位机
- pl_simulate.py: 模拟pl侧的操作，实现请求数据，读取，推理，返回结果的流程
- tkinter_kws.py: 使用tkinter通过vnc显示结果
- tmp.py: tkinter测试代码，实现了通过Queue让mainloop和工作进程的数据共享，因此该代码暂时留下
- test_tflite_numpy.py: 在zynq上测试数据集的准确率(未提前处理好MFCC)
- test_sdcard_numpy.py: 使用mfcc预处理好的数据进行前向推理。**使用该脚本发现精度差异均来自mfcc预处理**

### server
- 演示demo2、3中上位机运行脚本

### get_kernel_feature_area
该目录下的代码用于存储一次推理中各种中间结果，具体可看[README.md](../get_kernel_feature_area/README.md)

### test_log
该目录下存储了模型权重、MFCC处理后的数据(input_data)
- mobilenetv3_quant_mfcc_gen/layers_output_view.py: 查看保存的各层输出数据

----------------------------

## 环境要求
- python3
- Numpy
- python_speech_features==0.6 (test_tflite_numpy.py使用)
- vnc (在ZYNQ上运行演示demo1时需要)
- flask (演示demo2)
- flask_socketio (演示demo2)

## 运行说明

### PC / ZYNQ 均可运行
使用MFCC处理后的数据进行神经网络推理(若不需要考虑MFCC预处理过程，推荐使用此脚本)
```python
python3 zynq/test_sdcard_numpy.py
```

使用正常数据集，进行MFCC处理后再进行神经网络推理
```python
bash zynq/test_tflite_numpy.sh
```

想要查看或保存单次推理的中间数据，请参考[README.md](get_kernel_feature_area/README.md)

### ZYNQ 演示

#### 演示demo1（仅ZYNQ运行）

该部分需要ZNYQ上拥有Linux操作系统、[VNC环境](zynq/README.md)、Python3及对应的环境。

ZYNQ板子上的Python3及环境可使用[Berryconda3安装](https://github.com/jjhelmus/berryconda)。

1. ZYNQ板子开启一个vncserver
2. PC使用vnc软件(例如VNCViewer)连接ZYNQ板子对应的vncserver
3. 打开一个终端，运行PL侧模拟脚本（若拥有真实的PL侧bitstream并且已经烧写至ZYNQ板子上，此步骤可忽略）
```python
python3 zynq/pl_simulate.py
```
4. 打开另一个终端，运行PS端处理脚本
```python
python3 zynq/tkinter_kws.py
```

演示效果如图所示：

<img src="./images/demo1.gif" width="500px"></img>

#### 演示demo2（ZYNQ运行客户端，PC运行服务端）

(本步骤中需要去修改server和client脚本内的IP地址)

PC安装依赖：
```bash
cd server
conda env create -f flask.yaml
```

激活conda环境：
```bash
conda activate flask
```

PC运行脚本：
```bash
cd server
python realtime_kws_server.py
```

zynq运行脚本：
```bash
bash zynq/kws_dataset_client.sh
```

演示效果如图所示：

<img src="./images/demo2.gif" width="500px"></img>

### 演示demo3(需要麦克风，仅在两个PC之间)

激活conda环境：
```bash
conda activate flask
```

PC运行脚本：
```bash
cd server
python realtime_kws_server.py
```

带有麦克风的PC运行脚本：
```bash
python zynq/realtime_kws_client.py
```