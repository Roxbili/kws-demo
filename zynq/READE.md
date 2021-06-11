## 文件说明
- kws_dataset_client.py: 运行测试集，将结果返回上位机
- input_data_zynq.py: 加载测试集时的预处理文件
- layers.py: 使用numpy实现的神经网络算子
- test_tflite_numpy.py: 在zynq上测试数据集的准确率
- realtime_kws_client.py: 不在zynq上运行的代码，在一台linux上模拟有麦克分的板子，实时检测麦克风的输入识别后将结果返回上位机
- kws_ps_pl.py: 使用pl侧识别，将结果返回至ps，ps再将结果发送至上位机