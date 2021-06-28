## 文件说明
- kws_dataset_client.py: 运行测试集，将结果返回上位机
- input_data_zynq.py: 加载测试集时的预处理文件
- layers.py: 使用numpy实现的神经网络算子
- test_tflite_numpy.py: 在zynq上测试数据集的准确率
- realtime_kws_client.py: 不在zynq上运行的代码，在一台linux上模拟有麦克分的板子，实时检测麦克风的输入识别后将结果返回上位机
- kws_ps_pl.py: 使用pl侧识别，将结果返回至ps，ps再将结果发送至上位机
- pl_simulate.py: 模拟pl侧的操作，实现请求数据，读取，推理，返回结果的流程
- test_sdcard_numpy.py: 使用mfcc预处理好的数据进行前向推理。**使用该脚本发现精度差异均来自mfcc预处理**
- tkinter_kws.py: 使用tkinter通过vnc显示结果
- tmp.py: tkinter测试代码，实现了通过Queue让mainloop和工作进程的数据共享，因此该代码暂时留下

## root下python3.5的问题
1. 在```/usr/bin```下创建软链接，指向conda中的python3.5的可执行文件

## 显示问题
1. 安装vnc
    ```bash
    sudo apt install tightvnserver
    sudo apt install xfce4 xfce4-goodies
    ```

2. 安装字体
    ```bash
    sudo apt-get install xfonts-base
    ```

3. 使用vncserver命令创建vnc，然后kill，然后修改配置文件
    ```bash
    vncserver
    vncserver -kill :1
    vim ~/.vnc/xstartup
    ```
    在最后添加
    ```bash
    startxfce4 &
    ```

4. 给root x-window访问权限(不确定是不是这个修好的，但应该是xhost命令，使用的时候基本都报错，但是莫名其妙好了)
    ```bash
    xhost LOCAL:root
    ```
    备选方案：
    ```bash
    xhost LOCAL:fanding
    xhost + root
    xhost +
    ```