# ZYNQ

## ZYNQ 7020系统制作

按照以下步骤进行制作。

### SD卡分区
将SD卡分成两个分区，用于存放BOOT和跟文件系统：
| 分区名字  |   文件系统格式 |
|----------|--------------|
| FAT      | FAT          |
| EXT      | EXT4         |

### 拷贝BOOT文件至FAT分区
1. 下载BOOT文件：
   - 链接: https://pan.baidu.com/s/1ob4XSz-HZHvnBoXE8DVB7Q
   - 提取码: 8nmw

2. 将`BOOT.bin`以及`image.ub`两个文件拷贝至SD卡的`FAT`分区下。

### 拷贝根文件系统至EXT分区
为了防止根文件系统权限问题，请在Linux下完成操作

1. 下载跟文件系统：
   - 链接: https://pan.baidu.com/s/1d4W14foPIAIh6in7VWpVkQ
   - 提取码: 7gjh

2. 输入以下命令解压根文件系统
```bash
sudo tar zxpvf linaro-jessie-alip-20161117-32.tar.gz
```

3. 拷贝根文件系统
```bash
cd binary
sudo rsync -av ./ /media/<your user name>/EXT
```

需要注意的是，由于文件数量过多，即使命令运行完成，
也请等待能够正常推出U盘，**不能强行拔出U盘，否则拷贝不完整，数据可能留在缓冲区中**。

## root下python3的问题
1. 在```/usr/bin```下创建软链接，指向conda中的python3的可执行文件

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