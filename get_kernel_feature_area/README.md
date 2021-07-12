## 保存中心思路
保存分成以下两个大类：
- output: output保存每一层的实际输出，和网络结构图中的输出线shape一致
- intput: input保存conv和depthwise的感受野，就是卷积核在feature上移动的小矩阵组成的大矩阵

代码中先生成npy，通过npy2txt处理文件统一制作成两个txt，分别是output.txt和input.txt

后来增加了output中间结果的保存：保存加上bias之前的结果

## 文件功能
- generate_txt.sh: 运行这个文件就能得到所有的输入和输出

- layers.py: 算子实现，主要在里面有添加保存代码。（feature对应卷积核的方框行扫描，对于conv，先通道，再行扫描）

- test_sdcard_numpy.py: 分成output和input保存，output保存每一层的正常输出，input保存的是conv和depthwise的感受野。加入了保存加上bias之前的output

- npy2txt.py: 
  1. 将npy的input和output分别合并后保存成input.txt和output.txt
  2. 将加上bias之前的npy文件转成txt文件，可选择按通道将不同文件存储还是先扫描通道，再行扫描的顺序存储。(真正的output也提供这两种保存方式)


- log:
  - input: conv和depthwise的感受野
  - output: 每层的输出结果
  - sacle: 存储scale的数据
  - output_before_bias: 保存在加上bias之前的数据，分通道保存。其下面有npy和txt文件夹，npy是等着处理成txt的。
  - log文件夹下的txt文件为最终的输出结果