### 1.文件说明：
- __init__.py：main()
- log.py：运行时间、日志 
- data_sources.py：相机内参，单位，坐标系，各种阈值、参数，访问数据文件夹名称
- common_op.py：操作文件
- common_cal.py：多个过程都需要的计算
- preprocessing.py：数据预处理
- back_projection.py：反投影
- coarse_registration.py：粗配准，提取几何特征
- fine_registration.py：精配准，icp

### 2.架构说明
- 1）各个参数解耦，不能出现动一个参数、其他参数跟着动的情况
- 2）提出重复的与算法无关的功能：
    - log.py：AOP通常指Aspect-Oriented Programming，面向切面编程

### 3.数据格式说明
- 输入数据格式说明（颜色格式、数值精度等）
- 输出数据格式说明
- 矩阵/向量格式

### 4.文件依赖关系
log.py,     data_sources.py
common_op.py,   common_cal.py
preprocessing.py

__init__.py
    