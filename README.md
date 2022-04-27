# tensorRT_Demo
C++ tensorRT_Demo model

# 先决条件
1.cuda11.1及以上以及对应的cudann

2.tensorRT 8.2

3.opencv


# 安装：
git clone https://github.com/ljl86092297/tensorRT_Demo.git

# 运行：
## ① 运行我已编译完成的项目 

先到项目文件夹下 然后 执行以下命令

cd bin

./tensorRT_yolov5s 

## ② 自己编译运行 

### 先到项目文件夹下 然后 执行以下命令

cd build 

### 删除编译结果 ,以下命令必须先进入build

`rm -rf ./*`

`cmake ..`

`make`

### 编译完成以后

`cd ../bin`

####  一下操作只能选其中一种 参数不能共存 否侧会出现错误。
##### 图片操作 

`./tensorRT_yolov5s -ir=./ceshi.jpg -iw=./ceshi_new.jpg`

##### 视频操作

`./tensorRT_yolov5s -vr=./test.mp4 -vw=./test_new.mp4`

# 生成：
 执行上述语句会调用bin文件中的测试图片或视频 进行推理， 并将推理结果保存在同目录，（视频操作不用 -vw则不保存）。

