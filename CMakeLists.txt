cmake_minimum_required(VERSION 3.16.3)

project(tensorRT_yolov5s)

aux_source_directory(src SRC_LIST)
#aux_source_directory(/usr/src/tensorrt/samples/common COMMON_LIST)
add_executable(tensorRT_yolov5s  ${SRC_LIST}  )



#设置输出的可执行文件目录
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)



#cuda
include_directories(/usr/local/cuda/include)

#tensorrt
include_directories(/usr/include/x86_64-linux-gnu)
include_directories(/usr/src/tensorrt/samples/common)


#不知道干嘛的
set(THREADS_PREFER_PTHREAD_FLAG ON) #
find_package(Threads) #
target_link_libraries(tensorRT_yolov5s Threads::Threads) 

#下面都是设置opencv的
#找到opencv.cmake  文件
find_package(OpenCV REQUIRED)
#设置opencv 头文件地址
include_directories(${OpenCV_INCLUDE_DIRS})

#设置opencv 库目录  这样后面写库的时候就可以只写文件名了
link_directories(${OpenCV_LIBRARY_DIRS})
#设置opencv库
target_link_libraries(tensorRT_yolov5s ${OpenCV_LIBS})


#下面设置是tensorrt的
set(TENSORRT_LIB_PATH “/usr/lib/x86_64-linux-gnu”)
file(GLOB LIBS “${TENSORRT_LIB_PATH}/*.so”)
#target_link_libraries(tensorRT_yolov5s ${LIBS})

link_directories("/usr/lib/x86_64-linux-gnu")

target_link_libraries(tensorRT_yolov5s "libnvinfer.so"  "libnvinfer_plugin.so" "libnvonnxparser.so" "libnvparsers.so" "libonnx_proto.a")
target_link_libraries(tensorRT_yolov5s "libprotobuf.a")

#设置cuda
find_package(CUDA 11.3 REQUIRED)
message(STATUS "libraries:${CUDA_LIBRARIES}")
message(STATUS "inculde path:${CUDA_INCLUDE_DIRS}")
message(STATUS "woca":${TENSORRT_LIB_PATH})
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(tensorRT_yolov5s ${CUDA_LIBRARIES})
#target_link_libraries(tensorRT_yolov5s /usr/lib/pkgconfig/cudart-11.3.pc)
