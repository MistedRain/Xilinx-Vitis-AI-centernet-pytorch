#include <dirent.h>
#include <sys/stat.h>
#include <zconf.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <ctime>
#include <stdlib.h>
#include <semaphore.h>
#include <sched.h>

#include "common.h"
#include "utils.h"

#define _GNU_SOURCE
#define SET_WIDTH 320
#define SET_HEIGHT 240

using namespace std;
using namespace cv;
using namespace std::chrono;

int idxInputImage = 0;  // frame index of input video
int idxShowImage = 0;   // next frame index to be displayed
int idxcnt_M = 0; //1000进1
bool bReading = true;   // 可读标志，如果按'q' breading会设置为false
bool bExiting = false; //视频读完将bexiting设置为true

/*对共享队列需进行互斥操作*/
mutex mtxQueueInput;//摄像头获取图片队列加锁
mutex mtxQueueShow;//摄像头获取展示队列加锁
sem_t inputframe,showframe;//信号量，队列

int im_w = SET_WIDTH;
int im_h = SET_HEIGHT;
int new_w = 0;
int new_h = 0;

chrono::system_clock::time_point start_time;

typedef pair<int, Mat> imagePair;
/*仿函数类 用于构造优先级队列*/
class paircomp {
    public:
    /*输出图片元组顺序先后情况，返回true代表n1在n2前面*/
    bool operator()(const imagePair& n1, const imagePair& n2) const {
        if (n1.first == n2.first) {
        return (n1.first > n2.first);
        }

        return n1.first > n2.first;
    }
};

queue<pair<int, Mat>> queueInput;// 摄像头获取图片队列
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow;//效果展示图片队列，优先级队列，自动按由小到大顺序插入到队列

float input_scale;
vector<float> output_scale;

/*VOC数据集20个类别*/
const string classes[20] = {"aeroplane",
"bicycle",
"bird",
"boat",
"bottle",
"bus",
"car",
"cat",
"chair",
"cow",
"diningtable",
"dog",
"horse",
"motorbike",
"person",
"pottedplant",
"sheep",
"sofa",
"train",
"tvmonitor"};

GraphInfo shapes;//存储输入、输出张量形状的结构体

// fix_point to scale for output tensor
static float get_input_scale(const xir::Tensor* tensor) {
    int fixpos = tensor->template get_attr<int>("fix_point");
    return std::exp2f(1.0f * (float)fixpos);
}
static float get_output_scale(const xir::Tensor* tensor) {
    int fixpos = tensor->template get_attr<int>("fix_point");
    return std::exp2f(-1.0f * (float)fixpos);
}

/*从video或摄像头中读取每一帧存入InputQueue中*/
void readFrame(VideoCapture video){

	/*将此函数绑定在CPU2上运行*/
	cpu_set_t mask;
	cpu_set_t get;
	
	CPU_ZERO(&mask);//初始化set集，将set置为空
	CPU_SET(2,&mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        printf("Set CPU affinity failue");
        exit(0); 
    }
    /*设置获取摄像头的分辨率*/
    video.set(CV_CAP_PROP_FRAME_WIDTH, SET_WIDTH);
    video.set(CV_CAP_PROP_FRAME_HEIGHT, SET_HEIGHT);

    start_time = chrono::system_clock::now();
    /*乒乓缓冲：交替读和写*/ 
    bool ping = true;
    bool pang = true;
    Mat img1,img2;
    while(true) {
        if(ping){
            if (queueInput.size() < 100) {//如果目前输入缓冲已经有100张图片，执行延迟等待10操作
                if (!video.read(img1)) {//判断如果没有读到帧，说明没有要读的帧了，则break
                    break;
                }
                if(pang == false){
                    mtxQueueInput.lock();
                    queueInput.push(make_pair(idxInputImage++, img2));
					if(idxInputImage==1000) idxInputImage=0;//循环计数，1000归零
                    sem_post(&inputframe);
                    mtxQueueInput.unlock();
                }
            } 
            else {//如果目前输入缓冲已经有100张图片，执行延迟等待10操作
                usleep(10);
            }
            ping = false;
            pang = true;
        }
        else{
            if (queueInput.size() < 100) {
                if (!video.read(img2)) {
                    break;
                }
                if(pang == true){
                    mtxQueueInput.lock();
                    queueInput.push(make_pair(idxInputImage++, img1));
					if(idxInputImage==1000) idxInputImage=0;//循环计数，1000归零
                    sem_post(&inputframe);
                    mtxQueueInput.unlock();
                }
            } 
            else {
                usleep(10);
            }
            ping = true;
            pang = false;
        }
    }
	//如果没有需要读的帧，释放视频，将退出信号设置为true
    video.release();
    bExiting = true;
}

/*只保留置信度前topK个目标*/
void topK(vector<vector<int>> &TOPk,vector<float>result,int k,int height,int width,int channel){

    priority_queue< pair<float , vector<int>> >q;
    /*将所有大于置信度的中心点加入队列q中*/
    for (int c = 0; c < channel; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int temp = c * height * width + h * width + w;
                if(result[temp] > 0.5){
                    vector<int>tmp;
                    tmp.push_back(c);
                    tmp.push_back(h);
                    tmp.push_back(w);
                    q.push(pair<float, vector<int>>(result[temp],tmp));
                }
            
            }
        }
    }
    /*将前k个符合的中心点加入到TOPk队列中*/
    k = k > q.size() ? q.size():k;
    for (auto i = 0; i < k; ++i) {
        pair<float , vector<int>> ki = q.top();
        TOPk.push_back(ki.second);
        q.pop();
    }
}

/*对网络输出的数据results进行处理 并将结果框标注在frame上*/
void postProcess(vart::Runner* runner, Mat& frame, vector<int8_t*> &results,
                int sWidth, int sHeight, const float* output_scale) {

    vector<vector<float>> results_float;
    vector<vector<int>> TopK;
    /*构建输出张量的形状，因为长、宽都是128x128，因此仅获取一次*/
    int width = shapes.outTensorList[0].width;
    int height = shapes.outTensorList[0].height;
    /*通过3重循环（三个输出）得到每个输出*/
    for (int ii = 0; ii < 3; ii++) {
        int channel = shapes.outTensorList[ii].channel;
        int sizeOut = channel * width * height;
        vector<float>result(sizeOut);

        /*对第一个输出通道做特殊处理sigmoid->maxppol->topk*/
        if(ii == 0){
            for (int a = 0; a < channel; ++a) {
                for (int b = 0; b < height; ++b) {
                    for (int c = 0; c < width; ++c) {
						int offset = b * channel * width + c * channel + a;
                        result[a*height*width+b*width+c] = 1.0 / (1 + exp(-(results[ii][offset]*output_scale[ii]) * 1.0));//sigmoid,维度变换，尺度变换
                    }
                }
            }
            vector<float>result_tmp(sizeOut);
            /*utils中，函数将result的data经过centernet的maxpool操作后存在result_tmp数组中*/
            forward_maxpool_layer(width,height,channel,height,width,1,3,1,result.data(),result_tmp.data());
            result.swap(result_tmp);//为什么要进行赋值？
            topK(TopK,result,100,height,width,channel);//选择置信度排行前100个框
        }
		else{
			for (int a = 0; a < channel; ++a) {
                for (int b = 0; b < height; ++b) {
                    for (int c = 0; c < width; ++c) {
						int offset = b * channel * width + c * channel + a;
                        result[a*height*width+b*width+c] = results[ii][offset]*output_scale[ii];//维度变换，尺度变换
                    }
                }
            }
		}
        results_float.push_back(result);//将第一个输入的结果加入总结果中
    }

    /*将框标注在frame上步骤：确定缩放因子、确定框的实际宽高、通过左上和右下2个点坐标在frame上画框*/
    /*确定缩放因子*/
    int frame_width = frame.cols;
    int frame_height = frame.rows;
    float frame_scale_w;
    float frame_scale_h;
    if(frame_width>=frame_height){
        frame_scale_w = 1;
        frame_scale_h = frame_width / float(frame_height);
    }
    else{
        frame_scale_h = 1;
        frame_scale_w = frame_height / float(frame_width);
    }

    /*对每一个框进行处理，确定框的左上右下两个点*/
    for(auto &tmp : TopK){
        int c = tmp[0];
        int h = tmp[1];
        int w = tmp[2];
        int temp = c * height * width + h * width + w;
        //cout<<classes[c]<<" "<<results_float[0][temp]<<endl;
        if(results_float[0][temp] > CONF){//需要这个判断吗？
            float out_w = results_float[1][h * width + w] *1.0 / float(width);
            float out_h = results_float[1][height * width + h * width + w] *1.0 / float(height);
            float x = (results_float[2][h * width + w] + w)/ float(width) ;
            float y = (results_float[2][height * width + h * width + w] + h)/ float(height) ;

            float x_min = (x - out_w/2.0)*(frame_width)*frame_scale_w - (frame_width*frame_scale_w - frame_width)/2+ 1;
            float x_max = (x + out_w/2.0)*(frame_width)*frame_scale_w - (frame_width*frame_scale_w - frame_width)/2 + 1;

            float y_min = (y - out_h/2.0)*(frame_height)*frame_scale_h - (frame_height*frame_scale_h - frame_height)/2+ 1;
            float y_max = (y + out_h/2.0)*(frame_height)*frame_scale_h - (frame_height*frame_scale_h - frame_height)/2 + 1;

            rectangle(frame, Point(x_min, y_min), Point(x_max, y_max),Scalar(0, 255, 255), 1, 1, 0);//画框
        }
    }

}

/*获取图片->预处理->推断->后处理结果->画框*/
void runCenterNet(vart::Runner* runner){
	
	/*将此函数绑定在CPU0,1上运行*/
	cpu_set_t mask;
	cpu_set_t get;
	
	CPU_ZERO(&mask);//初始化set集，将set置为空
	CPU_SET(0,&mask);
	CPU_SET(1,&mask);
	CPU_SET(3,&mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        printf("Set CPU affinity failue");
        exit(0); 
    } 
	
    /*构建dpu输入、输出格式*/
    auto inputTensors = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();
    auto out_dims0 = outputTensors[0]->get_shape();
    auto out_dims1 = outputTensors[1]->get_shape();
    auto out_dims2 = outputTensors[2]->get_shape();
    auto in_dims = inputTensors[0]->get_shape();
    int width = shapes.inTensorList[0].width;
    int height = shapes.inTensorList[0].height;
    int size = shapes.inTensorList[0].size;

    int8_t* result0 = new int8_t[shapes.outTensorList[0].size * outputTensors[shapes.output_mapping[0]]->get_shape().at(0)];
    int8_t* result1 = new int8_t[shapes.outTensorList[1].size * outputTensors[shapes.output_mapping[1]]->get_shape().at(0)];
    int8_t* result2 = new int8_t[shapes.outTensorList[2].size * outputTensors[shapes.output_mapping[2]]->get_shape().at(0)];

    /*result中存放输入和输出格式*/
    int8_t* data = new int8_t[shapes.inTensorList[0].size * inputTensors[0]->get_shape().at(0)]; 
    vector<int8_t*> result;
    result.push_back(result0);
    result.push_back(result1);
    result.push_back(result2);

    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
    std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
    std::vector<std::shared_ptr<xir::Tensor>> batchTensors;
    float scale = pow(2, 7);
    pair<int, Mat> pairIndexImage;

    /*从输入队列中取一帧图片*/
    while (true)
    {
        sem_wait(&inputframe);
		
        mtxQueueInput.lock();
        pairIndexImage = queueInput.front();
        queueInput.pop();
        mtxQueueInput.unlock();
        if(!bReading) break;

        /*图片预处理：缩放、填充灰框，将预处理图片存放在bb数组中*/
        vector<float>bb(size);//size为输入图像的大小
        letterbox_image(pairIndexImage.second,bb,width,height,new_w,new_h);
        
        /*将预处理后的图片bb尺度变换后赋值给data*/
        for (int i = 0; i < size; ++i) {
            data[i] = (int8_t)(bb.data()[i] * input_scale);
            if (data[i] < 0) data[i] = (int8_t)((float)(127 / scale) * input_scale);
        }

        /*构建inputs*/
        batchTensors.push_back(std::shared_ptr<xir::Tensor>(
            xir::Tensor::create(inputTensors[0]->get_name(), in_dims,
                                xir::DataType{xir::DataType::XINT, 8u})));
        inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
            data, batchTensors.back().get()));
        /*构建outputs（有3个output）*/
        batchTensors.push_back(std::shared_ptr<xir::Tensor>(
            xir::Tensor::create(outputTensors[0]->get_name(), out_dims0,
                                xir::DataType{xir::DataType::XINT, 8u})));
        outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
            result[0], batchTensors.back().get()));
        batchTensors.push_back(std::shared_ptr<xir::Tensor>(
            xir::Tensor::create(outputTensors[1]->get_name(), out_dims1,
                                xir::DataType{xir::DataType::XINT, 8u})));
        outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
            result[1], batchTensors.back().get()));
        batchTensors.push_back(std::shared_ptr<xir::Tensor>(
            xir::Tensor::create(outputTensors[2]->get_name(), out_dims2,
                                xir::DataType{xir::DataType::XINT, 8u})));
        outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
            result[2], batchTensors.back().get()));
        
        /*dpu输入格式要求*/
        inputsPtr.push_back(inputs[0].get());
        outputsPtr.resize(3);
        outputsPtr[shapes.output_mapping[0]] = outputs[0].get();
        outputsPtr[shapes.output_mapping[1]] = outputs[1].get();
        outputsPtr[shapes.output_mapping[2]] = outputs[2].get();

        /*进行dpu推断*/
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);//返回的输出再outputsPtr中，也可通过result获得
        runner->wait(job_id.first, -1);

        /*后处理，将得到的result处理为识别分类和框信息，并还原为原图数据，画框*/
        postProcess(runner, pairIndexImage.second, result, width, height,output_scale.data());
       
        /*将画好框的图片放入展示队列中*/
        mtxQueueShow.lock();
        queueShow.push(pairIndexImage);
        mtxQueueShow.unlock();
        sem_post(&showframe);
        /*释放中间变量*/
        batchTensors.clear();
        inputs.clear();
        outputs.clear();
        inputsPtr.clear();
        outputsPtr.clear();
    }
    /*释放中间变量*/
    delete[] result0;
    delete[] result1;
    delete[] result2;
    delete[] data;

}

/*按顺序播放show队列中的图片*/
void displayFrame(){
	/*将此函数绑定在CPU2上运行*/
	cpu_set_t mask;
	cpu_set_t get;
	
	CPU_ZERO(&mask);//初始化set集，将set置为空
	CPU_SET(2,&mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        printf("Set CPU affinity failue");
        exit(0); 
    }
    Mat frame1,frame2;
    bool ping = true;
    bool pang = true;
    while (true) {
        if(ping){
            if (bExiting&&queueShow.empty()) break;           
			sem_wait(&showframe);
			/*轮到播放的图片为播放队列的第一张图片*/   			
			if (idxShowImage == queueShow.top().first) {//idxShowImage为全局变量，表示展示的序号
				mtxQueueShow.lock();
                auto show_time = chrono::system_clock::now();
                stringstream buffer;
                frame1 = queueShow.top().second;
                if (frame1.rows <= 0 || frame1.cols <= 0) {
                    mtxQueueShow.unlock();
                    continue;
                }
                else if(pang == false && frame2.rows > 0 && frame2.cols > 0){
                    /*计算帧率*/
                    
					auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
                    buffer << fixed << setprecision(1)
                            << (idxcnt_M*1000+idxShowImage+1) / (dura / 2000000.f);
                    string a = buffer.str() + " FPS";
                    /*展示图片*/
                    cv::putText(frame2, a, cv::Point(10, 15), 1, 1, cv::Scalar{0, 0, 240}, 1);
                    cv::imshow("CenterNet Detection@Xilinx DPU", frame2); 			
                }
				if(++idxShowImage==1000){
					idxShowImage=0;//循环计数，1000归零
					idxcnt_M++;
				}
				queueShow.pop();
                mtxQueueShow.unlock();
				/*每张图片展示时间为1ms，按'q'退出*/
                if (waitKey(1) == 'q') {
                    bReading = false;
					sem_destroy(&inputframe);
					sem_destroy(&showframe);
					cout<<"程序已退出"<<endl;
                    exit(0);
                }
            }
			/*需要展示的图片还未处理好，继续等待*/ 			
            else {
                sem_post(&showframe);
                mtxQueueShow.unlock();
            }
            ping = false;
            pang = true;
        }
        else{//ping==false，双通道，逻辑同上
            if (bExiting&&queueShow.empty()) break;
            sem_wait(&showframe);
			if (idxShowImage == queueShow.top().first) {
				mtxQueueShow.lock();
                auto show_time = chrono::system_clock::now();
                stringstream buffer;
                frame2 = queueShow.top().second;
                if (frame2.rows <= 0 || frame2.cols <= 0) {
                    mtxQueueShow.unlock();
                    continue;
                }
                else if(pang == true && frame1.rows > 0 && frame1.cols > 0){
                    
					
                    auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
                    buffer << fixed << setprecision(1)
                            << (idxcnt_M*1000+idxShowImage+1) / (dura / 2000000.f);
                    string a = buffer.str() + " FPS";
                    cv::putText(frame1, a, cv::Point(10, 15), 1, 1, cv::Scalar{0, 0, 240}, 1);
                    cv::imshow("CenterNet Detection@Xilinx DPU", frame1);
					
                }
				if(++idxShowImage==1000){
					idxShowImage=0;//循环计数，1000归零
					idxcnt_M++;
				}
                queueShow.pop();
                mtxQueueShow.unlock();
                if (waitKey(1) == 'q') {			
					bReading = false;
					sem_destroy(&inputframe);
					sem_destroy(&showframe);
					cout<<"程序已退出"<<endl;
                    exit(0);
                }                
            } 
            else {
                mtxQueueShow.unlock();
                sem_post(&showframe);
            }
            ping = true;
            pang = false;
        }

    }
}

/*主函数*/
int main(const int argc, const char** argv) {
    /*输入参数为文件名 视频文件 模型文件*/
    if (argc != 3) {
    cout << "Usage of CenterNet detection: " << argv[0]
            << "<video file> <model_file>" << endl;
    return -1;
    }

    sem_init(&inputframe,0,0);//input队列初始为0
    sem_init(&showframe,0,0);//show队列初始为0

    auto graph = xir::Graph::deserialize(argv[2]);//获取模型子图
    auto subgraph = get_dpu_subgraph(graph.get());//获取DPU可运行的子图
    CHECK_EQ(subgraph.size(), 1u)//判断DPU子图数量是否为1
        << "CenterNet should have one and only one dpu subgraph.";
    LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

    /*根据DPU子图 创建12个dpu runner用来多线程推断*/
    auto runner1 = vart::Runner::create_runner(subgraph[0], "run");
    auto runner2 = vart::Runner::create_runner(subgraph[0], "run");
    auto runner3 = vart::Runner::create_runner(subgraph[0], "run");
    auto runner4 = vart::Runner::create_runner(subgraph[0], "run");
    auto runner5 = vart::Runner::create_runner(subgraph[0], "run");
    auto runner6 = vart::Runner::create_runner(subgraph[0], "run");
    auto runner7 = vart::Runner::create_runner(subgraph[0], "run");
    auto runner8 = vart::Runner::create_runner(subgraph[0], "run");
    auto runner9 = vart::Runner::create_runner(subgraph[0], "run");
    auto runner10 = vart::Runner::create_runner(subgraph[0], "run");
    auto runner11 = vart::Runner::create_runner(subgraph[0], "run");
    auto runner12 = vart::Runner::create_runner(subgraph[0], "run");
    
    /*根据子图信息获取输入、输出张量的形状*/
    auto inputTensors = runner1->get_input_tensors();
    auto outputTensors = runner1->get_output_tensors();
    int inputCnt = inputTensors.size();
    int outputCnt = outputTensors.size();

    /*将输入、输出张量的形状参数存入shapes结构体中*/ 
    TensorShape inshapes[inputCnt];
    TensorShape outshapes[outputCnt];
    shapes.inTensorList = inshapes;
    shapes.outTensorList = outshapes;
    getTensorShape(runner1.get(), &shapes, inputCnt,outputCnt);

    /*获取需要传入网络的图片尺寸*/
    int width = shapes.inTensorList[0].width;
    int height = shapes.inTensorList[0].height;
    int size = shapes.inTensorList[0].size;

    /*获取需要将图片缩放的尺寸(new_w,new_h)*/
    if (((float)width / im_w) < ((float)height / im_h)) {
        new_w = width;
        new_h = (im_h * width) / im_w;
    } else {
        new_h = height;
        new_w = (im_w * height) / im_h;
    }

    /*获取输入、输出缩放尺度*/
    input_scale = get_input_scale(runner1->get_input_tensors()[0]);
    for (int i; i < 3; i++) {
        output_scale.push_back(get_output_scale(runner1->get_output_tensors()[shapes.output_mapping[i]]));
    }

    /*视频获取*/
    string filename = argv[1];
    VideoCapture capture;
    if(filename != "0" ){//如果是视频文件，直接打开
        capture.open(filename);
    }
    else{//如果是摄像头，转为整数0代表从摄像头中获取
        capture.open(atoi(argv[1]));
    }
    if(!capture.isOpened()){//打开摄像头失败情形
        cout<<"open cam failed !"<<endl;
        return -1;
    }
    Mat tmpMat;
    if (!capture.read(tmpMat)) {//读取每一帧图片失败情形
        cout<<"read image failed !"<<endl;
        return -1;
    }
    
    /*构建多线程运行列表：线程0从摄像头中读取帧、线程1-12进行预处理推断及后处理、线程13展示图片*/
    array<thread,14> threadsList = {
    thread(readFrame, capture), 
    thread(runCenterNet,runner1.get()),thread(runCenterNet,runner2.get()),
    thread(runCenterNet,runner3.get()),thread(runCenterNet,runner4.get()),
    thread(runCenterNet,runner5.get()),thread(runCenterNet,runner6.get()),
    thread(runCenterNet,runner7.get()),thread(runCenterNet,runner8.get()),
    thread(runCenterNet,runner9.get()),thread(runCenterNet,runner10.get()),
    thread(runCenterNet,runner11.get()),thread(runCenterNet,runner12.get()),
    thread(displayFrame)
    };
    for (int i = 0; i < 14; i++) {//主线程等待子线程结束
        threadsList[i].join();
    }
    
    return 0;
}