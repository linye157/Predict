# 钢铁大数据平台

## 描述
钢铁大数据平台是一个用于预测钢铁材料性能的机器学习平台。该平台包括数据预处理、模型训练、预测和评估等功能。前端使用Vue.js构建，后端使用Flask框架提供API服务。

## 目录结构
- `mltrain.py`: 训练机器学习模型的脚本。
- `mlpredict.py`: 预测和评估模型的脚本。
- `mlutils.py`: 机器学习模型定义，数据处理和模型定义的工具函数。
- `LSTMTransformer`:LSTMTransformer模型的定义，训练集，验证集处理，训练，测试
- `myflask.py`: Flask API服务，用于接收文件并返回预测结果。
- `my-vue-app/`: 前端Vue.ts应用。

## 预测模型使用
- `mltrain.py`: 直接运行，训练机器学习模型
- `mlpredict.py`: 预测结果，查看机器学习模型性能
- `CNN_BIGRU_Attention.ipynb`:直接运行，完成模型训练和测试等，指标和机器学习模型相同。来源于 “[Beluga whale optimization: A novel nature-inspired metaheuristic algorithm](https://www.sciencedirect.com/science/article/abs/pii/S0950705122006049)”，白鲸算法，虽然听起来不靠谱，但是性能和部分ML模型相当
- `CNN_Transformer.ipynb`:直接运行，完成完成模型训练和测试等，和上面相同，但是把CNN特征直接使用Transformer进行预测(该数据集没有时间特征，所以不打算使用BIGRU机制)，效果和原模型相当
- `resnet_BIGRU_Attention.ipynb`:这个是把数据集改为二位图像特征后使用resnet网络进行特征提取，但是效果一般，**未在本钢数据集进行尝试**
- `LTSMTranformer.ipynb`：这个是最开始的模型，**未在本钢数据集上进行尝试**，但是效果应该和上面的CNN_Transformer差不多，可以按照该文件修改此文件后运行
- `find_problem.ipynb`:这个是通过shap分析各个特征对结果的影响，可以看到各个特征对结果的影响程度，查找出哪个特征因为输入影响较大
- `ga.py`：师兄写的多目标优化算法，还没看，日后再看
- `dataprocess.ipynb`:杂七杂八的数据处理操作，缺失值处理，去除重复行，去除异常值、数据集分割等



## 安装
请按照以下步骤安装和设置项目：

### 后端
1. 克隆仓库：
    ```bash
    git clone https://github.com/yourusername/steel-big-data-platform.git
    ```
2. 进入项目目录：
    ```bash
    cd steel-big-data-platform
    ```

3. 新建"flask_file"文件夹
   该文件夹用于存放预测临时文件
   
4. 运行Flask服务：
    ```bash
    python myflask.py
    ```

### 前端
1. 进入前端项目目录：
    ```bash
    cd my-vue-app
    ```
2. 安装依赖：
    ```bash
    npm install
    ```
3. 运行前端开发服务器：
    ```bash
    npm run dev
    ```