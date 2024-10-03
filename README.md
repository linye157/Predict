# 钢铁大数据平台

## 描述
钢铁大数据平台是一个用于预测钢铁材料性能的机器学习平台。该平台包括数据预处理、模型训练、预测和评估等功能。前端使用Vue.js构建，后端使用Flask框架提供API服务。

## 目录结构
- `mltrain.py`: 训练机器学习模型的脚本。
- `mlpredict.py`: 预测和评估模型的脚本。
- `mlutils.py`: 数据处理和模型定义的工具函数。
- `myflask.py`: Flask API服务，用于接收文件并返回预测结果。
- `my-vue-app/`: 前端Vue.js应用。

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
    ```
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