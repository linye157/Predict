# 钢铁大数据平台

## 项目概述
面向钢材力学性能预测与工艺参数优化的机器学习平台，集成数据预处理、模型训练、在线预测、结果评估与可视化。后端基于 Flask 提供 API，前端使用 Vue 构建交互界面。

## 核心功能
- 多模型训练与预测：支持多种回归模型，统一的数据分割与特征缩放流程。
- 预测评估：提供 RMSE、MAPE、R2 等指标及业务阈值评估，覆盖屈服强度 YS、抗拉强度 TS、延伸率 EL。
- 在线服务：Flask API 支持文件上传与批量预测。
- 工艺参数优化：提供贝叶斯优化与强化学习（PPO）方案，探索最佳工艺窗口。
- 可视化与分析：SHAP 特征贡献分析与绘图脚本，辅助定位关键工艺因素。

## 目录结构（摘录）
- `mlutils.py`：数据处理、模型定义、路径配置等通用工具。
- `mltrain.py`：调用 `mlutils` 中的模型配置完成批量训练，输出至 `model/`。
- `mlpredict.py`：加载训练模型进行预测与多维指标评估。
- `mltrain_draw_pic.ipynb`：训练过程与结果的可视化示例。
- `find_problem.ipynb`：基于 SHAP 的特征影响分析。
- `dataprocess.ipynb`：缺失值处理、去重、异常值清洗与数据集切分。
- `CNN_BIGRU_Attention.ipynb` / `CNN_Transformer.ipynb` / `resnet_BIGRU_Attention.ipynb` / `LTSMTranformer.ipynb` 等：不同深度模型方案及实验记录。
- `optimize_bayes.ipynb`：贝叶斯优化工艺参数流程。
- `optimize_PPO.ipynb` / `optimize_PPO.py`：基于 PPO 的强化学习优化，需要配置 gym 与 stable-baselines3。
- `ga.py`：多目标进化优化原型。
- `myflask.py`：Flask API，负责接收上传文件并返回预测结果。
- `my-vue-app/`：前端 Vue 应用。
- `data/`、`model/`、`flask_file/` 等：数据、模型与临时上传存储目录。

## 快速开始

### 后端
1. 确保 Python 环境已安装项目依赖（Flask、pandas、scikit-learn、joblib 等）。
2. 创建上传目录 `flask_file/`（用于存放临时预测文件）。
3. 启动服务：
   ```bash
   python myflask.py
   ```

### 前端
1. 进入前端目录：
   ```bash
   cd my-vue-app
   ```
2. 安装依赖：
   ```bash
   npm install
   ```
3. 启动开发服务器：
   ```bash
   npm run dev
   ```

## 模型训练与评估
- 训练：在 `mlutils.py` 配置数据路径与模型列表后，运行 `python mltrain.py` 生成对应的 `.pkl` 模型。
- 评估：运行 `python mlpredict.py`，默认加载测试集并输出业务阈值通过率、RMSE、MAPE、R2 等指标；可按需替换测试集或指定模型名。

## 工艺参数优化
- 贝叶斯优化：`optimize_bayes.ipynb` 通过预测模型提取高贡献工艺参数，设定搜索空间后迭代优化并校验合格性。
- 强化学习优化：`optimize_PPO.ipynb` / `optimize_PPO.py` 结合 PPO 算法探索参数组合，运行前请按脚本提示安装 gym 与 stable-baselines3（注意 torch 版本兼容性，例如 `pip install stable-baselines3[extra] "torch==2.1.0"`）。

## 待办
- 完成模型与预测结果可视化。
- 完善工艺参数优化结果的可视化方案。
- 设计基于 3 项力学指标的逐步工艺参数推荐（输入前 k 个参数预测第 k+1 个）。
- 提供各工艺参数可选范围或最优组合建议。

## 需求状态
- [ ] 支持上传文件并完成力学性能预测（后端接口与前端交互仍需对齐）。
