<template>
  <div class="upload-section">
    <div class="file-upload">
      <label  class="file-label">请选择你需要预测的文件</label>
      <input id="file-upload" type="file" @change="handleFileUpload" />
    </div>
    <div class="model-select">
      <label for="model-select" class="file-label2">请选择需要使用的模型</label>
      <select id="model-select" v-model="selectedModel">
        <option value="Lasso">Lasso</option>
        <option value="Kernel Ridge">Kernel Ridge</option>
        <option value="Random Forest">Random Forest</option>
        <option value="Gradient Boosting">Gradient Boosting</option>
        <option value="XGBoost">XGBoost</option>
        <option value="LightGBM">LightGBM</option>
      </select>
    </div>
    <button @click="predict" class="predict-button">预测</button>
  </div>
</template>

<script>
import axios from 'axios';
import { saveAs } from 'file-saver';

export default {
  data() {
    return {
      selectedFile: null,
      selectedModel: 'Lasso',
    };
  },
  methods: {
    handleFileUpload(event) {
      this.selectedFile = event.target.files[0];
    },
    async predict() {
      if (!this.selectedFile) {
        alert('请先选择一个文件');
        return;
      }

      const formData = new FormData();
      formData.append('file', this.selectedFile);
      formData.append('model', this.selectedModel);

      try {
        const response = await axios.post('http://localhost:5050/predict', formData, {
          responseType: 'blob',
        });
        const blob = new Blob([response.data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
        
        const date = new Date();
        const year = date.getFullYear();
        const month = date.getMonth() + 1;
        const day = date.getDate();
        const hours = date.getHours();
        const minutes = date.getMinutes();
        const seconds = date.getSeconds();
        const formattedDate = `${year}-${month.toString().padStart(2, '0')}-${day.toString().padStart(2, '0')}T${hours.toString().padStart(2, '0')}-${minutes.toString().padStart(2, '0')}-${seconds.toString().padStart(2, '0')}`;
        const fileName = `${this.selectedFile.name.split('.')[0]}_${this.selectedModel}_${formattedDate}.xlsx`;
        
        saveAs(blob, fileName);
      } catch (error) {
        console.error('预测请求失败:', error);
      }
    },
  },
};
</script>

<style scoped>
.upload-section {
  display: flex;
  flex-direction: column;
  align-items: center; /* 水平居中 */
  justify-content: center; /* 垂直居中 */
  gap: 20px;
  height: 100%; /* 使组件高度占满 */
}

.file-upload,
.model-select {
  display: flex;
  flex-direction: column; /* 垂直排列标签和输入框 */
  align-items: center; /* 居中对齐 */
  justify-content: center;
  width: 100%; /* 使组件宽度占满 */
  max-width: 600px; /* 设置最大宽度 */
}

.file-label {
  font-size: 20px; /* 修改字体大小 */
  margin-bottom: 10px; /* 标签与输入框之间的间距 */
}

.file-label2 {
  font-size: 20px; /* 修改字体大小 */
  margin-bottom: 10px; /* 标签与输入框之间的间距 */
}

label {
  margin-bottom: 5px; /* 标签与输入框之间的间距 */
}

input[type="file"],
select {
  width: 100%; /* 使输入框和选择框占满剩余空间 */
  padding: 10px; /* 增加内边距 */
  font-size: 16px; /* 增加字体大小 */
}

.predict-button {
  padding: 10px 20px; /* 增加按钮内边距 */
  font-size: 18px; /* 增加按钮字体大小 */
  cursor: pointer; /* 鼠标悬停时显示手型 */
}
</style>
