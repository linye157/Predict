<template>
  <div class="sidebar">
    <h2>钢铁大数据平台</h2>
    <ul>
      <li><router-link to="/" class="nav-button">首页</router-link></li>
      <li><router-link to="/predict" class="nav-button">预测</router-link></li>
    </ul>
  </div>
</template>

<style scoped>
.sidebar {
  width: 200px; /* 固定宽度 */
  height: 100vh; /* 固定高度 */
  background-color: rgb(173, 186, 230); /* 淡蓝色背景 */
  padding: 20px;
  box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
  position: fixed; /* 固定位置 */
  top: 0;
  left: 0; 
  overflow-y: auto; /* 允许垂直滚动 */
}

.main-content {
  margin-left: 200px; /* 预留侧边栏宽度 */
  padding: 20px; /* 添加内边距以防内容紧贴侧边栏 */
}

.sidebar h2 {
  margin-bottom: 20px;
}
.sidebar ul {
  list-style-type: none;
  padding: 0;
}
.sidebar li {
  margin: 10px 0;
}
.nav-button {
  display: inline-block; /* 使链接成为块级元素 */
  padding: 10px 15px; /* 增加内边距以增大按钮 */
  font-size: 18px; /* 增加字体大小 */
  color: hsl(0, 0%, 20%); /* 字体颜色 */
  text-decoration: none; /* 去掉下划线 */
  border-radius: 5px; /* 圆角 */
  transition: background-color 0.3s; /* 添加过渡效果 */
}
.nav-button:hover {
  background-color: hwb(0 88% 12%); /* 悬停时的背景颜色 */
}
</style>
