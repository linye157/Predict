<template>
  <div id="app">
    <Sidebar />
    <div class="main-content">
      <router-view />
    </div>
  </div>
</template>

<script>
import Sidebar from './components/Sidebar.vue';

export default {
  components: {
    Sidebar,
  },
};
</script>


<style scoped>
#app {
  display: flex;
  height: 100vh; /* 使用视口高度，确保应用高度始终占满视口 */
  width: 100vw;  /* 使用视口宽度，确保应用宽度占满整个视口 */
}

.main-content {
  flex: 1;
  padding: 20px;
  margin-left: 200px; /* 固定侧边栏的宽度 */
  overflow-y: auto; /* 允许垂直滚动 */
  display: flex;
  flex-direction: column; /* 使内容垂直排列 */
  min-height: 0; /* 防止内容高度溢出 */
}

@media screen and (max-width: 1920px) {
  .main-content {
    padding: 15px; /* 在更小的分辨率下缩小边距 */
  }
}

@media screen and (max-height: 1080px) {
  .main-content {
    padding: 10px; /* 视口高度小于1080px时，调整边距 */
  }
}
</style>
