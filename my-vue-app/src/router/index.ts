import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'; // 确保引入新的首页视图
import PredictView from '../views/PredictView.vue'; // 新增预测视图

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView, // 使用新的首页视图
  },
  {
    path: '/predict',
    name: 'predict',
    component: PredictView, // 使用新的预测视图
  },
];

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
});

export default router;
