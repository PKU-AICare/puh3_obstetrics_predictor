<script setup>
import { ref, reactive, onMounted, computed } from 'vue';
import axios from 'axios';
import { ElMessage, ElLoading } from 'element-plus';

// --- Configuration ---
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// --- State Management ---
const singleFile = ref(null);
const batchFile = ref(null);
const predictionResults = ref([]);
const patientId = ref('');
const isLoadingSingle = ref(false);
const isLoadingBatch = ref(false);
const usageStats = ref([]);
const visitStats = ref([]);

const singleUploadRef = ref(null);
const batchUploadRef = ref(null);

const hasResults = computed(() => predictionResults.value.length > 0);

// --- Functions ---
const handleFileChange = (uploadFile, type) => {
  if (type === 'single') {
    singleFile.value = uploadFile.raw;
  } else {
    batchFile.value = uploadFile.raw;
  }
};

const beforeUpload = (rawFile, type) => {
  const isXlsx = rawFile.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
  const isZip = rawFile.type === 'application/zip';
  const isLt5M = rawFile.size / 1024 / 1024 < 5;

  if (!isLt5M) {
    ElMessage.error('文件大小不能超过 5MB!');
    return false;
  }

  if (type === 'single' && !isXlsx) {
    ElMessage.error('请上传 .xlsx 格式的文件!');
    return false;
  }
  if (type === 'batch' && !isZip) {
    ElMessage.error('请上传 .zip 格式的文件!');
    return false;
  }
  return true;
};

const fetchStats = async () => {
  try {
    const [usageRes, visitsRes] = await Promise.all([
      axios.get(`${API_BASE_URL}/api/stats/usage`),
      axios.get(`${API_BASE_URL}/api/stats/visits`),
    ]);
    usageStats.value = usageRes.data;
    visitStats.value = visitsRes.data;
  } catch (error) {
    console.error('获取统计数据失败:', error);
    // Silent fail on stats to not annoy user
  }
};

const downloadTemplate = async () => {
  const loading = ElLoading.service({ text: '正在生成模板...' });
  try {
    const response = await axios.get(`${API_BASE_URL}/api/download-template`, { responseType: 'blob' });
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'prediction_template.xlsx');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  } catch (error) {
    ElMessage.error('模板下载失败，请稍后重试。');
    console.error('模板下载错误:', error);
  } finally {
    loading.close();
  }
};

const submitSinglePrediction = async () => {
  if (!singleFile.value) {
    ElMessage.warning('请先选择一个文件。');
    return;
  }
  isLoadingSingle.value = true;
  predictionResults.value = []; // Clear previous results
  const formData = new FormData();
  formData.append('file', singleFile.value);

  try {
    const response = await axios.post(`${API_BASE_URL}/api/predict-single`, formData);
    patientId.value = response.data.patient_id;
    predictionResults.value = response.data.predictions;
    ElMessage.success(`患者 ${patientId.value} 的预测计算完成！`);
    fetchStats(); // Refresh stats after a successful prediction
  } catch (error) {
    const detail = error.response?.data?.detail || '未知错误';
    ElMessage.error(`计算失败: ${detail}`);
    console.error('单一预测错误:', error);
  } finally {
    isLoadingSingle.value = false;
    singleFile.value = null;
    singleUploadRef.value?.clearFiles();
  }
};

const submitBatchPrediction = async () => {
  if (!batchFile.value) {
    ElMessage.warning('请先选择一个ZIP压缩包。');
    return;
  }
  isLoadingBatch.value = true;
  const formData = new FormData();
  formData.append('file', batchFile.value);

  try {
    const response = await axios.post(`${API_BASE_URL}/api/predict-batch`, formData, { responseType: 'blob' });
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `batch_prediction_results_${new Date().toISOString().slice(0,10)}.xlsx`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    ElMessage.success('批量预测完成，结果文件已开始下载。');
    fetchStats();
  } catch (error) {
    const detail = error.response?.data?.detail || '未知错误';
    ElMessage.error(`批量计算失败: ${detail}`);
    console.error('批量预测错误:', error);
  } finally {
    isLoadingBatch.value = false;
    batchFile.value = null;
    batchUploadRef.value?.clearFiles();
  }
};

const getProbabilityClass = (prob) => {
  if (prob > 0.75) return 'risk-high';
  if (prob > 0.40) return 'risk-medium';
  return 'risk-low';
};

onMounted(() => {
  fetchStats();
});
</script>

<template>
  <div class="app-container">
    <header class="app-header">
      <div class="container header-content">
        <h1 class="title-main">再次妊娠孕期疾病发生风险评估</h1>
        <p class="title-sub">Assessment of Pregnancy-Related Disease Risks in Repeat Pregnancies</p>
      </div>
    </header>

    <main class="container main-content">
      <div class="main-grid">
        <!-- Left Panel: Prediction Tools -->
        <div class="prediction-panel">
          <el-card class="box-card">
            <template #header>
              <div class="card-header">
                <el-icon><User /></el-icon>
                <span>单个患者预测</span>
              </div>
            </template>
            <p class="card-description">上传单个患者的 Excel 文件 (<code>.xlsx</code>) 进行风险评估。文件名将作为患者ID。</p>
            <el-upload
              ref="singleUploadRef"
              class="upload-area"
              drag
              action="#"
              :limit="1"
              :auto-upload="false"
              :on-change="(file) => handleFileChange(file, 'single')"
              :before-upload="(file) => beforeUpload(file, 'single')"
              accept=".xlsx"
            >
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
            </el-upload>
            <div class="button-group">
              <el-button @click="downloadTemplate" :icon="Download">下载模板</el-button>
              <el-button type="primary" @click="submitSinglePrediction" :loading="isLoadingSingle" :icon="Position">开始计算</el-button>
            </div>
          </el-card>

          <el-card class="box-card">
            <template #header>
              <div class="card-header">
                <el-icon><Files /></el-icon>
                <span>批量预测</span>
              </div>
            </template>
             <p class="card-description">上传包含多个患者 Excel 文件的 ZIP 压缩包 (<code>.zip</code>) 进行批量预测。</p>
             <el-upload
              ref="batchUploadRef"
              class="upload-area"
              drag
              action="#"
              :limit="1"
              :auto-upload="false"
              :on-change="(file) => handleFileChange(file, 'batch')"
              :before-upload="(file) => beforeUpload(file, 'batch')"
              accept=".zip"
            >
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">将 ZIP 包拖到此处，或<em>点击上传</em></div>
            </el-upload>
            <div class="button-group">
              <el-button type="primary" @click="submitBatchPrediction" :loading="isLoadingBatch" :icon="Promotion">处理并导出结果</el-button>
            </div>
          </el-card>
        </div>

        <!-- Right Panel: Results and Stats -->
        <div class="results-panel">
          <el-card v-if="hasResults" class="box-card results-card">
             <template #header>
              <div class="card-header">
                <el-icon><DataAnalysis /></el-icon>
                <span>预测结果 (患者ID: {{ patientId }})</span>
              </div>
            </template>
            <el-table :data="predictionResults" stripe style="width: 100%">
              <el-table-column prop="disease_name_cn" label="疾病名称" min-width="180" />
              <el-table-column prop="disease_abbr" label="缩写" width="100" />
              <el-table-column label="预测概率" width="120" align="right">
                <template #default="scope">
                  <span :class="getProbabilityClass(scope.row.probability)">
                    {{ (scope.row.probability * 100).toFixed(2) }}%
                  </span>
                </template>
              </el-table-column>
            </el-table>
          </el-card>
           <el-card v-else class="box-card placeholder-card">
             <el-empty description="请上传文件以查看预测结果" />
           </el-card>

          <div class="stats-grid">
            <el-card class="box-card">
              <template #header>
                <div class="card-header">
                  <el-icon><Medal /></el-icon>
                  <span>使用次数统计</span>
                </div>
              </template>
              <ul v-if="usageStats.length > 0" class="stats-list">
                <li v-for="stat in usageStats" :key="stat.location">
                  <span class="location">{{ stat.location }}</span>
                  <span class="count">{{ stat.count }} 次</span>
                </li>
              </ul>
              <el-empty v-else description="暂无数据" :image-size="60" />
            </el-card>

            <el-card class="box-card">
               <template #header>
                <div class="card-header">
                  <el-icon><TrendCharts /></el-icon>
                  <span>访问地区统计</span>
                </div>
              </template>
              <ul v-if="visitStats.length > 0" class="stats-list">
                <li v-for="stat in visitStats" :key="stat.location">
                  <span class="location">{{ stat.location }}</span>
                  <span class="count">{{ stat.count }} 次</span>
                </li>
              </ul>
              <el-empty v-else description="暂无数据" :image-size="60" />
            </el-card>
          </div>
        </div>
      </div>
    </main>

    <footer class="app-footer">
      <div class="container">
        <p>国家妇产疾病临床医学研究中心 · 北京大学第三医院妇产科生殖医学中心</p>
        <p class="eng-footer">National Clinical Research Center for Obstetrics and Gynecology, Peking University Third Hospital</p>
      </div>
    </footer>
  </div>
</template>

<style>
/* --- Global Styles & Variables --- */
:root {
  --color-primary: #409eff;
  --color-primary-light: #ecf5ff;
  --color-success: #67c23a;
  --color-warning: #e6a23c;
  --color-danger: #f56c6c;
  --color-text-primary: #303133;
  --color-text-regular: #606266;
  --color-text-secondary: #909399;
  --border-color: #dcdfe6;
  --bg-color-page: #f2f3f5;
  --bg-color-card: #ffffff;
  --font-family-main: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
}

body {
  margin: 0;
  font-family: var(--font-family-main);
  background-color: var(--bg-color-page);
  color: var(--color-text-primary);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.container {
  width: 90%;
  max-width: 1400px;
  margin: 0 auto;
}

/* --- App Layout --- */
.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.app-header {
  background: linear-gradient(135deg, #3f51b5 0%, #2196f3 100%);
  color: white;
  padding: 24px 0;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.header-content {
  text-align: center;
}

.title-main {
  font-size: 2rem;
  font-weight: 600;
  margin: 0;
}

.title-sub {
  font-size: 1rem;
  font-weight: 300;
  opacity: 0.8;
  margin-top: 8px;
}

.main-content {
  padding: 32px 0;
  flex-grow: 1;
}

.main-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 32px;
}

.prediction-panel, .results-panel {
  display: flex;
  flex-direction: column;
  gap: 32px;
}

.app-footer {
  background-color: #303133;
  color: var(--color-text-secondary);
  text-align: center;
  padding: 20px 0;
  font-size: 0.875rem;
}

.app-footer p {
  margin: 4px 0;
}
.eng-footer {
  font-size: 0.75rem;
  opacity: 0.7;
}


/* --- Card Styles --- */
.box-card {
  border: 1px solid var(--border-color);
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.06);
}

.el-card__header {
  background-color: #fafafa;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--color-text-primary);
}

.card-description {
  font-size: 0.9rem;
  color: var(--color-text-regular);
  margin-top: 0;
  margin-bottom: 20px;
  line-height: 1.6;
}
.card-description code {
  background-color: var(--color-primary-light);
  color: var(--color-primary);
  padding: 2px 4px;
  border-radius: 4px;
  font-size: 0.85rem;
}

/* --- Components Styles --- */
.upload-area .el-upload-dragger {
  padding: 30px;
}

.button-group {
  margin-top: 20px;
  display: flex;
  justify-content: space-between;
}

.results-card .el-table {
  font-size: 0.9rem;
}

.risk-high {
  color: var(--color-danger);
  font-weight: bold;
}
.risk-medium {
  color: var(--color-warning);
  font-weight: bold;
}
.risk-low {
  color: var(--color-success);
  font-weight: bold;
}

.placeholder-card .el-card__body {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 300px;
}

.stats-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 32px;
}

.stats-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.stats-list li {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.95rem;
  color: var(--color-text-regular);
  padding-bottom: 8px;
  border-bottom: 1px solid #f0f0f0;
}
.stats-list li .location {
  font-weight: 500;
}
.stats-list li .count {
  font-weight: bold;
  color: var(--color-primary);
}

/* --- Responsive --- */
@media (max-width: 1200px) {
  .main-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .title-main {
    font-size: 1.5rem;
  }
  .title-sub {
    font-size: 0.9rem;
  }
  .button-group {
    flex-direction: column;
    gap: 12px;
  }
  .button-group .el-button {
    width: 100%;
    margin-left: 0 !important;
  }
}
</style>