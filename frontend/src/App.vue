<script setup>
import { ref, computed, onMounted } from 'vue';
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

// --- API & Helper Functions ---
const fetchStats = async () => {
  try {
    const [usageRes, visitsRes] = await Promise.all([
      axios.get(`${API_BASE_URL}/api/stats/usage`),
      axios.get(`${API_BASE_URL}/api/stats/visits`),
    ]);
    usageStats.value = usageRes.data;
    visitStats.value = visitsRes.data;
  } catch (error) {
    console.error('Failed to fetch stats:', error);
  }
};

const downloadTemplate = async () => {
  const loading = ElLoading.service({ lock: true, text: '正在生成模板...', background: 'rgba(0, 0, 0, 0.7)' });
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
    console.error('Template download error:', error);
  } finally {
    loading.close();
  }
};

const handleFileChange = (uploadFile, type) => {
  const file = uploadFile.raw;
  const isXlsx = file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
  const isZip = file.type === 'application/zip' || file.name.endsWith('.zip');

  if (type === 'single') {
    if (!isXlsx) { ElMessage.error('请上传 .xlsx 格式的文件!'); return false; }
    singleFile.value = file;
  } else {
    if (!isZip) { ElMessage.error('请上传 .zip 格式的文件!'); return false; }
    batchFile.value = file;
  }
};

const submitSinglePrediction = async () => {
  if (!singleFile.value) {
    ElMessage.warning('请先选择一个患者的 Excel 文件。');
    return;
  }
  isLoadingSingle.value = true;
  predictionResults.value = [];
  const formData = new FormData();
  formData.append('file', singleFile.value);

  try {
    const response = await axios.post(`${API_BASE_URL}/api/predict-single`, formData);
    patientId.value = response.data.patient_id;
    predictionResults.value = response.data.predictions.sort((a, b) => b.probability - a.probability);
    ElMessage.success(`患者 ${patientId.value} 的风险评估已完成！`);
    fetchStats();
  } catch (error) {
    const detail = error.response?.data?.detail || '未知错误，请检查文件格式或联系管理员。';
    ElMessage.error(`计算失败: ${detail}`);
  } finally {
    isLoadingSingle.value = false;
    singleUploadRef.value?.clearFiles();
    singleFile.value = null;
  }
};

const submitBatchPrediction = async () => {
  if (!batchFile.value) {
    ElMessage.warning('请先选择一个包含多个患者文件的 ZIP 压缩包。');
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
    const contentDisposition = response.headers['content-disposition'];
    let filename = `batch_results_${Date.now()}.xlsx`;
    if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
        if (filenameMatch.length === 2) filename = filenameMatch[1];
    }
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
    ElMessage.success('批量预测完成，结果文件已开始下载。');
    fetchStats();
  } catch (error) {
    const detail = error.response?.data?.detail || '未知错误，请检查ZIP包内容或联系管理员。';
    ElMessage.error(`批量计算失败: ${detail}`);
  } finally {
    isLoadingBatch.value = false;
    batchUploadRef.value?.clearFiles();
    batchFile.value = null;
  }
};

const getProbabilityClass = (prob) => {
  if (prob > 0.6) return 'risk-high';
  if (prob > 0.3) return 'risk-medium';
  return 'risk-low';
};

onMounted(fetchStats);
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
                <el-icon><User /></el-icon><span>单个患者预测 (Single Patient)</span>
              </div>
            </template>
            <p class="card-description">上传单个患者的 Excel 文件 (<code>.xlsx</code>) 进行风险评估。文件名将作为患者ID。</p>
            <el-upload ref="singleUploadRef" drag action="#" :limit="1" :auto-upload="false" :on-change="(file) => handleFileChange(file, 'single')" accept=".xlsx">
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
            </el-upload>
            <div class="button-group">
              <el-button @click="downloadTemplate" :icon="Download">下载模板 (Download Template)</el-button>
              <el-button type="primary" @click="submitSinglePrediction" :loading="isLoadingSingle" :icon="Position">开始计算 (Calculate)</el-button>
            </div>
          </el-card>

          <el-card class="box-card">
            <template #header>
              <div class="card-header">
                <el-icon><Files /></el-icon><span>批量预测 (Batch Prediction)</span>
              </div>
            </template>
            <p class="card-description">上传包含多个患者 Excel 文件的 ZIP 压缩包 (<code>.zip</code>) 进行批量预测。</p>
            <el-upload ref="batchUploadRef" drag action="#" :limit="1" :auto-upload="false" :on-change="(file) => handleFileChange(file, 'batch')" accept=".zip">
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">将 ZIP 包拖到此处，或<em>点击上传</em></div>
            </el-upload>
            <div class="button-group">
              <el-button type="primary" @click="submitBatchPrediction" :loading="isLoadingBatch" :icon="Promotion">处理并导出 (Process & Export)</el-button>
            </div>
          </el-card>
        </div>

        <!-- Right Panel: Results and Stats -->
        <div class="results-panel">
          <el-card v-if="hasResults" class="box-card results-card">
            <template #header>
              <div class="card-header">
                <el-icon><DataAnalysis /></el-icon><span>预测结果 (Patient ID: {{ patientId }})</span>
              </div>
            </template>
            <el-table :data="predictionResults" stripe height="450">
              <el-table-column prop="disease_name_cn" label="疾病名称 (Disease)" min-width="180" />
              <el-table-column prop="disease_abbr" label="缩写 (Abbr.)" width="100" />
              <el-table-column label="预测概率 (Probability)" width="150" align="center">
                <template #default="scope">
                  <span :class="['risk-tag', getProbabilityClass(scope.row.probability)]">
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
              <template #header><div class="card-header"><el-icon><Medal /></el-icon><span>使用次数排行 (Top Usage)</span></div></template>
              <ul v-if="usageStats.length" class="stats-list"><li v-for="(stat, index) in usageStats" :key="stat.location"><span class="rank">{{ index + 1 }}</span><span class="location">{{ stat.location }}</span><span class="count">{{ stat.count }} 次</span></li></ul>
              <el-empty v-else description="暂无数据" :image-size="60" />
            </el-card>
            <el-card class="box-card">
              <template #header><div class="card-header"><el-icon><TrendCharts /></el-icon><span>访问地区排行 (Top Visits)</span></div></template>
              <ul v-if="visitStats.length" class="stats-list"><li v-for="(stat, index) in visitStats" :key="stat.location"><span class="rank">{{ index + 1 }}</span><span class="location">{{ stat.location }}</span><span class="count">{{ stat.count }} 次</span></li></ul>
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
  --color-primary: #337ecc;
  --color-primary-light: #eaf2fa;
  --color-success: #67c23a;
  --color-warning: #e6a23c;
  --color-danger: #f56c6c;
  --color-text-primary: #2c3e50;
  --color-text-regular: #5a5e66;
  --color-text-secondary: #878d96;
  --border-color: #dcdfe6;
  --bg-color-page: #f5f7fa;
  --bg-color-card: #ffffff;
  --font-family-main: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
}
body { margin: 0; font-family: var(--font-family-main); background-color: var(--bg-color-page); color: var(--color-text-primary); }
.container { width: 90%; max-width: 1600px; margin: 0 auto; }

/* --- App Layout --- */
.app-container { display: flex; flex-direction: column; min-height: 100vh; }
.app-header { background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); color: white; padding: 2rem 0; text-align: center; }
.title-main { font-size: 2.25rem; font-weight: 600; margin: 0; letter-spacing: 1px; }
.title-sub { font-size: 1.1rem; font-weight: 300; opacity: 0.9; margin-top: 0.5rem; }
.main-content { padding: 2rem 0; flex-grow: 1; }
.main-grid { display: grid; grid-template-columns: 4fr 5fr; gap: 2rem; }
.app-footer { background-color: #2c3e50; color: var(--color-text-secondary); text-align: center; padding: 1.5rem 0; font-size: 0.875rem; }
.app-footer p { margin: 0.25rem 0; }
.eng-footer { font-size: 0.8rem; opacity: 0.7; }

/* --- Card & Component Styles --- */
.box-card { border: 1px solid var(--border-color); border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
.el-card__header { background-color: #fafbfd; }
.card-header { display: flex; align-items: center; gap: 0.75rem; font-size: 1.1rem; font-weight: 600; color: var(--color-text-primary); }
.card-description { font-size: 0.9rem; color: var(--color-text-regular); margin: 0 0 1.5rem 0; line-height: 1.6; }
.card-description code { background-color: var(--color-primary-light); color: var(--color-primary); padding: 2px 6px; border-radius: 4px; font-weight: 600; }
.button-group { margin-top: 1.5rem; display: flex; justify-content: space-between; gap: 1rem; }
.el-upload-dragger { padding: 2rem; }

/* --- Results & Stats --- */
.results-card, .placeholder-card { min-height: 526px; }
.placeholder-card .el-card__body { display: flex; align-items: center; justify-content: center; height: 100%; }
.risk-tag { padding: 5px 10px; border-radius: 12px; color: white; font-weight: bold; font-size: 0.85em; }
.risk-high { background-color: var(--color-danger); }
.risk-medium { background-color: var(--color-warning); }
.risk-low { background-color: var(--color-success); }

.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 2rem; }
.stats-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 1rem; }
.stats-list li { display: flex; align-items: center; font-size: 0.95rem; color: var(--color-text-regular); padding-bottom: 0.5rem; border-bottom: 1px solid #f0f2f5; }
.stats-list .rank { font-weight: bold; color: var(--color-text-secondary); width: 2em; text-align: center; }
.stats-list .location { flex-grow: 1; font-weight: 500; }
.stats-list .count { font-weight: bold; color: var(--color-primary); }

/* --- Responsive Design --- */
@media (max-width: 1200px) {
  .main-grid { grid-template-columns: 1fr; }
  .results-card, .placeholder-card { min-height: auto; }
}
@media (max-width: 768px) {
  .title-main { font-size: 1.75rem; }
  .title-sub { font-size: 1rem; }
  .button-group { flex-direction: column; }
  .button-group .el-button { width: 100%; margin-left: 0 !important; }
  .stats-grid { grid-template-columns: 1fr; }
}
</style>