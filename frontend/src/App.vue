<script setup>
import { ref, computed, onMounted, nextTick } from 'vue';
import axios from 'axios';
import * as XLSX from 'xlsx';
import { ElMessage, ElLoading, ElMessageBox } from 'element-plus';
import { use } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { PieChart } from 'echarts/charts';
import { TitleComponent, TooltipComponent, LegendComponent, GridComponent } from 'echarts/components';
import VChart from 'vue-echarts';

// --- ECharts Setup ---
use([CanvasRenderer, PieChart, TitleComponent, TooltipComponent, LegendComponent, GridComponent]);

// --- Configuration ---
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// --- i18n Translations ---
const locales = {
  zh: {
    title: '再次妊娠孕期疾病发生风险评估',
    subtitle: 'Assessment of Pregnancy-Related Disease Risks in Repeat Pregnancies',
    singlePatient: '单个患者预测',
    singlePatientDesc: '上传单个患者的 Excel 文件 (.xlsx) 进行风险评估。文件名将作为患者ID。',
    batchPatient: '批量预测',
    batchPatientDesc: '上传包含多个患者 Excel 文件的 ZIP 压缩包 (.zip) 进行批量预测。',
    downloadTemplate: '下载模板',
    startCalc: '开始计算',
    uploadDrag: '将文件拖到此处，或',
    uploadClick: '点击上传',
    processAndSubmit: '处理并提交',
    predictionResults: '预测结果',
    patientID: '患者ID',
    noResults: '请先上传文件以查看预测结果。',
    stats: '网站使用统计',
    totalVisits: '总访问次数',
    totalPredictions: '总预测次数',
    uniqueCountries: '覆盖国家数',
    usageRanking: '使用次数排行 (按国家)',
    visitRanking: '访问次数排行 (按国家)',
    noData: '暂无数据',
    downloadResults: '下载结果 (Excel)',
    patientSelection: '选择患者',
    allPatients: '所有患者',
    langSwitch: 'EN',
    disease: '疾病',
    probability: '预测概率',
    error: '错误',
    success: '成功',
    uploadSuccess: '文件上传成功',
    predictionSuccess: '患者 {patientId} 的风险评估完成！',
    batchSuccess: '批量预测完成！您现在可以从下拉菜单中选择患者查看结果。',
    downloadingTemplate: '正在生成模板...',
    templateError: '模板下载失败，请稍后重试。',
    selectSingleFile: '请先选择一个患者的 Excel 文件。',
    selectBatchFile: '请先选择一个 ZIP 压缩包。',
    calcFailed: '计算失败: {detail}',
    exportSuccess: '结果已导出为 Excel 文件。',
    exportError: '导出失败: {detail}',
  },
  en: {
    title: 'Assessment of Pregnancy-Related Disease Risks in Repeat Pregnancies',
    subtitle: 'Based on data from the first pregnancy',
    singlePatient: 'Single Patient Prediction',
    singlePatientDesc: 'Upload a single patient\'s Excel file (.xlsx) for risk assessment. The filename will be used as the Patient ID.',
    batchPatient: 'Batch Prediction',
    batchPatientDesc: 'Upload a ZIP archive (.zip) containing multiple patient Excel files for batch prediction.',
    downloadTemplate: 'Download Template',
    startCalc: 'Calculate',
    uploadDrag: 'Drag file here, or ',
    uploadClick: 'click to upload',
    processAndSubmit: 'Process & Submit',
    predictionResults: 'Prediction Results',
    patientID: 'Patient ID',
    noResults: 'Please upload a file to see prediction results.',
    stats: 'Website Usage Statistics',
    totalVisits: 'Total Visits',
    totalPredictions: 'Total Predictions',
    uniqueCountries: 'Countries Reached',
    usageRanking: 'Usage Ranking (by Country)',
    visitRanking: 'Visit Ranking (by Country)',
    noData: 'No Data Available',
    downloadResults: 'Download Results (Excel)',
    patientSelection: 'Select Patient',
    allPatients: 'All Patients',
    langSwitch: '中文',
    disease: 'Disease',
    probability: 'Probability',
    error: 'Error',
    success: 'Success',
    uploadSuccess: 'File uploaded successfully',
    predictionSuccess: 'Risk assessment for patient {patientId} is complete!',
    batchSuccess: 'Batch prediction complete! You can now select a patient from the dropdown to view results.',
    downloadingTemplate: 'Generating template...',
    templateError: 'Failed to download template. Please try again later.',
    selectSingleFile: 'Please select a patient\'s Excel file first.',
    selectBatchFile: 'Please select a ZIP archive first.',
    calcFailed: 'Calculation failed: {detail}',
    exportSuccess: 'Results have been exported to an Excel file.',
    exportError: 'Export failed: {detail}',
  }
};

const lang = ref('zh');
const t = computed(() => locales[lang.value]);
const toggleLang = () => { lang.value = lang.value === 'zh' ? 'en' : 'zh'; };

// --- State Management ---
const singleFile = ref(null);
const batchFile = ref(null);
const singleUploadRef = ref(null);
const batchUploadRef = ref(null);

const isLoadingSingle = ref(false);
const isLoadingBatch = ref(false);

const allPatientResults = ref([]);
const selectedPatientId = ref(null);

const stats = ref({
  total_visits: 0,
  total_predictions: 0,
  unique_countries_count: 0,
  visit_ranking_by_country: [],
  usage_ranking_by_country: [],
});

const chartGridKey = ref(0); // Key to force re-render of charts

// --- Computed Properties ---
const hasResults = computed(() => allPatientResults.value.length > 0);
const currentPatientData = computed(() => {
  if (!selectedPatientId.value) return null;
  return allPatientResults.value.find(p => p.patient_id === selectedPatientId.value);
});

// --- API & Helper Functions ---
const fetchStats = async () => {
  try {
    const { data } = await axios.get(`${API_BASE_URL}/api/stats`);
    stats.value = data;
  } catch (error) {
    console.error('Failed to fetch stats:', error);
  }
};

const downloadTemplate = async () => {
  const loading = ElLoading.service({ lock: true, text: t.value.downloadingTemplate, background: 'rgba(0, 0, 0, 0.7)' });
  try {
    const response = await axios.get(`${API_BASE_URL}/api/download-template`, { responseType: 'blob' });
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    const disposition = response.headers['content-disposition'];
    const filename = disposition?.split('filename=')[1]?.replace(/"/g, '') || 'prediction_template.xlsx';
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  } catch (error) {
    ElMessage.error(t.value.templateError);
    console.error('Template download error:', error);
  } finally {
    loading.close();
  }
};

const handleFileChange = (uploadFile, type) => {
  const file = uploadFile.raw;
  if (type === 'single') {
    if (!file.name.toLowerCase().endsWith('.xlsx')) {
      ElMessage.error('Please upload a .xlsx file!');
      singleUploadRef.value?.clearFiles();
      return;
    }
    singleFile.value = file;
  } else {
    if (!file.name.toLowerCase().endsWith('.zip')) {
      ElMessage.error('Please upload a .zip file!');
      batchUploadRef.value?.clearFiles();
      return;
    }
    batchFile.value = file;
  }
};

const submitSinglePrediction = async () => {
  if (!singleFile.value) {
    ElMessage.warning(t.value.selectSingleFile);
    return;
  }
  isLoadingSingle.value = true;
  const formData = new FormData();
  formData.append('file', singleFile.value);

  try {
    const { data: result } = await axios.post(`${API_BASE_URL}/api/predict-single`, formData);
    const existingIndex = allPatientResults.value.findIndex(p => p.patient_id === result.patient_id);
    if (existingIndex > -1) {
      allPatientResults.value[existingIndex] = result;
    } else {
      allPatientResults.value.push(result);
    }
    selectedPatientId.value = result.patient_id;
    ElMessage.success(t.value.predictionSuccess.replace('{patientId}', result.patient_id));
    fetchStats();
  } catch (error) {
    const detail = error.response?.data?.detail || 'Unknown error.';
    ElMessage.error(t.value.calcFailed.replace('{detail}', detail));
  } finally {
    isLoadingSingle.value = false;
    singleUploadRef.value?.clearFiles();
    singleFile.value = null;
    nextTick(() => chartGridKey.value++);
  }
};

const submitBatchPrediction = async () => {
  if (!batchFile.value) {
    ElMessage.warning(t.value.selectBatchFile);
    return;
  }
  isLoadingBatch.value = true;
  const formData = new FormData();
  formData.append('file', batchFile.value);

  try {
    const { data: results } = await axios.post(`${API_BASE_URL}/api/predict-batch`, formData);
    results.forEach(result => {
      const existingIndex = allPatientResults.value.findIndex(p => p.patient_id === result.patient_id);
      if (existingIndex > -1) {
        allPatientResults.value[existingIndex] = result;
      } else {
        allPatientResults.value.push(result);
      }
    });
    if (results.length > 0) {
      selectedPatientId.value = results[0].patient_id;
    }
    ElMessage.success(t.value.batchSuccess);
    fetchStats();
  } catch (error) {
    const detail = error.response?.data?.detail || 'Unknown error.';
    ElMessage.error(t.value.calcFailed.replace('{detail}', detail));
  } finally {
    isLoadingBatch.value = false;
    batchUploadRef.value?.clearFiles();
    batchFile.value = null;
    nextTick(() => chartGridKey.value++);
  }
};

const exportResultsToExcel = () => {
  try {
    const dataToExport = allPatientResults.value.map(patient => {
      const row = { 'Patient ID': patient.patient_id };
      patient.predictions.forEach(pred => {
        const key = `${pred.disease_abbr} (${lang.value === 'zh' ? pred.disease_name_cn : pred.disease_name_en})`;
        row[key] = (pred.probability * 100).toFixed(2) + '%';
      });
      return row;
    });

    const worksheet = XLSX.utils.json_to_sheet(dataToExport);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Prediction Results");
    XLSX.writeFile(workbook, `Prediction_Results_${new Date().toISOString().slice(0, 10)}.xlsx`);
    ElMessage.success(t.value.exportSuccess);
  } catch(e) {
    ElMessage.error(t.value.exportError.replace('{detail}', e.message));
  }
};

// --- Charting ---
const getChartOption = (diseaseName, probability) => {
  const probPercent = (probability * 100).toFixed(2);
  const color = probability > 0.5 ? '#f56c6c' : (probability > 0.2 ? '#e6a23c' : '#67c23a');

  return {
    title: {
      text: diseaseName,
      left: 'center',
      bottom: '5%',
      textStyle: {
        fontSize: 14,
        fontWeight: 'normal',
        color: '#606266',
      },
    },
    series: [
      {
        type: 'pie',
        radius: ['70%', '90%'],
        avoidLabelOverlap: false,
        silent: true,
        label: {
          show: true,
          position: 'center',
          formatter: `{c}%`,
          fontSize: 20,
          fontWeight: 'bold',
          color: color
        },
        data: [
          { value: probPercent, name: 'Probability', itemStyle: { color: color } },
          { value: 100 - probPercent, name: 'Remainder', itemStyle: { color: '#f0f2f5' } }
        ],
        emphasis: {
            label: {
                show: true,
                fontSize: 22
            }
        },
      }
    ]
  };
};

onMounted(fetchStats);
</script>

<template>
  <div class="app-container">
    <header class="app-header">
      <div class="header-content">
        <div class="logo-area">
          <el-icon :size="40" color="#409EFF"><School /></el-icon>
          <div>
            <h1 class="title-main">{{ t.title }}</h1>
            <p class="title-sub">{{ t.subtitle }}</p>
          </div>
        </div>
        <el-button @click="toggleLang" type="primary" plain round>{{ t.langSwitch }}</el-button>
      </div>
    </header>

    <main class="container main-content">
      <div class="main-grid">
        <!-- Left Panel: Controls -->
        <div class="control-panel">
          <el-card class="box-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon><User /></el-icon><span>{{ t.singlePatient }}</span>
              </div>
            </template>
            <p class="card-description">{{ t.singlePatientDesc }}</p>
            <el-upload ref="singleUploadRef" drag action="#" :limit="1" :auto-upload="false" @change="(file) => handleFileChange(file, 'single')" accept=".xlsx">
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">{{ t.uploadDrag }}<em>{{ t.uploadClick }}</em></div>
            </el-upload>
            <div class="button-group">
              <el-button @click="downloadTemplate" :icon="Download">{{ t.downloadTemplate }}</el-button>
              <el-button type="primary" @click="submitSinglePrediction" :loading="isLoadingSingle" :icon="Position">{{ t.startCalc }}</el-button>
            </div>
          </el-card>

          <el-card class="box-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon><Files /></el-icon><span>{{ t.batchPatient }}</span>
              </div>
            </template>
            <p class="card-description">{{ t.batchPatientDesc }}</p>
             <el-upload ref="batchUploadRef" drag action="#" :limit="1" :auto-upload="false" @change="(file) => handleFileChange(file, 'batch')" accept=".zip">
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">{{ t.uploadDrag }}<em>{{ t.uploadClick }}</em></div>
            </el-upload>
            <div class="button-group">
              <el-button type="success" @click="submitBatchPrediction" :loading="isLoadingBatch" :icon="Promotion">{{ t.processAndSubmit }}</el-button>
            </div>
          </el-card>
        </div>

        <!-- Right Panel: Results & Stats -->
        <div class="results-panel">
          <el-card class="box-card results-card" shadow="hover">
            <template #header>
              <div class="card-header-flex">
                <div class="card-header">
                  <el-icon><DataAnalysis /></el-icon><span>{{ t.predictionResults }}</span>
                </div>
                <div v-if="hasResults" class="header-controls">
                    <el-select v-model="selectedPatientId" :placeholder="t.patientSelection" size="small" style="width: 180px; margin-right: 10px;">
                      <el-option v-for="patient in allPatientResults" :key="patient.patient_id" :label="patient.patient_id" :value="patient.patient_id" />
                    </el-select>
                    <el-button @click="exportResultsToExcel" type="primary" size="small" :icon="Document">{{ t.downloadResults }}</el-button>
                </div>
              </div>
            </template>

            <div v-if="currentPatientData" class="charts-grid" :key="`${selectedPatientId}-${chartGridKey}`">
              <div v-for="pred in currentPatientData.predictions" :key="pred.disease_abbr" class="chart-container">
                <v-chart class="chart" :option="getChartOption(lang === 'zh' ? pred.disease_name_cn : pred.disease_name_en, pred.probability)" autoresize />
              </div>
            </div>

            <el-empty v-else :description="t.noResults" />
          </el-card>
        </div>
      </div>

      <!-- Stats Section -->
      <el-card class="box-card stats-card" shadow="hover">
         <template #header>
            <div class="card-header"><el-icon><TrendCharts /></el-icon><span>{{ t.stats }}</span></div>
         </template>
         <div class="stats-overview">
            <div class="stat-item">
                <div class="stat-value">{{ stats.total_visits }}</div>
                <div class="stat-label">{{ t.totalVisits }}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ stats.total_predictions }}</div>
                <div class="stat-label">{{ t.totalPredictions }}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ stats.unique_countries_count }}</div>
                <div class="stat-label">{{ t.uniqueCountries }}</div>
            </div>
         </div>
         <div class="stats-rankings">
            <div class="ranking-list">
                <h3>{{ t.usageRanking }}</h3>
                <ul v-if="stats.usage_ranking_by_country.length > 0">
                    <li v-for="(stat, index) in stats.usage_ranking_by_country" :key="stat.location">
                        <span class="rank-badge">{{ index + 1 }}</span>
                        <span class="location">{{ stat.location }}</span>
                        <span class="count">{{ stat.count }}</span>
                    </li>
                </ul>
                <el-empty v-else :description="t.noData" :image-size="50" />
            </div>
            <div class="ranking-list">
                <h3>{{ t.visitRanking }}</h3>
                <ul v-if="stats.visit_ranking_by_country.length > 0">
                    <li v-for="(stat, index) in stats.visit_ranking_by_country" :key="stat.location">
                        <span class="rank-badge">{{ index + 1 }}</span>
                        <span class="location">{{ stat.location }}</span>
                        <span class="count">{{ stat.count }}</span>
                    </li>
                </ul>
                <el-empty v-else :description="t.noData" :image-size="50" />
            </div>
         </div>
      </el-card>

    </main>

    <footer class="app-footer">
      <p>国家妇产疾病临床医学研究中心 · 北京大学第三医院妇产科生殖医学中心</p>
      <p class="eng-footer">National Clinical Research Center for Obstetrics and Gynecology, Department of Obstetrics and Gynecology, Peking University Third Hospital</p>
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
  --bg-color-page: #f5f7fa;
  --bg-color-card: #ffffff;
  --font-family-main: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
  --card-border-radius: 12px;
  --card-shadow: 0 8px 16px rgba(0,0,0,0.08);
}
body { margin: 0; font-family: var(--font-family-main); background-color: var(--bg-color-page); color: var(--color-text-primary); -webkit-font-smoothing: antialiased; }
.container { width: 95%; max-width: 1800px; margin: 0 auto; }

/* --- App Layout --- */
.app-container { display: flex; flex-direction: column; min-height: 100vh; }
.app-header { background: var(--bg-color-card); padding: 1.5rem 0; border-bottom: 1px solid var(--border-color); }
.header-content { display: flex; justify-content: space-between; align-items: center; width: 95%; max-width: 1800px; margin: 0 auto;}
.logo-area { display: flex; align-items: center; gap: 1rem; }
.title-main { font-size: 1.75rem; font-weight: 600; margin: 0; color: var(--color-text-primary); }
.title-sub { font-size: 0.9rem; font-weight: 400; color: var(--color-text-secondary); margin: 0.25rem 0 0 0; }
.main-content { padding: 2rem 0; flex-grow: 1; }
.main-grid { display: grid; grid-template-columns: minmax(400px, 1fr) 2fr; gap: 2rem; }
.app-footer { background-color: #303133; color: var(--color-text-secondary); text-align: center; padding: 2rem 0; font-size: 0.875rem; margin-top: 2rem; }
.app-footer p { margin: 0.25rem 0; }
.eng-footer { font-size: 0.8rem; opacity: 0.7; }

/* --- Card & Component Styles --- */
.box-card { border: none; border-radius: var(--card-border-radius); transition: all 0.3s ease; }
.box-card.el-card { --el-card-padding: 24px; }
.el-card__header { border-bottom: 1px solid #e4e7ed; }
.card-header { display: flex; align-items: center; gap: 0.75rem; font-size: 1.1rem; font-weight: 600; color: var(--color-text-primary); }
.card-description { font-size: 0.9rem; color: var(--color-text-regular); margin: 0 0 1.5rem 0; line-height: 1.6; }
.button-group { margin-top: 1.5rem; display: flex; justify-content: space-between; gap: 1rem; }
.el-upload-dragger { padding: 2rem; border-radius: 8px; }

/* --- Panels --- */
.control-panel { display: flex; flex-direction: column; gap: 2rem; }
.results-panel { min-height: 600px; }
.results-card { display: flex; flex-direction: column; height: 100%;}
.results-card .el-card__body { flex-grow: 1; }

.card-header-flex { display: flex; justify-content: space-between; align-items: center; }
.header-controls { display: flex; align-items: center; }

/* --- Charts Grid --- */
.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
  padding: 1rem;
  height: 100%;
  overflow-y: auto;
}
.chart-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-color: #fafafa;
  border-radius: 8px;
  padding: 10px;
  height: 220px;
}
.chart {
  width: 100%;
  height: 100%;
}

/* --- Stats Section --- */
.stats-card { margin-top: 2rem; }
.stats-overview { display: flex; justify-content: space-around; text-align: center; padding: 1rem 0 2rem 0; border-bottom: 1px solid #e4e7ed; }
.stat-item .stat-value { font-size: 2.5rem; font-weight: bold; color: var(--color-primary); }
.stat-item .stat-label { font-size: 1rem; color: var(--color-text-secondary); margin-top: 0.5rem; }
.stats-rankings { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; padding-top: 2rem; }
.ranking-list h3 { font-size: 1.1rem; margin: 0 0 1rem 0; color: var(--color-text-primary); }
.ranking-list ul { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 1rem; }
.ranking-list li { display: flex; align-items: center; font-size: 0.95rem; color: var(--color-text-regular); }
.rank-badge {
    background-color: var(--color-primary-light);
    color: var(--color-primary);
    font-weight: bold;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-right: 1rem;
}
.ranking-list li:nth-child(1) .rank-badge { background-color: #f56c6c; color: white; }
.ranking-list li:nth-child(2) .rank-badge { background-color: #e6a23c; color: white; }
.ranking-list li:nth-child(3) .rank-badge { background-color: #67c23a; color: white; }

.ranking-list .location { flex-grow: 1; font-weight: 500; }
.ranking-list .count { font-weight: bold; color: var(--color-text-primary); }

/* --- Responsive Design --- */
@media (max-width: 1200px) {
  .main-grid { grid-template-columns: 1fr; }
  .results-panel { order: -1; }
}
@media (max-width: 768px) {
  .header-content { flex-direction: column; gap: 1rem; }
  .title-main { font-size: 1.5rem; }
  .stats-overview, .stats-rankings { grid-template-columns: 1fr; flex-direction: column; gap: 2rem;}
  .charts-grid { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); }
  .chart-container { height: 180px; }
}
</style>