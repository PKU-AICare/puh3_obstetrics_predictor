<script setup>
import { ref, computed, onMounted, nextTick, watch } from 'vue';
import axios from 'axios';
import * as XLSX from 'xlsx';
import { ElMessage, ElLoading } from 'element-plus';
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
    // Logos Alt Text
    logo1_alt: '国家产科专业医疗质量控制中心',
    logo2_alt: '国家妇产疾病临床医学研究中心',
    logo3_alt: '北京大学第三医院',
    // UI Text
    singlePatient: '单个患者预测',
    singlePatientDesc: '上传单个患者的 Excel 文件 (.xlsx) 进行风险评估。文件名将作为患者ID。',
    batchPatient: '批量预测',
    batchPatientDesc: '上传包含多个患者 Excel 文件的压缩包 (.zip / .rar / .7z) 进行批量预测。',
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
    langSwitch: 'EN',
    // Footer
    footerLine1: '国家产科专业医疗质量控制中心 | 国家妇产疾病临床医学研究中心 | 北京大学第三医院妇产科',
    footerLine2: 'National Centre for Healthcare Quality Management in Obstetrics | National Clinical Research Center for Obstetrics and Gynecology | Department of Obstetrics and Gynecology, Peking University Third Hospital',
    // Messages
    uploadSuccess: '文件上传成功。',
    predictionSuccess: '患者 {patientId} 的风险评估完成！',
    batchSuccess: '批量预测完成！您现在可以从下拉菜单中选择患者查看结果。',
    downloadingTemplate: '正在生成模板...',
    templateError: '模板下载失败，请稍后重试。',
    selectSingleFile: '请先选择一个患者的 Excel 文件。',
    selectBatchFile: '请先选择一个压缩包。',
    calcFailed: '计算失败: {detail}',
    exportSuccess: '结果已导出为 Excel 文件。',
    exportError: '导出失败: {detail}',
  },
  en: {
    title: 'Risk Assessment of Pregnancy-Related Diseases',
    // Logos Alt Text
    logo1_alt: 'National Centre for Healthcare Quality Management in Obstetrics',
    logo2_alt: 'National Clinical Research Center for Obstetrics and Gynecology',
    logo3_alt: 'Peking University Third Hospital',
    // UI Text
    singlePatient: 'Single Patient Prediction',
    singlePatientDesc: 'Upload a single patient\'s Excel file (.xlsx). The filename will be used as Patient ID.',
    batchPatient: 'Batch Prediction',
    batchPatientDesc: 'Upload an archive (.zip / .rar / .7z) containing multiple patient Excel files for batch prediction.',
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
    langSwitch: '中文',
    // Footer
    footerLine1: '国家产科专业医疗质量控制中心 | 国家妇产疾病临床医学研究中心 | 北京大学第三医院妇产科',
    footerLine2: 'National Centre for Healthcare Quality Management in Obstetrics | National Clinical Research Center for Obstetrics and Gynecology | Department of Obstetrics and Gynecology, Peking University Third Hospital',
    // Messages
    uploadSuccess: 'File uploaded successfully.',
    predictionSuccess: 'Risk assessment for patient {patientId} is complete!',
    batchSuccess: 'Batch prediction complete! You can now select a patient from the dropdown to view results.',
    downloadingTemplate: 'Generating template...',
    templateError: 'Failed to download template. Please try again later.',
    selectSingleFile: 'Please select a patient\'s Excel file first.',
    selectBatchFile: 'Please select an archive first.',
    calcFailed: 'Calculation failed: {detail}',
    exportSuccess: 'Results have been exported to an Excel file.',
    exportError: 'Export failed: {detail}',
  }
};

const lang = ref('zh');
const t = computed(() => locales[lang.value]);
const toggleLang = () => { lang.value = lang.value === 'zh' ? 'en' : 'zh'; };
watch(lang, (newLang) => {
  document.documentElement.lang = newLang === 'zh' ? 'zh-CN' : 'en';
  document.title = t.value.title;
});


// --- State Management ---
const singleFile = ref(null);
const batchFile = ref(null);
const singleUploadRef = ref(null);
const batchUploadRef = ref(null);
const isLoadingSingle = ref(false);
const isLoadingBatch = ref(false);
const allPatientResults = ref([]);
const selectedPatientId = ref(null);
const stats = ref({ total_visits: 0, total_predictions: 0, unique_countries_count: 0, visit_ranking_by_country: [], usage_ranking_by_country: [], });
const chartGridKey = ref(0);

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
  } catch (error) { console.error('Failed to fetch stats:', error); }
};

const downloadTemplate = async () => {
  const loading = ElLoading.service({ lock: true, text: t.value.downloadingTemplate, background: 'rgba(0, 0, 0, 0.8)' });
  try {
    const response = await axios.get(`${API_BASE_URL}/api/download-template`, { responseType: 'blob' });
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'Prediction_Template_Bilingual.xlsx');
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  } catch (error) {
    ElMessage.error(t.value.templateError);
  } finally {
    loading.close();
  }
};

const handleFileChange = (uploadFile, type, uploadRef) => {
  const file = uploadFile.raw;
  const fileExt = file.name.split('.').pop().toLowerCase();
  if (type === 'single') {
    if (fileExt !== 'xlsx') {
      ElMessage.error('Please upload a .xlsx file!');
      uploadRef.clearFiles();
      return;
    }
    singleFile.value = file;
  } else {
    if (fileExt !== 'zip' && fileExt !== 'rar' && fileExt !== '7z') {
      ElMessage.error('Please upload a valid archive!');
      uploadRef.clearFiles();
      return;
    }
    batchFile.value = file;
  }
};

const submitPrediction = async (type) => {
  const isSingle = type === 'single';
  const file = isSingle ? singleFile.value : batchFile.value;
  const endpoint = isSingle ? '/api/predict-single' : '/api/predict-batch';
  const loadingRef = isSingle ? isLoadingSingle : isLoadingBatch;
  const uploadRef = isSingle ? singleUploadRef : batchUploadRef;

  if (!file) {
    ElMessage.warning(isSingle ? t.value.selectSingleFile : t.value.selectBatchFile);
    return;
  }

  loadingRef.value = true;
  const formData = new FormData();
  formData.append('file', file);

  try {
    const { data } = await axios.post(`${API_BASE_URL}${endpoint}`, formData);
    const results = Array.isArray(data) ? data : [data];

    results.forEach(result => {
      const existingIndex = allPatientResults.value.findIndex(p => p.patient_id === result.patient_id);
      if (existingIndex > -1) {
        allPatientResults.value.splice(existingIndex, 1);
      }
      allPatientResults.value.unshift(result);
    });

    if (results.length > 0) {
      selectedPatientId.value = results[0].patient_id;
      ElMessage.success(isSingle ? t.value.predictionSuccess.replace('{patientId}', results[0].patient_id) : t.value.batchSuccess);
    }
    fetchStats();
  } catch (error) {
    const detail = error.response?.data?.detail || 'Unknown error.';
    ElMessage.error(t.value.calcFailed.replace('{detail}', detail));
  } finally {
    loadingRef.value = false;
    uploadRef.value?.clearFiles();
    if(isSingle) singleFile.value = null; else batchFile.value = null;
    nextTick(() => chartGridKey.value++);
  }
};

const exportResultsToExcel = () => {
  if (!hasResults.value) return;
  try {
    const dataToExport = allPatientResults.value.map(patient => {
      const row = { [t.value.patientID]: patient.patient_id };
      patient.predictions.forEach(pred => {
        const key = `${pred.disease_abbr} (${lang.value === 'zh' ? pred.disease_name_cn : pred.disease_name_en})`;
        row[key] = pred.probability;
      });
      return row;
    });

    const worksheet = XLSX.utils.json_to_sheet(dataToExport);
    // Format probability columns as percentages
    const range = XLSX.utils.decode_range(worksheet['!ref']);
    for (let C = 1; C <= range.e.c; ++C) {
        for (let R = 1; R <= range.e.r; ++R) {
            const cell_address = { c: C, r: R };
            const cell_ref = XLSX.utils.encode_cell(cell_address);
            if (worksheet[cell_ref] && typeof worksheet[cell_ref].v === 'number') {
                worksheet[cell_ref].z = '0.00%';
            }
        }
    }

    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Prediction Results");
    XLSX.writeFile(workbook, `Prediction_Results_${new Date().toISOString().slice(0, 10)}.xlsx`);
    ElMessage.success(t.value.exportSuccess);
  } catch(e) {
    ElMessage.error(t.value.exportError.replace('{detail}', e.message));
  }
};

const getChartOption = (diseaseName, probability) => {
  const probPercent = (probability * 100);
  const color = probPercent > 50 ? '#f56c6c' : (probPercent > 20 ? '#e6a23c' : '#67c23a');

  return {
    title: {
      text: diseaseName,
      left: 'center',
      bottom: '5%',
      textStyle: { fontSize: 14, fontWeight: 'normal', color: '#606266', overflow: 'truncate', width: 180 },
    },
    series: [{
      type: 'pie',
      radius: ['75%', '95%'],
      avoidLabelOverlap: false,
      silent: true,
      label: { show: true, position: 'center', formatter: `{c}%`, fontSize: 22, fontWeight: 'bold', color: color },
      data: [
        { value: probPercent.toFixed(1), name: 'Prob', itemStyle: { color } },
        { value: (100 - probPercent).toFixed(1), name: 'Rem', itemStyle: { color: '#EBEEF5' } }
      ],
    }]
  };
};

onMounted(() => {
    fetchStats();
    document.title = t.value.title;
});
</script>

<template>
  <div class="app-container">
    <header class="app-header">
      <div class="header-content container">
        <div class="logo-area">
          <img src="/logo1.png" :alt="t.logo1_alt" class="logo-placeholder" />
          <img src="/logo2.png" :alt="t.logo2_alt" class="logo-placeholder" />
          <img src="/logo3.png" :alt="t.logo3_alt" class="logo-placeholder" />
        </div>
        <h1 class="app-title">{{ t.title }}</h1>
        <el-button @click="toggleLang" type="primary" plain round>{{ t.langSwitch }}</el-button>
      </div>
    </header>

    <main class="container main-content">
      <div class="main-grid">
        <div class="control-panel">
          <el-card class="box-card" shadow="hover">
            <template #header><div class="card-header"><el-icon><User /></el-icon><span>{{ t.singlePatient }}</span></div></template>
            <p class="card-description">{{ t.singlePatientDesc }}</p>
            <el-upload ref="singleUploadRef" drag action="#" :limit="1" :auto-upload="false" @change="(file) => handleFileChange(file, 'single', singleUploadRef)" accept=".xlsx">
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">{{ t.uploadDrag }}<em>{{ t.uploadClick }}</em></div>
            </el-upload>
            <div class="button-group">
              <el-button @click="downloadTemplate" :icon="Download">{{ t.downloadTemplate }}</el-button>
              <el-button type="primary" @click="submitPrediction('single')" :loading="isLoadingSingle" :icon="Position">{{ t.startCalc }}</el-button>
            </div>
          </el-card>

          <el-card class="box-card" shadow="hover">
            <template #header><div class="card-header"><el-icon><Files /></el-icon><span>{{ t.batchPatient }}</span></div></template>
            <p class="card-description">{{ t.batchPatientDesc }}</p>
            <el-upload ref="batchUploadRef" drag action="#" :limit="1" :auto-upload="false" @change="(file) => handleFileChange(file, 'batch', batchUploadRef)" accept=".zip,.rar,.7z">
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">{{ t.uploadDrag }}<em>{{ t.uploadClick }}</em></div>
            </el-upload>
             <div class="button-group-single">
              <el-button type="success" @click="submitPrediction('batch')" :loading="isLoadingBatch" :icon="Promotion" class="full-width-btn">{{ t.processAndSubmit }}</el-button>
            </div>
          </el-card>
        </div>

        <div class="results-panel">
          <el-card class="box-card results-card" shadow="hover">
            <template #header>
              <div class="card-header-flex">
                <div class="card-header"><el-icon><DataAnalysis /></el-icon><span>{{ t.predictionResults }}</span></div>
                <div v-if="hasResults" class="header-controls">
                  <el-select-v2 v-model="selectedPatientId" :options="allPatientResults.map(p => ({ value: p.patient_id, label: p.patient_id }))" :placeholder="t.patientSelection" style="width: 200px; margin-right: 12px;" filterable />
                  <el-button @click="exportResultsToExcel" type="primary" :icon="Document">{{ t.downloadResults }}</el-button>
                </div>
              </div>
            </template>
            <div v-if="currentPatientData" class="charts-grid-wrapper">
              <div class="charts-grid" :key="`${selectedPatientId}-${chartGridKey}`">
                <div v-for="pred in currentPatientData.predictions" :key="pred.disease_abbr" class="chart-container">
                  <v-chart class="chart" :option="getChartOption(lang === 'zh' ? pred.disease_name_cn : pred.disease_name_en, pred.probability)" autoresize />
                </div>
              </div>
            </div>
            <el-empty v-else :description="t.noResults" class="full-height-empty" />
          </el-card>
        </div>
      </div>

      <el-card class="box-card stats-card" shadow="hover">
         <template #header><div class="card-header"><el-icon><TrendCharts /></el-icon><span>{{ t.stats }}</span></div></template>
         <div class="stats-overview">
            <div class="stat-item"><div class="stat-value">{{ stats.total_visits }}</div><div class="stat-label">{{ t.totalVisits }}</div></div>
            <div class="stat-item"><div class="stat-value">{{ stats.total_predictions }}</div><div class="stat-label">{{ t.totalPredictions }}</div></div>
            <div class="stat-item"><div class="stat-value">{{ stats.unique_countries_count }}</div><div class="stat-label">{{ t.uniqueCountries }}</div></div>
         </div>
         <div class="stats-rankings">
            <div class="ranking-list">
                <h3>{{ t.usageRanking }}</h3>
                <ul v-if="stats.usage_ranking_by_country.length > 0">
                    <li v-for="(stat, index) in stats.usage_ranking_by_country" :key="stat.location">
                        <span class="rank-badge">{{ index + 1 }}</span><span class="location">{{ stat.location }}</span><span class="count">{{ stat.count }}</span>
                    </li>
                </ul>
                <el-empty v-else :description="t.noData" :image-size="50" />
            </div>
             <div class="ranking-list">
                <h3>{{ t.visitRanking }}</h3>
                <ul v-if="stats.visit_ranking_by_country.length > 0">
                    <li v-for="(stat, index) in stats.visit_ranking_by_country" :key="stat.location">
                        <span class="rank-badge">{{ index + 1 }}</span><span class="location">{{ stat.location }}</span><span class="count">{{ stat.count }}</span>
                    </li>
                </ul>
                <el-empty v-else :description="t.noData" :image-size="50" />
            </div>
         </div>
      </el-card>
    </main>

    <footer class="app-footer">
      <p>{{t.footerLine1}}</p>
      <p class="eng-footer">{{t.footerLine2}}</p>
    </footer>
  </div>
</template>

<style>
:root {
  --color-primary: #409eff; --color-primary-light: #ecf5ff;
  --color-success: #67c23a; --color-warning: #e6a23c; --color-danger: #f56c6c;
  --color-text-primary: #303133; --color-text-regular: #606266; --color-text-secondary: #909399;
  --border-color: #dcdfe6; --bg-color-page: #f5f7fa; --bg-color-card: #ffffff;
  --font-family-main: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
  --card-border-radius: 12px;
}
body { margin: 0; font-family: var(--font-family-main); background-color: var(--bg-color-page); color: var(--color-text-primary); -webkit-font-smoothing: antialiased; }
.container { width: 95%; max-width: 1600px; margin: 0 auto; }
.app-container { display: flex; flex-direction: column; min-height: 100vh; }

.app-header { background: var(--bg-color-card); padding: 1rem 0; border-bottom: 1px solid #e4e7ed; box-shadow: 0 2px 8px rgba(0,0,0,0.05); z-index: 10; }
.header-content { display: flex; justify-content: space-between; align-items: center; }
.logo-area { display: flex; align-items: center; gap: 1rem; }
.logo-placeholder { height: 40px; width: auto; object-fit: contain; background-color: #f0f2f5; border-radius: 4px; padding: 4px;}
.app-title { font-size: 1.5rem; font-weight: 600; color: var(--color-text-primary); text-align: center; flex-grow: 1; }

.main-content { padding: 2rem 0; flex-grow: 1; }
.main-grid { display: grid; grid-template-columns: minmax(380px, 1fr) 2fr; gap: 2rem; align-items: flex-start;}

.app-footer { background-color: #303133; color: rgba(255,255,255,0.6); text-align: center; padding: 1.5rem 1rem; font-size: 0.8rem; line-height: 1.6; }
.app-footer p { margin: 0.25rem 0; }
.eng-footer { opacity: 0.8; }

.box-card { border: none; border-radius: var(--card-border-radius); }
.el-card { --el-card-padding: 24px; }
.el-card__header { border-bottom: 1px solid #e4e7ed; }
.card-header { display: flex; align-items: center; gap: 0.75rem; font-size: 1.1rem; font-weight: 600; color: var(--color-text-primary); }
.card-description { font-size: 0.9rem; color: var(--color-text-regular); margin: 0 0 1.5rem 0; line-height: 1.6; }
.button-group { margin-top: 1.5rem; display: flex; justify-content: space-between; gap: 1rem; }
.button-group-single { margin-top: 1.5rem; }
.full-width-btn { width: 100%; }
.el-upload-dragger { padding: 2rem; border-radius: 8px; }

.control-panel { display: flex; flex-direction: column; gap: 2rem; }
.results-card { display: flex; flex-direction: column; min-height: 650px; height: 100%;}
.results-card .el-card__body { flex-grow: 1; padding: 10px; display: flex; flex-direction: column;}
.full-height-empty { flex-grow: 1; }
.card-header-flex { display: flex; justify-content: space-between; align-items: center; }

.charts-grid-wrapper { overflow-y: auto; flex-grow: 1; margin: 0 -10px; padding: 0 10px;}
.charts-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; padding: 10px 0;}
.chart-container { display: flex; flex-direction: column; background-color: #fafcfe; border: 1px solid #f0f2f5; border-radius: 8px; padding: 10px 5px; height: 210px; transition: all 0.2s ease-in-out; }
.chart-container:hover { transform: translateY(-3px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
.chart { width: 100%; height: 100%; }

.stats-card { margin-top: 2rem; }
.stats-overview { display: grid; grid-template-columns: repeat(3, 1fr); text-align: center; padding: 1rem 0 2rem 0; border-bottom: 1px solid #e4e7ed; }
.stat-item .stat-value { font-size: 2.5rem; font-weight: bold; color: var(--color-primary); font-family: 'DIN-Alternate-Bold', 'Helvetica Neue', sans-serif;}
.stat-item .stat-label { font-size: 1rem; color: var(--color-text-secondary); margin-top: 0.5rem; }
.stats-rankings { display: grid; grid-template-columns: 1fr 1fr; gap: 2.5rem; padding-top: 2rem; }
.ranking-list h3 { font-size: 1.1rem; margin: 0 0 1rem 0; color: var(--color-text-primary); border-bottom: 2px solid var(--color-primary); padding-bottom: 8px; display: inline-block;}
.ranking-list ul { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 1rem; }
.ranking-list li { display: flex; align-items: center; font-size: 0.95rem; color: var(--color-text-regular); }
.rank-badge { background-color: #f0f2f5; color: var(--color-text-secondary); font-weight: bold; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: 1rem; flex-shrink: 0; }
.ranking-list li:nth-child(1) .rank-badge { background-color: #FFD700; color: white; }
.ranking-list li:nth-child(2) .rank-badge { background-color: #C0C0C0; color: white; }
.ranking-list li:nth-child(3) .rank-badge { background-color: #CD7F32; color: white; }
.ranking-list .location { flex-grow: 1; font-weight: 500; }
.ranking-list .count { font-weight: bold; color: var(--color-text-primary); }

@media (max-width: 1200px) {
  .main-grid { grid-template-columns: 1fr; }
  .results-panel { order: -1; }
  .app-title { font-size: 1.25rem; }
}
@media (max-width: 768px) {
  .header-content { flex-direction: column; gap: 1rem; align-items: center; }
  .logo-area { order: 2; }
  .app-title { order: 1; }
  .el-button { order: 3; }
  .stats-overview, .stats-rankings { grid-template-columns: 1fr; }
  .stats-overview { gap: 2rem; }
  .charts-grid { grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); }
  .chart-container { height: 190px; }
  .card-header-flex { flex-direction: column; align-items: flex-start; gap: 10px;}
  .header-controls { display: flex; flex-direction: column; align-items: stretch; width: 100%; gap: 10px; }
  .header-controls .el-select-v2 { margin-right: 0 !important; }
}
</style>