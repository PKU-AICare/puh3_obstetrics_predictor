<script setup>
import { ref, computed, onMounted, nextTick, watch } from 'vue'
import axios from 'axios'
import * as XLSX from 'xlsx'
import { ElMessage, ElLoading } from 'element-plus'
import {
  UploadFilled, DataAnalysis, TrendCharts, Download, Position, Document, Close, Upload
} from '@element-plus/icons-vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { PieChart } from 'echarts/charts'
import { TitleComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import VChart from 'vue-echarts'

// --- ECharts Setup ---
use([CanvasRenderer, PieChart, TitleComponent, TooltipComponent, LegendComponent])

// --- Configuration ---
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

// --- i18n Composable ---
const useI18n = () => {
  const locale = ref('zh')
  const messages = {
    zh: {
      title: '再次妊娠孕期疾病发生风险评估',
      subtitle: 'Assessment of Pregnancy-Related Disease Risks in Repeat Pregnancies',
      uploadTitle: '上传与预测',
      uploadDesc: '支持单个患者(.xlsx)或包含多个患者文件的压缩包(.zip, .rar, .7z)。文件名将作为患者ID。',
      downloadTemplate: '下载模板',
      startCalc: '开始计算',
      uploadDrag: '将文件拖到此处，或',
      uploadClick: '点击上传',
      predictionResults: '预测结果',
      patientID: '患者ID',
      noResults: '上传文件后，在此查看预测结果。',
      stats: '网站使用统计',
      totalVisits: '总访问次数',
      totalPredictions: '总预测人次',
      uniqueCountries: '覆盖国家数',
      usageRanking: '使用次数排行 (按国家)',
      visitRanking: '访问次数排行 (按国家)',
      noData: '暂无数据',
      downloadResults: '下载结果 (Excel)',
      patientSelection: '选择患者',
      langSwitch: 'EN',
      footerLine1: '国家产科专业医疗质量控制中心 | 国家妇产疾病临床医学研究中心 | 北京大学第三医院妇产科',
      footerLine2: 'National Centre for Healthcare Quality Management in Obstetrics | National Clinical Research Center for Obstetrics and Gynecology | Department of Obstetrics and Gynecology, Peking University Third Hospital',
      uploadSuccess: '文件上传成功。',
      predictionSuccess: '患者 {patientId} 的风险评估完成！',
      batchSuccess: '批量预测完成！您现在可以从下拉菜单中选择患者查看结果。',
      downloadingTemplate: '正在生成模板...',
      templateError: '模板下载失败，请稍后重试。',
      selectFile: '请先选择一个文件。',
      invalidFileType: '不支持的文件类型。请上传 .xlsx, .zip, .rar, 或 .7z 文件。',
      calcFailed: '计算失败: {detail}',
      exportSuccess: '结果已导出为 Excel 文件。',
      exportError: '导出失败: {detail}',
      fileSelected: '已选择文件:',
      removeFile: '移除文件',
      filePlaceholder: '待上传的文件将显示在此处',
    },
    en: {
      title: 'Assessment of Pregnancy-Related Disease Risks in Repeat Pregnancies',
      subtitle: '再次妊娠孕期疾病发生风险评估',
      uploadTitle: 'Upload & Predict',
      uploadDesc: 'Supports single patient (.xlsx) or an archive (.zip, .rar, .7z) with multiple patient files. Filename will be used as Patient ID.',
      downloadTemplate: 'Download Template',
      startCalc: 'Calculate',
      uploadDrag: 'Drag file here, or ',
      uploadClick: 'click to upload',
      predictionResults: 'Prediction Results',
      patientID: 'Patient ID',
      noResults: 'Upload a file to see prediction results here.',
      stats: 'Website Usage Statistics',
      totalVisits: 'Total Visits',
      totalPredictions: 'Total Predictions',
      countriesReached: 'Countries Reached',
      usageRanking: 'Usage Ranking (by Country)',
      visitRanking: 'Visit Ranking (by Country)',
      noData: 'No Data Available',
      downloadResults: 'Download Results (Excel)',
      patientSelection: 'Select Patient',
      langSwitch: '中文',
      footerLine1: '国家产科专业医疗质量控制中心 | 国家妇产疾病临床医学研究中心 | 北京大学第三医院妇产科',
      footerLine2: 'National Centre for Healthcare Quality Management in Obstetrics | National Clinical Research Center for Obstetrics and Gynecology | Department of Obstetrics and Gynecology, Peking University Third Hospital',
      uploadSuccess: 'File uploaded successfully.',
      predictionSuccess: 'Risk assessment for patient {patientId} is complete!',
      batchSuccess: 'Batch prediction complete! You can now select a patient from the dropdown to view results.',
      downloadingTemplate: 'Generating template...',
      templateError: 'Failed to download template. Please try again later.',
      selectFile: 'Please select a file first.',
      invalidFileType: 'Invalid file type. Please upload a .xlsx, .zip, .rar, or .7z file.',
      calcFailed: 'Calculation failed: {detail}',
      exportSuccess: 'Results have been exported to an Excel file.',
      exportError: 'Export failed: {detail}',
      fileSelected: 'Selected file:',
      removeFile: 'Remove file',
      filePlaceholder: 'The file to be uploaded will be displayed here',
    }
  }
  const t = computed(() => messages[locale.value])
  const toggleLang = () => { locale.value = locale.value === 'zh' ? 'en' : 'zh' }
  return { locale, t, toggleLang }
}

// --- API Composable ---
const useApi = () => {
  const logVisit = async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/log-visit`)
    } catch (error) { console.warn('Could not log visit:', error) }
  }
  const fetchStats = async () => {
    try {
      const { data } = await axios.get(`${API_BASE_URL}/api/stats`)
      return data
    } catch (error) {
      console.error('Failed to fetch stats:', error)
      return { total_visits: 0, total_predictions: 0, unique_countries_count: 0, visit_ranking_by_country: [], usage_ranking_by_country: [] }
    }
  }
  const downloadTemplate = async () => {
    const response = await axios.get(`${API_BASE_URL}/api/download-template`, { responseType: 'blob' })
    const url = window.URL.createObjectURL(new Blob([response.data]))
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', 'Prediction_Template_Bilingual.xlsx')
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.URL.revokeObjectURL(url)
  }
  const submitPrediction = async (file, endpoint) => {
    const formData = new FormData()
    formData.append('file', file)
    const { data } = await axios.post(`${API_BASE_URL}${endpoint}`, formData)
    return Array.isArray(data) ? data : [data]
  }
  return { logVisit, fetchStats, downloadTemplate, submitPrediction }
}

// --- Chart Composable (REVISED AND FIXED) ---
const useCharts = () => {
  const getChartOption = (probability) => {
    // Ensure probability is a valid number (0-100), defaulting to 0 if not.
    const probPercent = (typeof probability === 'number' && !isNaN(probability)) ? probability : 0;

    let color;
    // Define risk levels and corresponding colors based on percentage (0-100)
    if (probPercent >= 50) {
      color = '#f56c6c'; // Danger (Red)
    } else if (probPercent >= 20) {
      color = '#e6a23c'; // Warning (Orange)
    } else {
      color = '#67c23a'; // Success (Green)
    }

    return {
      series: [{
        type: 'pie',
        radius: ['70%', '90%'],
        center: ['50%', '50%'],
        avoidLabelOverlap: false,
        silent: true,
        label: {
          show: true,
          position: 'center',
          // DEFINITIVE FIX: Use a formatter function.
          // This completely decouples the displayed text from the data values.
          // The text will ALWAYS be the correct `probPercent`, while the data
          // values below are only used for visual representation of the ring.
          formatter: () => `${probPercent.toFixed(1)}%`,
          fontSize: 18,
          fontWeight: 'bold',
          color: color,
        },
        // Data for visual representation ONLY.
        data: [
          // This part determines the size of the colored slice.
          { value: probPercent, name: 'Probability', itemStyle: { color: color } },
          // This part determines the size of the grey "remaining" slice.
          { value: 100 - probPercent, name: 'Remaining', itemStyle: { color: '#f0f2f5' } }
        ]
      }]
    };
  }
  return { getChartOption };
}


// --- Main component setup ---
const { locale, t, toggleLang } = useI18n()
const { logVisit, fetchStats, downloadTemplate, submitPrediction } = useApi()
const { getChartOption } = useCharts()

// --- State ---
const isLoading = ref(false)
const uploadFile = ref(null)
const uploadRef = ref(null)
const allPatientResults = ref([])
const selectedPatientId = ref(null)
const stats = ref({
  total_visits: 0, total_predictions: 0, unique_countries_count: 0,
  visit_ranking_by_country: [], usage_ranking_by_country: []
})
const chartGridKey = ref(0) // To force re-render charts

// --- Computed ---
const hasResults = computed(() => allPatientResults.value.length > 0)
const currentPatientData = computed(() => {
  if (!selectedPatientId.value) return null
  return allPatientResults.value.find(p => p.patient_id === selectedPatientId.value)
})
const patientOptions = computed(() =>
  allPatientResults.value.map(p => ({ value: p.patient_id, label: p.patient_id }))
)

// --- Methods ---
const handleFileChange = (file) => {
  const rawFile = file.raw
  const fileExt = rawFile.name.split('.').pop().toLowerCase()
  if (!['xlsx', 'zip', 'rar', '7z'].includes(fileExt)) {
    ElMessage.error(t.value.invalidFileType)
    uploadRef.value.clearFiles()
    uploadFile.value = null
    return
  }
  uploadFile.value = rawFile
}

const removeSelectedFile = () => {
  uploadFile.value = null
  uploadRef.value.clearFiles()
}

const handleDownloadTemplate = async () => {
  const loading = ElLoading.service({ lock: true, text: t.value.downloadingTemplate, background: 'rgba(0, 0, 0, 0.8)' })
  try {
    await downloadTemplate()
  } catch (error) {
    ElMessage.error(t.value.templateError)
  } finally {
    loading.close()
  }
}

const handlePrediction = async () => {
  if (!uploadFile.value) {
    ElMessage.warning(t.value.selectFile)
    return
  }

  const loadingText = locale.value === 'zh' ? '正在进行风险评估...' : 'Performing risk assessment...'
  isLoading.value = true
  const loadingInstance = ElLoading.service({
    lock: true,
    text: loadingText,
    background: 'rgba(0, 0, 0, 0.8)'
  })

  const fileExt = uploadFile.value.name.split('.').pop().toLowerCase()
  const endpoint = fileExt === 'xlsx' ? '/api/predict-single' : '/api/predict-batch'

  try {
    const results = await submitPrediction(uploadFile.value, endpoint)

    results.forEach(result => {
      const existingIndex = allPatientResults.value.findIndex(p => p.patient_id === result.patient_id)
      if (existingIndex > -1) allPatientResults.value.splice(existingIndex, 1)
      allPatientResults.value.unshift(result)
    })

    if (results.length > 0) {
      selectedPatientId.value = results[0].patient_id
      const message = fileExt === 'xlsx' ? t.value.predictionSuccess.replace('{patientId}', results[0].patient_id) : t.value.batchSuccess
      ElMessage.success(message)
    }

    stats.value = await fetchStats()
  } catch (error) {
    const detail = error.response?.data?.detail || 'Unknown error.'
    ElMessage.error(t.value.calcFailed.replace('{detail}', detail))
  } finally {
    isLoading.value = false
    loadingInstance.close()
    removeSelectedFile()
    nextTick(() => chartGridKey.value++) // Force re-render of charts
  }
}

const exportResultsToExcel = () => {
  if (!hasResults.value) return
  try {
    const dataToExport = allPatientResults.value.flatMap(patient =>
      patient.predictions.map(pred => ({
        [t.value.patientID]: patient.patient_id,
        'Disease Abbreviation': pred.disease_abbr,
        '疾病名称 (CN)': pred.disease_name_cn,
        'Disease Name (EN)': pred.disease_name_en,
        // The value is already a percentage. Just format it to 2 decimal places.
        'Probability (%)': pred.probability.toFixed(2)
      }))
    )
    const worksheet = XLSX.utils.json_to_sheet(dataToExport)
    const workbook = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(workbook, worksheet, "Prediction Results")
    XLSX.writeFile(workbook, `Prediction_Results_${new Date().toISOString().slice(0, 10)}.xlsx`)
    ElMessage.success(t.value.exportSuccess)
  } catch (e) {
    ElMessage.error(t.value.exportError.replace('{detail}', e.message))
  }
}

// --- Watchers ---
watch(locale, (newLang) => {
  document.documentElement.lang = newLang === 'zh' ? 'zh-CN' : 'en'
  document.title = t.value.title
})

// --- Lifecycle ---
onMounted(async () => {
  document.title = t.value.title
  await logVisit()
  stats.value = await fetchStats()
})
</script>

<template>
  <div class="app-container">
    <header class="app-header">
      <div class="header-content container">
        <div class="logo-area">
          <img src="/logo1.png" alt="Logo 1" class="logo-image" />
          <img src="/logo2.png" alt="Logo 2" class="logo-image" />
          <img src="/logo3.png" alt="Logo 3" class="logo-image" />
        </div>
        <div class="title-area">
          <h1 class="app-title">{{ t.title }}</h1>
          <p class="app-subtitle">{{ t.subtitle }}</p>
        </div>
        <div class="header-actions">
          <el-button @click="toggleLang" type="primary" plain round size="default">
            {{ t.langSwitch }}
          </el-button>
        </div>
      </div>
    </header>

    <main class="container main-content">
      <div class="main-grid">
        <el-card class="control-panel-card" shadow="never">
          <template #header>
            <div class="card-header">
              <el-icon><UploadFilled /></el-icon><span>{{ t.uploadTitle }}</span>
            </div>
          </template>
          <div class="control-panel-body">
            <p class="card-description">{{ t.uploadDesc }}</p>

            <div class="upload-section">
              <el-upload
                ref="uploadRef"
                drag
                action="#"
                :limit="1"
                :auto-upload="false"
                :show-file-list="false"
                @change="handleFileChange"
                accept=".xlsx,.zip,.rar,.7z"
                class="upload-area"
              >
                  <el-icon class="el-icon--upload"><Upload /></el-icon>
                  <div class="el-upload__text">{{ t.uploadDrag }}<em>{{ t.uploadClick }}</em></div>
              </el-upload>

              <!-- Fixed height area for file info, prevents layout shift -->
              <div class="file-info-area">
                <div v-if="uploadFile" class="file-selected-box">
                    <span class="file-label">{{ t.fileSelected }}</span>
                    <span class="file-name" :title="uploadFile.name">{{ uploadFile.name }}</span>
                    <el-button :icon="Close" circle plain type="danger" size="small" @click="removeSelectedFile" :title="t.removeFile" class="remove-btn"/>
                </div>
                 <div v-else class="file-placeholder-box">
                    {{ t.filePlaceholder }}
                </div>
              </div>
            </div>

            <div class="button-group">
              <el-button @click="handleDownloadTemplate" :icon="Download" size="large">{{ t.downloadTemplate }}</el-button>
              <el-button type="primary" @click="handlePrediction" :disabled="!uploadFile" :icon="Position" size="large">{{ t.startCalc }}</el-button>
            </div>
          </div>
        </el-card>

        <el-card class="results-panel-card" shadow="never">
          <template #header>
            <div class="results-header">
              <div class="card-header">
                <el-icon><DataAnalysis /></el-icon><span>{{ t.predictionResults }}</span>
              </div>
              <div v-if="hasResults" class="header-controls">
                <el-select-v2 v-model="selectedPatientId" :options="patientOptions" :placeholder="t.patientSelection" style="width: 200px;" filterable size="default" />
                <el-button @click="exportResultsToExcel" type="success" plain :icon="Document" size="default">{{ t.downloadResults }}</el-button>
              </div>
            </div>
          </template>
          <div v-if="currentPatientData" class="results-content">
            <div class="charts-grid" :key="`${selectedPatientId}-${chartGridKey}`">
              <div v-for="pred in currentPatientData.predictions" :key="pred.disease_abbr" class="chart-container">
                <div class="disease-title" :title="locale === 'zh' ? pred.disease_name_cn : pred.disease_name_en">
                  {{ locale === 'zh' ? pred.disease_name_cn : pred.disease_name_en }}
                </div>
                <v-chart class="chart" :option="getChartOption(pred.probability)" autoresize />
              </div>
            </div>
          </div>
          <el-empty v-else :description="t.noResults" class="full-height-empty" />
        </el-card>
      </div>

      <el-card class="stats-card" shadow="never">
        <template #header>
          <div class="card-header">
            <el-icon><TrendCharts /></el-icon><span>{{ t.stats }}</span>
          </div>
        </template>
        <div class="stats-overview">
          <div class="stat-item">
            <div class="stat-value">{{ stats.total_visits.toLocaleString() }}</div>
            <div class="stat-label">{{ t.totalVisits }}</div>
          </div>
          <div class="stat-item">
            <div class="stat-value">{{ stats.total_predictions.toLocaleString() }}</div>
            <div class="stat-label">{{ t.totalPredictions }}</div>
          </div>
          <div class="stat-item">
            <div class="stat-value">{{ stats.unique_countries_count }}</div>
            <div class="stat-label">{{ t.uniqueCountries }}</div>
          </div>
        </div>
        <div class="stats-rankings">
          <div class="ranking-section">
            <h3>{{ t.usageRanking }}</h3>
            <el-scrollbar max-height="200px">
              <ul v-if="stats.usage_ranking_by_country.length > 0">
                <li v-for="(stat, index) in stats.usage_ranking_by_country" :key="stat.location">
                  <span class="rank-badge" :class="`rank-${index + 1}`">{{ index + 1 }}</span>
                  <span class="location">{{ stat.location }}</span>
                  <span class="count">{{ stat.count }}</span>
                </li>
              </ul>
              <el-empty v-else :description="t.noData" :image-size="50" />
            </el-scrollbar>
          </div>
          <div class="ranking-section">
            <h3>{{ t.visitRanking }}</h3>
             <el-scrollbar max-height="200px">
              <ul v-if="stats.visit_ranking_by_country.length > 0">
                <li v-for="(stat, index) in stats.visit_ranking_by_country" :key="stat.location">
                  <span class="rank-badge" :class="`rank-${index + 1}`">{{ index + 1 }}</span>
                  <span class="location">{{ stat.location }}</span>
                  <span class="count">{{ stat.count }}</span>
                </li>
              </ul>
              <el-empty v-else :description="t.noData" :image-size="50" />
            </el-scrollbar>
          </div>
        </div>
      </el-card>
    </main>

    <footer class="app-footer">
      <div class="footer-content">
        <p>{{ t.footerLine1 }}</p>
        <p class="footer-en">{{ t.footerLine2 }}</p>
      </div>
    </footer>
  </div>
</template>

<style>
/* --- Global Styles & Variables --- */
:root {
  --color-primary: #409EFF;
  --color-primary-light-9: #ecf5ff;
  --color-success: #67c23a;
  --color-warning: #e6a23c;
  --color-danger: #f56c6c;
  --color-text-primary: #303133;
  --color-text-regular: #606266;
  --color-text-secondary: #909399;
  --color-border: #dcdfe6;
  --color-border-light: #e4e7ed;
  --bg-color-page: #f5f7fa; /* Slightly cooler grey */
  --bg-color-card: #ffffff;
  --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
  --border-radius: 12px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;
  --font-family: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  font-family: var(--font-family);
  background-color: var(--bg-color-page);
  color: var(--color-text-regular);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.container {
  width: 95%;
  max-width: 1800px;
  margin: 0 auto;
  padding: 0 16px;
}

/* --- App Layout --- */
.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}
.main-content {
  flex-grow: 1; /* Pushes footer down */
  padding-top: var(--spacing-xl);
  padding-bottom: var(--spacing-xl);
}

/* --- General Components --- */
.el-card {
  border: 1px solid var(--color-border-light);
  border-radius: var(--border-radius);
  box-shadow: none;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.el-card__header {
  border-bottom: 1px solid var(--color-border-light);
  padding: var(--spacing-md) var(--spacing-lg);
  background-color: #fafcfe;
  flex-shrink: 0;
}
.card-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--color-text-primary);
}
.card-header .el-icon {
  font-size: 1.3rem;
  color: var(--color-primary);
}

/* --- Header --- */
.app-header {
  background: rgba(255, 255, 255, 0.8);
  padding: var(--spacing-md) 0;
  border-bottom: 1px solid var(--color-border-light);
  position: sticky;
  top: 0;
  z-index: 100;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}
.header-content { display: flex; justify-content: space-between; align-items: center; gap: var(--spacing-lg); }
.logo-area { display: flex; align-items: center; gap: var(--spacing-md); }
.logo-image { height: 48px; object-fit: contain; }
.title-area { text-align: left; flex-grow: 1; }
.app-title { font-size: 1.6rem; font-weight: 700; color: var(--color-text-primary); margin: 0; letter-spacing: 1px; }
.app-subtitle { font-size: 0.9rem; color: var(--color-text-secondary); margin: 4px 0 0; font-weight: 500; }
.header-actions { flex-shrink: 0; }

/* --- Main Grid --- */
.main-grid {
  display: grid;
  grid-template-columns: 420px 1fr;
  gap: var(--spacing-xl);
  align-items: stretch; /* Make cards same height */
  margin-bottom: var(--spacing-xl);
}

/* --- Control Panel (Left) --- */
.control-panel-card .el-card__body { padding: var(--spacing-lg); flex-grow: 1; display: flex; flex-direction: column; }
.card-description { font-size: 0.9rem; color: var(--color-text-regular); margin: 0 0 var(--spacing-lg); line-height: 1.6; }

.upload-section { flex-grow: 1; display: flex; flex-direction: column; }
.upload-area .el-upload { width: 100%; }
.upload-area .el-upload-dragger {
  width: 100%;
  height: 120px; /* Reduced height */
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  border-radius: var(--border-radius);
  border: 2px dashed var(--color-border);
  transition: all 0.2s ease;
}
.upload-area .el-upload-dragger:hover { border-color: var(--color-primary); background-color: var(--color-primary-light-9); }
.el-upload__text { font-size: 1rem; }
.el-icon--upload { font-size: 48px; color: var(--color-text-secondary); margin-bottom: 10px; }

/* Critical for preventing layout shift */
.file-info-area {
  min-height: 65px; /* Fixed height to prevent layout shift */
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: var(--spacing-md);
  width: 100%;
}
.file-placeholder-box, .file-selected-box {
  width: 100%;
  padding: 12px var(--spacing-md);
  border: 1px solid var(--color-border-light);
  background-color: var(--bg-color-page);
  border-radius: 8px;
  font-size: 0.9rem;
  color: var(--color-text-secondary);
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
}
.file-selected-box {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: var(--spacing-sm);
  background-color: var(--color-primary-light-9);
  border-color: var(--color-primary);
  color: var(--color-text-primary);
}
.file-label { font-weight: 600; flex-shrink: 0; }
.file-name {
  flex-grow: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  text-align: left;
}
.remove-btn { margin-left: auto; flex-shrink: 0; }

.button-group { display: flex; gap: var(--spacing-md); margin-top: auto; padding-top: var(--spacing-lg); }
.button-group .el-button { flex-grow: 1; }

/* --- Results Panel (Right) - OPTIMIZED --- */
.results-panel-card .el-card__body {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  padding: 0;
}
.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}
.header-controls {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}
.results-content {
  flex-grow: 1;
  padding: var(--spacing-md);
  overflow-y: auto;
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(7, 1fr); /* Enforce a 7-column layout with equal width */
  gap: 12px;
}

.chart-container {
  background-color: var(--bg-color-page);
  border: 1px solid var(--color-border-light);
  border-radius: var(--border-radius);
  padding: var(--spacing-sm);
  height: 160px;
  display: flex;
  flex-direction: column;
  transition: all 0.2s ease;
  min-width: 0; /* KEY FIX: Allows grid item to shrink and content to truncate */
  overflow: hidden; /* Ensures no content spills out */
}

.chart-container:hover {
  transform: translateY(-3px);
  box-shadow: var(--shadow);
  border-color: var(--color-primary);
}

.disease-title {
  flex-shrink: 0;
  font-size: 0.9rem;
  font-weight: 500;
  text-align: center;
  padding: var(--spacing-sm);
  color: var(--color-text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.chart {
  flex-grow: 1;
  width: 100%;
  min-height: 0;
}

.full-height-empty {
  flex-grow: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--spacing-lg);
}


/* --- Statistics Card --- */
.stats-card .el-card__body { padding: var(--spacing-lg); }
.stats-overview { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--spacing-lg); text-align: center; padding-bottom: var(--spacing-lg); margin-bottom: var(--spacing-lg); border-bottom: 1px solid var(--color-border-light); }
.stat-item .stat-value { font-size: 2.5rem; font-weight: 700; color: var(--color-primary); line-height: 1.2; }
.stat-item .stat-label { font-size: 1rem; color: var(--color-text-secondary); margin-top: var(--spacing-sm); }
.stats-rankings { display: grid; grid-template-columns: 1fr 1fr; gap: var(--spacing-xl); }
.ranking-section h3 { font-size: 1.2rem; margin: 0 0 var(--spacing-md); color: var(--color-text-primary); }
.ranking-section ul { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: var(--spacing-sm); }
.ranking-section li { display: flex; align-items: center; font-size: 0.9rem; padding: var(--spacing-sm) var(--spacing-md); background-color: var(--bg-color-page); border-radius: 6px; }
.rank-badge { color: var(--color-text-secondary); font-weight: 600; border-radius: 4px; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: var(--spacing-md); background-color: #f4f4f5; }
.rank-badge.rank-1 { background-color: #fef08a; color: #a16207; }
.rank-badge.rank-2 { background-color: #e5e7eb; color: #4b5563; }
.rank-badge.rank-3 { background-color: #fde68a; color: #b45309; }
.ranking-section .location { flex-grow: 1; font-weight: 500; }
.ranking-section .count { font-weight: 600; color: var(--color-primary); }
.ranking-section .el-scrollbar { border: 1px solid var(--color-border-light); border-radius: 8px; padding: var(--spacing-sm); }

/* --- Footer --- */
.app-footer {
  background: #303133;
  color: #e5e7eb;
  text-align: center;
  padding: var(--spacing-lg) 0;
  margin-top: auto; /* Important for flex layout */
  flex-shrink: 0;
}
.footer-content p { margin: 6px 0; }
.footer-en { color: #909399; font-size: 0.8rem; }

/* --- Responsive --- */
@media (max-width: 1200px) {
  .main-grid { grid-template-columns: 1fr; align-items: stretch; }
  .control-panel-card, .results-panel-card { min-height: auto; }
}
@media (max-width: 768px) {
  .header-content { flex-direction: column; text-align: center; gap: var(--spacing-md); }
  .logo-image { height: 40px; }
  .app-title { font-size: 1.3rem; }
  .title-area { text-align: center; }
  .stats-overview, .stats-rankings { grid-template-columns: 1fr; }
  .stats-rankings { gap: var(--spacing-lg); }
  /* Make charts grid responsive on smaller screens */
  .charts-grid { grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); }
  .chart-container { height: 160px; }
}
</style>