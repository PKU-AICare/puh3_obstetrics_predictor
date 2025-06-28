<script setup>
import { ref, computed, onMounted, nextTick, watch } from 'vue'
import axios from 'axios'
import * as XLSX from 'xlsx'
import { ElMessage, ElLoading } from 'element-plus'
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
      uniqueCountries: 'Countries Reached',
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

// --- Chart Composable ---
const useCharts = () => {
  const getChartOption = (diseaseName, probability) => {
    const probPercent = probability * 100
    const color = probPercent >= 50 ? '#f56c6c' : probPercent >= 20 ? '#e6a23c' : '#67c23a'

    return {
      series: [{
        type: 'pie',
        radius: ['70%', '90%'],
        center: ['50%', '50%'],
        avoidLabelOverlap: false,
        silent: true, // Disable all mouse events on the chart as requested
        label: {
          show: true,
          position: 'center',
          formatter: `{c}%`,
          fontSize: 18,
          fontWeight: 'bold',
          color: color,
        },
        data: [
          { value: probPercent.toFixed(1), name: 'Probability', itemStyle: { color: color } },
          { value: (100 - probPercent).toFixed(1), name: 'Remaining', itemStyle: { color: '#f0f2f5' } }
        ]
      }]
    }
  }
  return { getChartOption }
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

const handleDownloadTemplate = async () => {
  const loading = ElLoading.service({ lock: true, text: t.value.downloadingTemplate, background: 'rgba(0, 0, 0, 0.8)' })
  try {
    await downloadTemplate()
    ElMessage.success(t.value.templateSuccess || 'Template downloaded successfully!')
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

  isLoading.value = true
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

    // Refresh stats after prediction
    stats.value = await fetchStats()
  } catch (error) {
    const detail = error.response?.data?.detail || 'Unknown error.'
    ElMessage.error(t.value.calcFailed.replace('{detail}', detail))
  } finally {
    isLoading.value = false
    uploadRef.value.clearFiles()
    uploadFile.value = null
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
        'Probability (%)': (pred.probability * 100).toFixed(2)
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
          <img src="/logo1.png" :alt="t.logo1_alt" class="logo-image" />
          <img src="/logo2.png" :alt="t.logo2_alt" class="logo-image" />
          <img src="/logo3.png" :alt="t.logo3_alt" class="logo-image" />
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
          <p class="card-description">{{ t.uploadDesc }}</p>
          <el-upload
            ref="uploadRef" drag action="#" :limit="1" :auto-upload="false"
            @change="handleFileChange" accept=".xlsx,.zip,.rar,.7z" class="upload-area"
            @remove="() => uploadFile = null"
          >
            <el-icon class="el-icon--upload"><Upload /></el-icon>
            <div class="el-upload__text">{{ t.uploadDrag }}<em>{{ t.uploadClick }}</em></div>
          </el-upload>
          <div class="button-group">
            <el-button @click="handleDownloadTemplate" :icon="Download" size="large">{{ t.downloadTemplate }}</el-button>
            <el-button type="primary" @click="handlePrediction" :loading="isLoading" :icon="Position" size="large">{{ t.startCalc }}</el-button>
          </div>
        </el-card>

        <el-card class="results-panel-card" shadow="never">
          <template #header>
            <div class="results-header">
              <div class="card-header">
                <el-icon><DataAnalysis /></el-icon><span>{{ t.predictionResults }}</span>
              </div>
              <div v-if="hasResults" class="header-controls">
                <el-select-v2 v-model="selectedPatientId" :options="patientOptions" :placeholder="t.patientSelection" style="width: 200px;" filterable size="small" />
                <el-button @click="exportResultsToExcel" type="primary" plain :icon="Document" size="small">{{ t.downloadResults }}</el-button>
              </div>
            </div>
          </template>
          <div v-if="currentPatientData" class="results-content">
            <div class="charts-grid" :key="`${selectedPatientId}-${chartGridKey}`">
              <div v-for="pred in currentPatientData.predictions" :key="pred.disease_abbr" class="chart-container">
                <div class="disease-title" :title="locale === 'zh' ? pred.disease_name_cn : pred.disease_name_en">
                  {{ locale === 'zh' ? pred.disease_name_cn : pred.disease_name_en }}
                </div>
                <v-chart class="chart" :option="getChartOption(locale === 'zh' ? pred.disease_name_cn : pred.disease_name_en, pred.probability)" autoresize />
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
            <ul v-if="stats.usage_ranking_by_country.length > 0">
              <li v-for="(stat, index) in stats.usage_ranking_by_country" :key="stat.location">
                <span class="rank-badge" :class="`rank-${index + 1}`">{{ index + 1 }}</span>
                <span class="location">{{ stat.location }}</span>
                <span class="count">{{ stat.count }}</span>
              </li>
            </ul>
            <el-empty v-else :description="t.noData" :image-size="50" />
          </div>
          <div class="ranking-section">
            <h3>{{ t.visitRanking }}</h3>
            <ul v-if="stats.visit_ranking_by_country.length > 0">
              <li v-for="(stat, index) in stats.visit_ranking_by_country" :key="stat.location">
                <span class="rank-badge" :class="`rank-${index + 1}`">{{ index + 1 }}</span>
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
      <div class="footer-content">
        <p>{{ t.footerLine1 }}</p>
        <p class="footer-en">{{ t.footerLine2 }}</p>
      </div>
    </footer>
  </div>
</template>

<style>
:root {
  --color-primary: #4a90e2;
  --color-primary-light: #e8f3ff;
  --color-text-primary: #333;
  --color-text-regular: #555;
  --color-text-secondary: #888;
  --color-border: #e5e7eb;
  --bg-color-page: #f9fafb;
  --bg-color-card: #ffffff;
  --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
  --border-radius: 8px;
  --spacing-sm: 8px; --spacing-md: 16px; --spacing-lg: 24px; --spacing-xl: 32px;
  --font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: var(--font-family); background-color: var(--bg-color-page); color: var(--color-text-regular); -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
.container { width: 95%; max-width: 1800px; margin: 0 auto; }

/* General Components */
.el-card { border: 1px solid var(--color-border); border-radius: var(--border-radius); box-shadow: none; }
.el-card__header { border-bottom: 1px solid var(--color-border); padding: var(--spacing-md) var(--spacing-lg); }
.card-header { display: flex; align-items: center; gap: var(--spacing-sm); font-size: 1.1rem; font-weight: 600; color: var(--color-text-primary); }
.card-header .el-icon { font-size: 1.3rem; color: var(--color-primary); }

/* Header */
.app-header { background: var(--bg-color-card); padding: var(--spacing-md) 0; border-bottom: 1px solid var(--color-border); position: sticky; top: 0; z-index: 100; backdrop-filter: blur(10px); }
.header-content { display: flex; justify-content: space-between; align-items: center; gap: var(--spacing-lg); }
.logo-area { display: flex; align-items: center; gap: var(--spacing-md); }
.logo-image { height: 45px; object-fit: contain; }
.title-area { text-align: left; flex-grow: 1; }
.app-title { font-size: 1.5rem; font-weight: 700; color: var(--color-text-primary); margin: 0; }
.app-subtitle { font-size: 0.9rem; color: var(--color-text-secondary); margin: 4px 0 0; font-weight: 500; }
.header-actions { flex-shrink: 0; }

/* Main Content */
.main-content { padding: var(--spacing-xl) 0; }
.main-grid { display: grid; grid-template-columns: minmax(400px, 1.2fr) 3fr; gap: var(--spacing-xl); align-items: flex-start; margin-bottom: var(--spacing-xl); }

/* Control & Results Panels */
.control-panel-card, .results-panel-card { display: flex; flex-direction: column; height: 100%; }
.el-card__body { flex-grow: 1; display: flex; flex-direction: column; padding: var(--spacing-lg); }
.card-description { font-size: 0.9rem; color: var(--color-text-regular); margin: 0 0 var(--spacing-lg); line-height: 1.6; }
.upload-area { margin-bottom: var(--spacing-lg); flex-grow: 1; display: flex; }
.upload-area .el-upload { width: 100%; }
.upload-area .el-upload-dragger { width: 100%; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; border-radius: var(--border-radius); border: 2px dashed var(--color-border); transition: all 0.2s ease; }
.upload-area .el-upload-dragger:hover { border-color: var(--color-primary); background-color: var(--color-primary-light); }
.button-group { display: flex; gap: var(--spacing-md); }
.button-group .el-button { flex-grow: 1; }

.results-header { display: flex; justify-content: space-between; align-items: center; width: 100%; }
.header-controls { display: flex; align-items: center; gap: var(--spacing-md); }
.results-content { flex-grow: 1; }
.charts-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: var(--spacing-lg); padding: var(--spacing-sm) 0; }
.chart-container { background-color: var(--bg-color-page); border: 1px solid var(--color-border); border-radius: var(--border-radius); padding: var(--spacing-sm); height: 160px; display: flex; flex-direction: column; transition: all 0.2s ease; }
.chart-container:hover { transform: translateY(-2px); box-shadow: var(--shadow); }
.disease-title { font-size: 0.85rem; font-weight: 500; text-align: center; padding: var(--spacing-sm); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chart { flex-grow: 1; width: 100%; height: 100%; }
.full-height-empty { flex-grow: 1; display: flex; align-items: center; justify-content: center; }

/* Statistics Card */
.stats-overview { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--spacing-lg); text-align: center; padding: var(--spacing-md) 0; margin-bottom: var(--spacing-lg); border-bottom: 1px solid var(--color-border); }
.stat-item .stat-value { font-size: 2.5rem; font-weight: 700; color: var(--color-primary); line-height: 1.2; }
.stat-item .stat-label { font-size: 1rem; color: var(--color-text-secondary); margin-top: var(--spacing-sm); }
.stats-rankings { display: grid; grid-template-columns: 1fr 1fr; gap: var(--spacing-xl); }
.ranking-section h3 { font-size: 1.2rem; margin: 0 0 var(--spacing-md); }
.ranking-section ul { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: var(--spacing-sm); }
.ranking-section li { display: flex; align-items: center; font-size: 0.9rem; padding: var(--spacing-sm) var(--spacing-md); background-color: var(--bg-color-page); border-radius: 6px; }
.rank-badge { color: var(--color-text-secondary); font-weight: 600; border-radius: 4px; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; margin-right: var(--spacing-md); background-color: #e5e7eb; }
.rank-badge.rank-1 { background-color: #fef08a; color: #a16207; }
.rank-badge.rank-2 { background-color: #e5e7eb; color: #4b5563; }
.rank-badge.rank-3 { background-color: #fde68a; color: #b45309; }
.ranking-section .location { flex-grow: 1; font-weight: 500; }
.ranking-section .count { font-weight: 600; color: var(--color-primary); }

/* Footer */
.app-footer { background: #333; color: #ccc; text-align: center; padding: var(--spacing-lg); font-size: 0.9rem; }
.footer-content p { margin: 5px 0; }
.footer-en { color: #aaa; font-size: 0.8rem; }

/* Responsive */
@media (max-width: 1200px) {
  .main-grid { grid-template-columns: 1fr; }
  .results-panel-card { min-height: 500px; }
  .charts-grid { grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); }
}
@media (max-width: 768px) {
  .header-content { flex-direction: column; text-align: center; gap: var(--spacing-md); }
  .title-area { text-align: center; }
  .stats-overview, .stats-rankings { grid-template-columns: 1fr; }
  .charts-grid { grid-template-columns: repeat(2, 1fr); }
  .chart-container { height: 140px; }
}
@media (min-width: 1600px) {
    .charts-grid { grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); }
    .chart-container { height: 180px; }
}
</style>