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
      logo1_alt: '国家产科专业医疗质量控制中心',
      logo2_alt: '国家妇产疾病临床医学研究中心',
      logo3_alt: '北京大学第三医院',
      singlePatient: '单个患者预测',
      singlePatientDesc: '上传单个患者的 Excel 文件 (.xlsx)。文件名将作为患者ID。',
      batchPatient: '批量预测',
      batchPatientDesc: '上传包含多个患者 Excel 文件的压缩包 (.zip / .rar / .7z)。',
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
      footerLine1: '国家产科专业医疗质量控制中心 | 国家妇产疾病临床医学研究中心 | 北京大学第三医院妇产科',
      footerLine2: 'National Centre for Healthcare Quality Management in Obstetrics | National Clinical Research Center for Obstetrics and Gynecology | Department of Obstetrics and Gynecology, Peking University Third Hospital',
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
      predictionTips: '提示：概率值显示为百分比形式，数值越高表示风险越大。',
      riskLevel: {
        low: '低风险',
        medium: '中等风险',
        high: '高风险'
      }
    },
    en: {
      title: 'Assessment of Pregnancy-Related Disease Risks in Repeat Pregnancies',
      subtitle: '再次妊娠孕期疾病发生风险评估',
      logo1_alt: 'National Centre for Healthcare Quality Management in Obstetrics',
      logo2_alt: 'National Clinical Research Center for Obstetrics and Gynecology',
      logo3_alt: 'Peking University Third Hospital',
      singlePatient: 'Single Patient Prediction',
      singlePatientDesc: 'Upload a single patient\'s Excel file (.xlsx). The filename will be used as Patient ID.',
      batchPatient: 'Batch Prediction',
      batchPatientDesc: 'Upload an archive (.zip / .rar / .7z) containing multiple patient Excel files.',
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
      footerLine1: '国家产科专业医疗质量控制中心 | 国家妇产疾病临床医学研究中心 | 北京大学第三医院妇产科',
      footerLine2: 'National Centre for Healthcare Quality Management in Obstetrics | National Clinical Research Center for Obstetrics and Gynecology | Department of Obstetrics and Gynecology, Peking University Third Hospital',
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
      predictionTips: 'Tips: Probability values are shown as percentages. Higher values indicate greater risk.',
      riskLevel: {
        low: 'Low Risk',
        medium: 'Medium Risk',
        high: 'High Risk'
      }
    }
  }

  const t = computed(() => messages[locale.value])

  const toggleLang = () => {
    locale.value = locale.value === 'zh' ? 'en' : 'zh'
  }

  return { locale, t, toggleLang }
}

// --- API Composable ---
const useApi = () => {
  const fetchStats = async () => {
    try {
      const { data } = await axios.get(`${API_BASE_URL}/api/stats`)
      return data
    } catch (error) {
      console.error('Failed to fetch stats:', error)
      return {
        total_visits: 0,
        total_predictions: 0,
        unique_countries_count: 0,
        visit_ranking_by_country: [],
        usage_ranking_by_country: [],
      }
    }
  }

  const downloadTemplate = async () => {
    const response = await axios.get(`${API_BASE_URL}/api/download-template`, {
      responseType: 'blob'
    })
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

  return { fetchStats, downloadTemplate, submitPrediction }
}

// --- File Upload Composable ---
const useFileUpload = () => {
  const singleFile = ref(null)
  const batchFile = ref(null)
  const singleUploadRef = ref(null)
  const batchUploadRef = ref(null)

  const handleFileChange = (uploadFile, type, uploadRef) => {
    const file = uploadFile.raw
    const fileExt = file.name.split('.').pop().toLowerCase()

    if (type === 'single') {
      if (fileExt !== 'xlsx') {
        ElMessage.error('Please upload a .xlsx file!')
        uploadRef.clearFiles()
        return
      }
      singleFile.value = file
    } else {
      if (!['zip', 'rar', '7z'].includes(fileExt)) {
        ElMessage.error('Please upload a valid archive (.zip, .rar, or .7z)!')
        uploadRef.clearFiles()
        return
      }
      batchFile.value = file
    }
  }

  const clearFile = (type, uploadRef) => {
    if (type === 'single') {
      singleFile.value = null
    } else {
      batchFile.value = null
    }
    uploadRef?.clearFiles()
  }

  return {
    singleFile,
    batchFile,
    singleUploadRef,
    batchUploadRef,
    handleFileChange,
    clearFile
  }
}

// --- Chart Composable ---
const useCharts = () => {
  const getRiskLevel = (probability) => {
    if (probability >= 0.5) return 'high'
    if (probability >= 0.2) return 'medium'
    return 'low'
  }

  const getRiskColor = (probability) => {
    const level = getRiskLevel(probability)
    const colors = {
      low: '#67c23a',
      medium: '#e6a23c',
      high: '#f56c6c'
    }
    return colors[level]
  }

  const getChartOption = (diseaseName, probability, riskLabels) => {
    const probPercent = probability * 100
    const color = getRiskColor(probability)
    const riskLevel = getRiskLevel(probability)

    return {
      title: {
        text: diseaseName,
        left: 'center',
        bottom: '8%',
        textStyle: {
          fontSize: 13,
          fontWeight: '600',
          color: '#606266',
          overflow: 'truncate',
          width: 200
        },
      },
      tooltip: {
        trigger: 'item',
        formatter: `${diseaseName}<br/>概率: {c}%<br/>风险等级: ${riskLabels[riskLevel]}`
      },
      series: [{
        type: 'pie',
        radius: ['70%', '90%'],
        avoidLabelOverlap: false,
        silent: false,
        label: {
          show: true,
          position: 'center',
          formatter: `${probPercent.toFixed(1)}%`,
          fontSize: 20,
          fontWeight: 'bold',
          color: color
        },
        emphasis: {
          label: {
            fontSize: 24,
            fontWeight: 'bold'
          },
          scale: true,
          scaleSize: 5
        },
        data: [
          {
            value: probPercent.toFixed(1),
            name: 'Probability',
            itemStyle: { color }
          },
          {
            value: (100 - probPercent).toFixed(1),
            name: 'Remaining',
            itemStyle: { color: '#EBEEF5' }
          }
        ],
      }]
    }
  }

  return { getChartOption, getRiskLevel, getRiskColor }
}

// --- Main component setup ---
const { locale, t, toggleLang } = useI18n()
const { fetchStats, downloadTemplate, submitPrediction } = useApi()
const { singleFile, batchFile, singleUploadRef, batchUploadRef, handleFileChange, clearFile } = useFileUpload()
const { getChartOption } = useCharts()

// --- State ---
const isLoadingSingle = ref(false)
const isLoadingBatch = ref(false)
const allPatientResults = ref([])
const selectedPatientId = ref(null)
const stats = ref({
  total_visits: 0,
  total_predictions: 0,
  unique_countries_count: 0,
  visit_ranking_by_country: [],
  usage_ranking_by_country: [],
})
const chartGridKey = ref(0)

// --- Computed ---
const hasResults = computed(() => allPatientResults.value.length > 0)
const currentPatientData = computed(() => {
  if (!selectedPatientId.value) return null
  return allPatientResults.value.find(p => p.patient_id === selectedPatientId.value)
})

const patientOptions = computed(() =>
  allPatientResults.value.map(p => ({
    value: p.patient_id,
    label: `${p.patient_id} (${p.predictions.length} predictions)`
  }))
)

// --- Methods ---
const handleDownloadTemplate = async () => {
  const loading = ElLoading.service({
    lock: true,
    text: t.value.downloadingTemplate,
    background: 'rgba(0, 0, 0, 0.8)'
  })
  try {
    await downloadTemplate()
    ElMessage.success('模板下载成功！')
  } catch (error) {
    ElMessage.error(t.value.templateError)
  } finally {
    loading.close()
  }
}

const handlePrediction = async (type) => {
  const isSingle = type === 'single'
  const file = isSingle ? singleFile.value : batchFile.value
  const endpoint = isSingle ? '/api/predict-single' : '/api/predict-batch'
  const loadingRef = isSingle ? isLoadingSingle : isLoadingBatch
  const uploadRef = isSingle ? singleUploadRef : batchUploadRef

  if (!file) {
    ElMessage.warning(isSingle ? t.value.selectSingleFile : t.value.selectBatchFile)
    return
  }

  loadingRef.value = true

  try {
    const results = await submitPrediction(file, endpoint)

    // Update results with deduplication
    results.forEach(result => {
      const existingIndex = allPatientResults.value.findIndex(p => p.patient_id === result.patient_id)
      if (existingIndex > -1) {
        allPatientResults.value.splice(existingIndex, 1)
      }
      allPatientResults.value.unshift(result)
    })

    if (results.length > 0) {
      selectedPatientId.value = results[0].patient_id
      ElMessage.success(
        isSingle
          ? t.value.predictionSuccess.replace('{patientId}', results[0].patient_id)
          : t.value.batchSuccess
      )
    }

    // Refresh stats
    const newStats = await fetchStats()
    stats.value = newStats
  } catch (error) {
    const detail = error.response?.data?.detail || 'Unknown error.'
    ElMessage.error(t.value.calcFailed.replace('{detail}', detail))
  } finally {
    loadingRef.value = false
    clearFile(type, uploadRef.value)
    nextTick(() => chartGridKey.value++)
  }
}

const exportResultsToExcel = () => {
  if (!hasResults.value) return

  try {
    const dataToExport = allPatientResults.value.map(patient => {
      const row = { [t.value.patientID]: patient.patient_id }
      patient.predictions.forEach(pred => {
        const key = `${pred.disease_abbr} (${locale.value === 'zh' ? pred.disease_name_cn : pred.disease_name_en})`
        row[key] = (pred.probability * 100).toFixed(2) + '%'
      })
      return row
    })

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
  const newStats = await fetchStats()
  stats.value = newStats
  document.title = t.value.title
})
</script>

<template>
  <div class="app-container">
    <!-- Header -->
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
          <el-button @click="toggleLang" type="primary" plain round size="large">
            {{ t.langSwitch }}
          </el-button>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="container main-content">
      <div class="main-grid">
        <!-- Control Panel -->
        <div class="control-panel">
          <!-- Single Patient Card -->
          <el-card class="prediction-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon class="header-icon"><User /></el-icon>
                <span>{{ t.singlePatient }}</span>
              </div>
            </template>

            <p class="card-description">{{ t.singlePatientDesc }}</p>

            <el-upload
              ref="singleUploadRef"
              drag
              action="#"
              :limit="1"
              :auto-upload="false"
              @change="(file) => handleFileChange(file, 'single', singleUploadRef)"
              accept=".xlsx"
              class="upload-area"
            >
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">
                {{ t.uploadDrag }}<em>{{ t.uploadClick }}</em>
              </div>
            </el-upload>

            <div class="button-group">
              <el-button @click="handleDownloadTemplate" :icon="Download" size="large">
                {{ t.downloadTemplate }}
              </el-button>
              <el-button
                type="primary"
                @click="handlePrediction('single')"
                :loading="isLoadingSingle"
                :icon="Position"
                size="large"
              >
                {{ t.startCalc }}
              </el-button>
            </div>
          </el-card>

          <!-- Batch Prediction Card -->
          <el-card class="prediction-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon class="header-icon"><Files /></el-icon>
                <span>{{ t.batchPatient }}</span>
              </div>
            </template>

            <p class="card-description">{{ t.batchPatientDesc }}</p>

            <el-upload
              ref="batchUploadRef"
              drag
              action="#"
              :limit="1"
              :auto-upload="false"
              @change="(file) => handleFileChange(file, 'batch', batchUploadRef)"
              accept=".zip,.rar,.7z"
              class="upload-area"
            >
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">
                {{ t.uploadDrag }}<em>{{ t.uploadClick }}</em>
              </div>
            </el-upload>

            <div class="button-group-single">
              <el-button
                type="success"
                @click="handlePrediction('batch')"
                :loading="isLoadingBatch"
                :icon="Promotion"
                class="full-width-btn"
                size="large"
              >
                {{ t.processAndSubmit }}
              </el-button>
            </div>
          </el-card>
        </div>

        <!-- Results Panel -->
        <div class="results-panel">
          <el-card class="results-card" shadow="hover">
            <template #header>
              <div class="results-header">
                <div class="card-header">
                  <el-icon class="header-icon"><DataAnalysis /></el-icon>
                  <span>{{ t.predictionResults }}</span>
                </div>

                <div v-if="hasResults" class="header-controls">
                  <el-select-v2
                    v-model="selectedPatientId"
                    :options="patientOptions"
                    :placeholder="t.patientSelection"
                    style="width: 260px; margin-right: 12px;"
                    filterable
                    size="default"
                  />
                  <el-button @click="exportResultsToExcel" type="primary" :icon="Document" size="default">
                    {{ t.downloadResults }}
                  </el-button>
                </div>
              </div>
            </template>

            <div v-if="currentPatientData" class="results-content">
              <div class="prediction-tips">
                <el-alert :title="t.predictionTips" type="info" :closable="false" show-icon />
              </div>

              <div class="charts-grid" :key="`${selectedPatientId}-${chartGridKey}`">
                <div
                  v-for="pred in currentPatientData.predictions"
                  :key="pred.disease_abbr"
                  class="chart-container"
                >
                  <v-chart
                    class="chart"
                    :option="getChartOption(
                      locale === 'zh' ? pred.disease_name_cn : pred.disease_name_en,
                      pred.probability,
                      t.riskLevel
                    )"
                    autoresize
                  />
                </div>
              </div>
            </div>

            <el-empty v-else :description="t.noResults" class="full-height-empty">
              <template #image>
                <el-icon size="100" color="#c0c4cc"><DataAnalysis /></el-icon>
              </template>
            </el-empty>
          </el-card>
        </div>
      </div>

      <!-- Statistics Card -->
      <el-card class="stats-card" shadow="hover">
        <template #header>
          <div class="card-header">
            <el-icon class="header-icon"><TrendCharts /></el-icon>
            <span>{{ t.stats }}</span>
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

    <!-- Footer -->
    <footer class="app-footer">
      <div class="footer-content">
        <p class="footer-cn">{{ t.footerLine1 }}</p>
        <p class="footer-en">{{ t.footerLine2 }}</p>
      </div>
    </footer>
  </div>
</template>

<style>
:root {
  --color-primary: #409eff;
  --color-primary-light: #ecf5ff;
  --color-success: #67c23a;
  --color-warning: #e6a23c;
  --color-danger: #f56c6c;
  --color-text-primary: #303133;
  --color-text-regular: #606266;
  --color-text-secondary: #909399;
  --color-border: #dcdfe6;
  --color-border-light: #ebeef5;
  --bg-color-page: #f8fafc;
  --bg-color-card: #ffffff;
  --shadow-light: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  --shadow-base: 0 2px 4px rgba(0, 0, 0, 0.12), 0 0 6px rgba(0, 0, 0, 0.04);
  --border-radius: 12px;
  --border-radius-large: 16px;
  --spacing-xs: 0.5rem;
  --spacing-sm: 1rem;
  --spacing-md: 1.5rem;
  --spacing-lg: 2rem;
  --spacing-xl: 2.5rem;
  --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: var(--font-family);
  background-color: var(--bg-color-page);
  color: var(--color-text-primary);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.container {
  width: 95%;
  max-width: 1400px;
  margin: 0 auto;
}

/* Header Styles */
.app-header {
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  padding: var(--spacing-lg) 0;
  border-bottom: 1px solid var(--color-border-light);
  box-shadow: var(--shadow-base);
  position: sticky;
  top: 0;
  z-index: 100;
  backdrop-filter: blur(10px);
}

.header-content {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: var(--spacing-lg);
}

.logo-area {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.logo-image {
  height: 50px;
  width: auto;
  object-fit: contain;
  border-radius: 8px;
  transition: all 0.3s ease;
  background: transparent;
}

.logo-image:hover {
  transform: translateY(-2px);
  filter: brightness(1.1);
}

.title-area {
  text-align: center;
}

.app-title {
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--color-text-primary);
  margin: 0 0 0.25rem 0;
  background: linear-gradient(135deg, var(--color-primary), var(--color-success));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.app-subtitle {
  font-size: 0.9rem;
  color: var(--color-text-secondary);
  margin: 0;
  font-weight: 500;
}

.header-actions {
  display: flex;
  align-items: center;
}

/* Main Content */
.main-content {
  padding: var(--spacing-xl) 0;
  min-height: calc(100vh - 200px);
}

.main-grid {
  display: grid;
  grid-template-columns: minmax(400px, 1fr) 2fr;
  gap: var(--spacing-xl);
  align-items: flex-start;
  margin-bottom: var(--spacing-xl);
}

/* Control Panel */
.control-panel {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
}

.prediction-card {
  border: none;
  border-radius: var(--border-radius-large);
  overflow: hidden;
  transition: all 0.3s ease;
}

.prediction-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-light);
}

.card-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--color-text-primary);
}

.header-icon {
  font-size: 1.5rem;
  color: var(--color-primary);
}

.card-description {
  font-size: 0.95rem;
  color: var(--color-text-regular);
  margin: 0 0 var(--spacing-lg) 0;
  line-height: 1.7;
  background-color: #f8fafc;
  padding: var(--spacing-md);
  border-radius: var(--border-radius);
  border-left: 4px solid var(--color-primary);
}

.upload-area {
  margin-bottom: var(--spacing-lg);
}

.el-upload-dragger {
  padding: var(--spacing-xl) var(--spacing-lg);
  border-radius: var(--border-radius-large);
  border: 2px dashed #d9ecff;
  background-color: #fafcff;
  transition: all 0.3s ease;
}

.el-upload-dragger:hover {
  border-color: var(--color-primary);
  background-color: var(--color-primary-light);
}

.button-group {
  display: flex;
  justify-content: space-between;
  gap: var(--spacing-sm);
}

.button-group-single {
  display: flex;
}

.full-width-btn {
  width: 100%;
}

/* Results Panel */
.results-card {
  border: none;
  border-radius: var(--border-radius-large);
  overflow: hidden;
  min-height: 700px;
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: var(--spacing-sm);
}

.header-controls {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  flex-wrap: wrap;
}

.results-content {
  height: 100%;
}

.prediction-tips {
  margin-bottom: var(--spacing-lg);
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: var(--spacing-lg);
  padding: var(--spacing-sm) 0;
}

.chart-container {
  background: linear-gradient(135deg, #fafcfe, #f8fafc);
  border: 2px solid var(--color-border-light);
  border-radius: var(--border-radius-large);
  padding: var(--spacing-md);
  height: 240px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.chart-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, var(--color-primary), var(--color-success));
  opacity: 0;
  transition: opacity 0.3s ease;
}

.chart-container:hover::before {
  opacity: 1;
}

.chart-container:hover {
  transform: translateY(-6px);
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.15);
  border-color: var(--color-primary);
}

.chart {
  width: 100%;
  height: 100%;
}

.full-height-empty {
  height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Statistics Card */
.stats-card {
  border: none;
  border-radius: var(--border-radius-large);
  overflow: hidden;
}

.stats-overview {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--spacing-lg);
  text-align: center;
  padding: var(--spacing-xl) 0;
  border-bottom: 2px solid var(--color-border-light);
}

.stat-item {
  padding: var(--spacing-lg);
  border-radius: var(--border-radius-large);
  background: linear-gradient(135deg, #fafcfe, #f8fafc);
  transition: all 0.3s ease;
}

.stat-item:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-light);
}

.stat-value {
  font-size: 3.5rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--color-primary), var(--color-success));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-family: 'SF Pro Display', -apple-system, sans-serif;
  line-height: 1.2;
}

.stat-label {
  font-size: 1.1rem;
  color: var(--color-text-secondary);
  margin-top: var(--spacing-sm);
  font-weight: 600;
}

.stats-rankings {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--spacing-xl);
  padding-top: var(--spacing-xl);
}

.ranking-section h3 {
  font-size: 1.4rem;
  margin: 0 0 var(--spacing-lg) 0;
  color: var(--color-text-primary);
  border-bottom: 3px solid var(--color-primary);
  padding-bottom: var(--spacing-sm);
  display: inline-block;
  font-weight: 700;
}

.ranking-section ul {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.ranking-section li {
  display: flex;
  align-items: center;
  font-size: 1rem;
  color: var(--color-text-regular);
  padding: var(--spacing-md);
  background-color: #fafcfe;
  border-radius: var(--border-radius);
  transition: all 0.3s ease;
}

.ranking-section li:hover {
  background-color: var(--color-primary-light);
  transform: translateX(6px);
}

.rank-badge {
  background: linear-gradient(135deg, #f0f2f5, #e4e7ed);
  color: var(--color-text-secondary);
  font-weight: 800;
  border-radius: 50%;
  width: 36px;
  height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-right: var(--spacing-md);
  flex-shrink: 0;
  font-size: 0.9rem;
}

.rank-badge.rank-1 {
  background: linear-gradient(135deg, #ffd700, #ffed4e);
  color: #8b5a00;
  box-shadow: 0 4px 12px rgba(255, 215, 0, 0.4);
}

.rank-badge.rank-2 {
  background: linear-gradient(135deg, #c0c0c0, #e5e5e5);
  color: #666;
  box-shadow: 0 4px 12px rgba(192, 192, 192, 0.4);
}

.rank-badge.rank-3 {
  background: linear-gradient(135deg, #cd7f32, #d4a574);
  color: #5a3a1a;
  box-shadow: 0 4px 12px rgba(205, 127, 50, 0.4);
}

.ranking-section .location {
  flex-grow: 1;
  font-weight: 600;
  color: var(--color-text-primary);
}

.ranking-section .count {
  font-weight: 800;
  color: var(--color-primary);
  background-color: var(--color-primary-light);
  padding: 0.4rem 1rem;
  border-radius: 20px;
  font-size: 0.9rem;
}

/* Footer */
.app-footer {
  background: linear-gradient(135deg, #303133, #262629);
  color: rgba(255, 255, 255, 0.8);
  text-align: center;
  padding: var(--spacing-xl) var(--spacing-sm);
  border-top: 2px solid var(--color-primary);
}

.footer-content {
  max-width: 1200px;
  margin: 0 auto;
}

.footer-cn,
.footer-en {
  margin: var(--spacing-xs) 0;
  line-height: 1.8;
}

.footer-cn {
  font-size: 1rem;
  font-weight: 500;
}

.footer-en {
  font-size: 0.9rem;
  opacity: 0.9;
}

/* Responsive Design */
@media (max-width: 1200px) {
  .main-grid {
    grid-template-columns: 1fr;
    gap: var(--spacing-lg);
  }

  .results-panel {
    order: -1;
  }

  .charts-grid {
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: var(--spacing-md);
  }
}

@media (max-width: 768px) {
  .header-content {
    grid-template-columns: 1fr;
    gap: var(--spacing-md);
    text-align: center;
  }

  .logo-area {
    order: 2;
    justify-content: center;
    gap: var(--spacing-sm);
  }

  .logo-image {
    height: 40px;
  }

  .title-area {
    order: 1;
  }

  .app-title {
    font-size: 1.5rem;
  }

  .app-subtitle {
    font-size: 0.8rem;
  }

  .header-actions {
    order: 3;
    justify-content: center;
  }

  .stats-overview,
  .stats-rankings {
    grid-template-columns: 1fr;
    gap: var(--spacing-md);
  }

  .charts-grid {
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: var(--spacing-sm);
  }

  .chart-container {
    height: 220px;
    padding: var(--spacing-sm);
  }

  .results-header {
    flex-direction: column;
    align-items: stretch;
    gap: var(--spacing-md);
  }

  .header-controls {
    justify-content: stretch;
    flex-direction: column;
  }

  .header-controls .el-select-v2 {
    margin-right: 0 !important;
    margin-bottom: var(--spacing-sm);
  }

  .main-content {
    padding: var(--spacing-lg) 0;
  }

  .container {
    width: 98%;
  }

  .button-group {
    flex-direction: column;
    gap: var(--spacing-sm);
  }
}

@media (max-width: 480px) {
  .app-title {
    font-size: 1.3rem;
  }

  .logo-area {
    gap: 0.75rem;
  }

  .logo-image {
    height: 35px;
  }

  .stat-value {
    font-size: 2.5rem;
  }

  .charts-grid {
    grid-template-columns: 1fr;
  }

  .chart-container {
    height: 200px;
  }
}

/* Element Plus Customizations */
.el-card {
  --el-card-padding: 24px;
}

.el-card__header {
  border-bottom: 2px solid var(--color-border-light);
  background: linear-gradient(90deg, #fafcfe, #f8fafc);
  padding: 20px 24px;
}

.el-alert {
  border-radius: var(--border-radius);
}

.el-button {
  border-radius: var(--border-radius);
  font-weight: 600;
}

.el-button--large {
  padding: 14px 24px;
  font-size: 1rem;
}

.el-select-v2 {
  border-radius: var(--border-radius);
}

/* Chart animations */
.chart-container {
  animation: fadeInUp 0.6s ease-out;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Loading states */
.el-loading-mask {
  border-radius: var(--border-radius-large);
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: var(--color-border-light);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb {
  background: var(--color-primary);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: #337ecc;
}
</style>