<script setup>
import { ref, reactive, onMounted, computed, watch } from 'vue'
import axios from 'axios'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { GaugeChart, MapChart, EffectScatterChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GeoComponent,
  VisualMapComponent
} from 'echarts/components'
import VChart, { THEME_KEY } from 'vue-echarts' // THEME_KEY might not be needed unless using themes
import { ElMessage, ElLoading } from 'element-plus'
import 'element-plus/theme-chalk/el-message.css';
import 'element-plus/theme-chalk/el-loading.css';

// ECharts components
use([
  CanvasRenderer,
  GaugeChart,
  MapChart,
  EffectScatterChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GeoComponent,
  VisualMapComponent
]);

// If world map data isn't bundled with ECharts, you might need:
// import 'echarts/map/js/world.js'; // Ensure this is installed if needed: pnpm add echarts -D

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const form = reactive({
  amh: '',
  menstrual_start: '',
  menstrual_end: '',
  bmi: '',
  androstenedione: ''
});

const calculating = ref(false);
const result = ref(null);
const usageStats = ref([]);
const visitStats = ref([]);
const worldMapPoints = ref([]); // Renamed for clarity
const showBatchModal = ref(false);
const uploadFileInstance = ref(null); // To store the ElUpload instance
const uploadedFile = ref(null); // To store the actual file
const batchCalculating = ref(false);
const batchResults = ref([]);


const riskLevelTextMap = {
  '高危': { text: '高危人群', colorClass: 'high-risk', english: 'High-Risk Group' },
  '中危': { text: '中危人群', colorClass: 'medium-risk', english: 'Medium-Risk Group' },
  '低危': { text: '低危人群', colorClass: 'low-risk', english: 'Low-Risk Group' },
  '未知': { text: '未知风险', colorClass: 'unknown-risk', english: 'Unknown Risk Group' }
};

const currentRiskInfo = computed(() => {
  if (result.value && result.value.risk_level) {
    return riskLevelTextMap[result.value.risk_level] || riskLevelTextMap['未知'];
  }
  return riskLevelTextMap['未知'];
});


// Probability Gauge Chart
const probabilityOption = computed(() => ({
  series: [{
    type: 'gauge',
    center: ['50%', '60%'], // Adjusted center
    radius: '100%', // Adjusted radius
    startAngle: 180,
    endAngle: 0,
    min: 0,
    max: 100,
    splitNumber: 10,
    progress: {
      show: true,
      width: 18,
      itemStyle: {
        color: 'auto'
      }
    },
    axisLine: {
      lineStyle: {
        width: 18,
        color: [
          [0.2, '#67e0e3'], // Low risk color
          [0.6, '#FFDB5C'], // Medium risk color
          [1, '#fd666d']   // High risk color
        ]
      }
    },
    axisTick: { show: false },
    splitLine: { show: false },
    axisLabel: {
      distance: 5,
      color: '#999',
      fontSize: 10
    },
    pointer: {
      icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
      length: '60%', // Adjusted length
      width: 8,    // Adjusted width
      offsetCenter: [0, '-15%'], // Adjusted offset
      itemStyle: { color: 'auto' }
    },
    anchor: {
      show: true,
      showAbove: true,
      size: 15,
      itemStyle: {
        borderColor: '#ddd',
        borderWidth: 4
      }
    },
    title: {
      offsetCenter: [0, '30%'], // Adjusted position
      fontSize: 16,
      color: '#333'
    },
    detail: {
      valueAnimation: true,
      fontSize: 24, // Adjusted size
      offsetCenter: [0, '10%'], // Adjusted position
      formatter: '{value}%',
      color: 'auto'
    },
    data: [{
      value: result.value ? result.value.risk_percentage : 0,
      name: '患病概率'
    }]
  }]
}));

// World Map Chart
const worldMapOption = ref({
  backgroundColor: 'transparent', // Match page background or specific theme
  tooltip: {
    trigger: 'item',
    formatter: function (params) {
      if (params.data) {
        return `${params.data.name}<br/>访问次数: ${params.data.count || params.data.value[2]}`;
      }
      return params.name;
    }
  },
  geo: {
    map: 'world',
    roam: true,
    zoom: 1.2,
    itemStyle: {
      areaColor: '#A2D2FF', // Light blue areas
      borderColor: '#FFF',   // White borders
      borderWidth: 0.5
    },
    emphasis: {
      label: { show: false },
      itemStyle: {
        areaColor: '#66B2FF' // Darker blue on hover
      }
    },
    select: { // If selection is needed
        disabled: true
    }
  },
  series: [{
    name: '访问点',
    type: 'effectScatter',
    coordinateSystem: 'geo',
    data: [], // To be populated by worldMapPoints
    symbolSize: function (val) {
      // Scale symbol size based on count, with min/max
      return Math.max(5, Math.min(20, Math.log2(val[2] + 1) * 3)); 
    },
    rippleEffect: {
      brushType: 'stroke',
      scale: 3
    },
    hoverAnimation: true,
    label: { show: false },
    itemStyle: {
      color: '#FF6B6B', // Hot pink/red for points
      shadowBlur: 5,
      shadowColor: 'rgba(0,0,0,0.3)'
    },
    emphasis: {
        scale: true
    }
  }]
});

watch(worldMapPoints, (newData) => {
  worldMapOption.value.series[0].data = newData;
}, { deep: true });


const validateForm = () => {
  const numericFields = ['amh', 'menstrual_start', 'menstrual_end', 'bmi', 'androstenedione'];
  for (const field of numericFields) {
    if (form[field] !== '' && isNaN(parseFloat(form[field]))) {
      ElMessage.error(`${field} 必须是数字`);
      return false;
    }
     if (form[field] !== '' && parseFloat(form[field]) < 0) {
      ElMessage.error(`${field} 不能为负数`);
      return false;
    }
  }
  if (form.menstrual_start !== '' && form.menstrual_end !== '' && parseFloat(form.menstrual_start) > parseFloat(form.menstrual_end)) {
    ElMessage.error('月经周期开始天数不能大于结束天数');
    return false;
  }
  // Check if all fields are empty
  if (Object.values(form).every(value => value === '')) {
      ElMessage.error('请至少填写一项指标进行计算');
      return false;
  }
  return true;
};

const calculatePCOS = async () => {
  if (!validateForm()) return;

  calculating.value = true;
  let loadingInstance;
  try {
    loadingInstance = ElLoading.service({ text: '计算中...', background: 'rgba(0, 0, 0, 0.7)' });
    const payload = {};
    // Only include fields that have values
    for (const key in form) {
        if (form[key] !== '') {
            payload[key] = parseFloat(form[key]);
        } else {
            payload[key] = null; // Explicitly send null for empty fields if backend expects it
        }
    }

    const response = await axios.post(`${API_BASE}/api/calculate`, payload);
    result.value = response.data;
    ElMessage.success('计算完成！');
    loadStatistics(); // Refresh stats
  } catch (error) {
    console.error("Calculation error:", error);
    const errorMsg = error.response?.data?.detail || error.message || '计算失败，请稍后重试。';
    ElMessage.error(errorMsg);
    result.value = null; // Clear previous result on error
  } finally {
    calculating.value = false;
    if (loadingInstance) loadingInstance.close();
  }
};

const loadStatistics = async () => {
  try {
    const [usageResponse, visitResponse, mapResponse] = await Promise.all([
      axios.get(`${API_BASE}/api/usage-stats`),
      axios.get(`${API_BASE}/api/visit-stats`),
      axios.get(`${API_BASE}/api/world-map-data`)
    ]);
    usageStats.value = usageResponse.data;
    visitStats.value = visitResponse.data;
    worldMapPoints.value = mapResponse.data;
  } catch (error) {
    console.error('加载统计数据失败:', error);
    // ElMessage.error('加载统计数据失败'); // Avoid too many errors on interval
  }
};

const downloadTemplate = async () => {
  try {
    const response = await axios.get(`${API_BASE}/api/download-template`, {
      responseType: 'blob'
    });
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'PCOS_批量计算模板.xlsx');
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error("Download template error:", error);
    ElMessage.error('下载模板失败');
  }
};

const handleFileChangeForBatch = (file) => {
  // file is an object with raw property
  uploadedFile.value = file.raw;
};

const beforeUploadBatch = (file) => {
    const isXlsx = file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
    const isXls = file.type === 'application/vnd.ms-excel';
    if (!isXlsx && !isXls) {
        ElMessage.error('只能上传 .xlsx 或 .xls 文件!');
    }
    // Max size, e.g. 2MB
    const isLt2M = file.size / 1024 / 1024 < 2;
    if (!isLt2M) {
        ElMessage.error('上传文件大小不能超过 2MB!');
    }
    return (isXlsx || isXls) && isLt2M;
};


const processBatchCalculation = async () => {
  if (!uploadedFile.value) {
    ElMessage.error('请先选择一个文件');
    return;
  }

  batchCalculating.value = true;
  batchResults.value = []; // Clear previous results
  let loadingInstance;
  try {
    loadingInstance = ElLoading.service({ target: '.el-dialog', text: '批量计算中...', background: 'rgba(0, 0, 0, 0.7)' });
    const formData = new FormData();
    formData.append('file', uploadedFile.value);
    
    const response = await axios.post(`${API_BASE}/api/batch-calculate`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    
    batchResults.value = response.data.results.map(item => ({
      index: item.original_index,
      probability: item.status === "成功" ? (item.risk_percentage || 0).toFixed(3) + '%' : '-',
      risk_level: item.status === "成功" ? item.risk_level : '-',
      status: item.status,
      error: item.error || ''
    }));
    ElMessage.success('批量计算完成');
    loadStatistics(); // Refresh global stats
  } catch (error) {
    console.error("Batch calculation error:", error);
    const errorMsg = error.response?.data?.detail || error.message || '批量计算失败，请检查文件内容或稍后重试。';
    ElMessage.error(errorMsg);
  } finally {
    batchCalculating.value = false;
    if (loadingInstance) loadingInstance.close();
    // Reset upload state
    if (uploadFileInstance.value) {
      uploadFileInstance.value.clearFiles();
    }
    uploadedFile.value = null;
  }
};

// For the risk stratification display
const riskScaleLevels = [
  { p: 100, c: 'high' }, { p: 90, c: 'high' }, { p: 80, c: 'high' }, { p: 70, c: 'high' },
  { p: 60, c: 'medium' }, { p: 50, c: 'medium' }, { p: 40, c: 'medium' },
  { p: 30, c: 'low' }, { p: 20, c: 'low' }, { p: 10, c: 'low' }, { p: 0, c: 'low' }
];


onMounted(() => {
  loadStatistics();
  // setInterval(loadStatistics, 60000); // Refresh stats every 60 seconds
});

</script>

<template>
  <div id="app-container">
    <header class="app-header">
      <div class="container header-content">
        <!-- <img src="/logo.png" alt="PCOS Tool Logo" class="header-logo"> -->
        <div class="header-title">
          <h1>多囊卵巢综合征筛查工具</h1>
          <p>PCOSt: Polycystic Ovary Syndrome Screening Tool</p>
        </div>
      </div>
    </header>

    <main class="container">
      <section class="map-section card-style">
        <h2 class="section-title">
          <span class="title-ch">世界浏览记录实时监测</span>
          <span class="title-en">Global Access Monitoring</span>
        </h2>
        <div class="map-container">
          <v-chart :option="worldMapOption" autoresize class="echart-instance" />
        </div>
      </section>

      <div class="content-grid">
        <section class="calculator-main-section">
          <div class="card-style form-card">
            <h2 class="section-title form-title">
              <!-- <img src="/logo.png" alt="logo" class="title-logo-small"> -->
              <span class="title-ch">多囊卵巢综合征(PCOS)预测计算工具</span>
              <div class="title-en small-en">Calculation Tool For Polycystic Ovarian Syndrome(PCOS)</div>
            </h2>
            <el-form :model="form" label-position="top" class="pcos-form">
              <el-form-item>
                <template #label><span class="label-en">AMH</span> <span class="label-ch">抗缪勒管激素</span></template>
                <el-input v-model="form.amh" type="number" placeholder="例如 5.2"><template #append>ng/ml</template></el-input>
              </el-form-item>
              <el-form-item>
                <template #label><span class="label-en">Menstrual cycle days</span> <span class="label-ch">月经周期天数</span></template>
                <div class="cycle-input-group">
                  <el-input v-model="form.menstrual_start" type="number" placeholder="例如 28" />
                  <span>-</span>
                  <el-input v-model="form.menstrual_end" type="number" placeholder="例如 35" />
                  <span class="unit-text">天(days)</span>
                </div>
              </el-form-item>
              <el-form-item>
                <template #label><span class="label-en">BMI</span> <span class="label-ch">体重指数</span></template>
                <el-input v-model="form.bmi" type="number" placeholder="例如 22.5"><template #append>kg/㎡</template></el-input>
              </el-form-item>
              <el-form-item>
                <template #label><span class="label-en">Androstenedione</span> <span class="label-ch">雄烯二酮</span></template>
                <el-input v-model="form.androstenedione" type="number" placeholder="例如 8.0"><template #append>nmol/L</template></el-input>
              </el-form-item>
              <el-form-item class="form-buttons">
                <el-button type="info" @click="showBatchModal = true" size="large">
                  <span class="btn-ch">批量计算</span> <span class="btn-en">Batch</span>
                </el-button>
                <el-button type="primary" @click="calculatePCOS" :loading="calculating" size="large">
                  <span class="btn-ch">点击计算</span> <span class="btn-en">Calculate</span>
                </el-button>
              </el-form-item>
            </el-form>
          </div>

          <div class="results-grid">
            <div class="card-style result-card probability-card">
              <h3 class="section-title">
                <!-- <img src="/logo.png" alt="logo" class="title-logo-small"> -->
                <span class="title-ch">患病概率</span><span class="title-en small-en">PCOS Probability</span>
              </h3>
              <div class="probability-chart-container">
                <v-chart :option="probabilityOption" autoresize class="echart-instance" />
              </div>
              <div v-if="result" class="probability-summary">
                <p>您的患病概率为 <strong>{{ result.risk_percentage.toFixed(2) }}%</strong></p>
                <p>属于 <strong :class="currentRiskInfo.colorClass">{{ currentRiskInfo.text }}</strong></p>
              </div>
               <div v-else class="probability-summary-placeholder">
                <p>请输入指标后点击计算</p>
              </div>
            </div>

            <div class="card-style result-card risk-stratification-card">
              <h3 class="section-title">
                <!-- <img src="/logo.png" alt="logo" class="title-logo-small"> -->
                <span class="title-ch">危险分层模型</span><span class="title-en small-en">Risk Group</span>
              </h3>
              <div class="risk-display-area">
                <div class="risk-scale">
                  <div v-for="level in riskScaleLevels" :key="level.p" class="risk-scale-segment">
                    <span class="risk-percentage-label">{{ level.p }}%</span>
                    <div class="risk-bar" :class="`bar-${level.c}`"></div>
                  </div>
                </div>
                <div class="risk-result-box-container">
                    <div v-if="result" class="risk-result-box" :class="currentRiskInfo.colorClass">
                        <div class="result-box-title-ch">检查结果</div>
                        <div class="result-box-value">{{ result.risk_level }}</div>
                        <div class="result-box-title-en">Test Result</div>
                        <div class="result-box-value-en">{{ currentRiskInfo.english }}</div>
                    </div>
                    <div v-else class="risk-result-box-placeholder">
                        等待计算结果
                    </div>
                  <div class="risk-legend">
                    <div><span class="legend-color legend-high"></span> 高危组 High-Risk</div>
                    <div><span class="legend-color legend-medium"></span> 中危组 Medium-Risk</div>
                    <div><span class="legend-color legend-low"></span> 低危组 Low-Risk</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <aside class="stats-sidebar">
          <div class="card-style stats-card">
            <h3 class="section-title">
              <!-- <img src="/logo.png" alt="logo" class="title-logo-small"> -->
              <span class="title-ch">使用量排名</span><span class="title-en small-en">Usage Ranking</span>
            </h3>
            <ul class="ranking-list">
              <li v-for="(item, index) in usageStats" :key="`usage-${index}`" class="ranking-item">
                <span class="rank">NO.{{ index + 1 }}</span>
                <span class="location">{{ item.location }}</span>
                <span class="count">{{ item.count }}</span>
                <div class="progress-bar-bg">
                  <div class="progress-bar-fill" :style="{ width: item.percentage + '%' }"></div>
                </div>
              </li>
               <el-empty v-if="!usageStats.length" description="暂无数据" :image-size="50"></el-empty>
            </ul>
          </div>
          <div class="card-style stats-card">
            <h3 class="section-title">
              <!-- <img src="/logo.png" alt="logo" class="title-logo-small"> -->
              <span class="title-ch">访问量排名</span><span class="title-en small-en">Visit Ranking</span>
            </h3>
            <ul class="ranking-list">
              <li v-for="(item, index) in visitStats" :key="`visit-${index}`" class="ranking-item">
                <span class="rank">NO.{{ index + 1 }}</span>
                <span class="location">{{ item.location }}</span>
                <span class="count">{{ item.count }}</span>
                <div class="progress-bar-bg">
                  <div class="progress-bar-fill" :style="{ width: item.percentage + '%' }"></div>
                </div>
              </li>
              <el-empty v-if="!visitStats.length" description="暂无数据" :image-size="50"></el-empty>
            </ul>
          </div>
        </aside>
      </div>
    </main>

    <el-dialog v-model="showBatchModal" title="批量计算 PCOS 概率" width="700px" top="5vh">
      <div class="batch-modal-content">
        <el-button @click="downloadTemplate" type="primary" icon="Download" style="margin-bottom: 20px;">
          下载Excel模板
        </el-button>
        
        <el-upload
          ref="uploadFileInstance"
          class="batch-uploader"
          drag
          action="#" 
          :auto-upload="false"
          :on-change="handleFileChangeForBatch"
          :before-upload="beforeUploadBatch"
          :limit="1"
          accept=".xlsx,.xls"
        >
          <!-- <img src="/upload.png" class="upload-icon-img" alt="upload"> -->
          <el-icon class="el-icon--upload"><upload-filled /></el-icon>
          <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
          <template #tip>
            <div class="el-upload__tip">只能上传 .xlsx/.xls 文件，且不超过2MB</div>
          </template>
        </el-upload>

        <el-button 
          v-if="uploadedFile" 
          @click="processBatchCalculation" 
          type="success" 
          :loading="batchCalculating"
          style="margin-top: 20px;"
          icon="Promotion"
        >
          开始处理批量计算
        </el-button>

        <div v-if="batchResults.length > 0" class="batch-results-table">
          <h4>计算结果:</h4>
          <el-table :data="batchResults" stripe height="250">
            <el-table-column prop="index" label="行号" width="70" />
            <el-table-column prop="probability" label="患病概率" width="120"/>
            <el-table-column prop="risk_level" label="风险等级" width="100"/>
            <el-table-column prop="status" label="状态" width="80">
                <template #default="scope">
                    <el-tag :type="scope.row.status === '成功' ? 'success' : 'danger'">{{ scope.row.status }}</el-tag>
                </template>
            </el-table-column>
            <el-table-column prop="error" label="备注/错误"/>
          </el-table>
        </div>
      </div>
    </el-dialog>

    <footer class="app-footer">
      <div class="container">
        <p>© {{ new Date().getFullYear() }} PCOS Screening Tool. All rights reserved.</p>
        <p><span class="departfont">国家妇产疾病临床医学研究中心 北京大学第三医院妇产科生殖医学中心</span></p>
        <p><span class="engfont">Center for Reproductive Medicine, Department of Obstetrics and Gynecology, National Clinical Research Center for Obstetrics and Gynecology, Peking University Third Hospital, Beijing, China</span></p>
      </div>
    </footer>
  </div>
</template>

<style>
/* Global Styles & Resets (inspired by original and modernized) */
:root {
  --primary-color: #4A90E2; /* A calm blue */
  --secondary-color: #50E3C2; /* A vibrant teal/green */
  --accent-color: #F5A623; /* Orange for accents */
  --danger-color: #D0021B; /* Red for high risk/errors */
  --warning-color: #F8E71C; /* Yellow for medium risk/warnings */
  --success-color: #7ED321; /* Green for low risk/success */
  
  --text-color: #333;
  --text-light: #555;
  --text-extra-light: #777;
  --bg-color: #f4f7f9; /* Light gray background */
  --card-bg: #ffffff;
  --border-color: #e0e6ed;
  
  --header-bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%); /* Original's purple gradient */
  --header-text-color: #ffffff;

  --font-main: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

body {
  margin: 0;
  font-family: var(--font-main);
  background-color: var(--bg-color);
  color: var(--text-color);
  line-height: 1.6;
}

#app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.container {
  width: 90%;
  max-width: 1300px; /* Increased max-width for better large screen layout */
  margin: 0 auto;
  padding: 0 15px;
}

/* Card Style */
.card-style {
  background-color: var(--card-bg);
  border-radius: 12px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
  padding: 25px;
  margin-bottom: 25px;
}

/* Section Titles (like original's card headers) */
.section-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 1.3rem;
  font-weight: 600;
  color: var(--text-color);
  margin-bottom: 20px;
  padding-bottom: 15px;
  border-bottom: 1px solid var(--border-color);
}
.section-title .title-logo-small { width: 28px; height: 28px; }
.section-title .title-ch { font-weight: 600; }
.section-title .title-en { font-size: 0.9rem; color: var(--text-light); margin-left: 8px; }
.section-title .small-en { font-size: 0.8rem; color: var(--text-extra-light); font-weight: 400; }


/* Header */
.app-header {
  background: var(--header-bg);
  color: var(--header-text-color);
  padding: 20px 0;
  margin-bottom: 30px;
}
.header-content { display: flex; align-items: center; gap: 15px; }
.header-logo { height: 50px; }
.header-title h1 { font-size: 1.8rem; margin: 0 0 5px 0; font-weight: 600;}
.header-title p { font-size: 0.9rem; margin: 0; opacity: 0.9; }

/* Map Section */
.map-section h2 { border-bottom: none; margin-bottom: 10px; }
.map-container { height: 450px; border-radius: 8px; overflow: hidden; background-color: #eaf2f8; } /* Added bg for map */
.echart-instance { width: 100%; height: 100%; }


/* Content Grid Layout */
.content-grid {
  display: grid;
  grid-template-columns: 2fr 1fr; /* Adjust ratio as needed, e.g., 60% 40% */
  gap: 30px;
}

/* Calculator Section */
.form-card .section-title { border-bottom: none; padding-bottom: 0; margin-bottom: 15px; }
.pcos-form .el-form-item { margin-bottom: 18px; }
.pcos-form .el-form-item__label .label-en { font-weight: 600; color: var(--text-color); }
.pcos-form .el-form-item__label .label-ch { font-size: 0.9em; color: var(--text-light); }
.cycle-input-group { display: flex; align-items: center; gap: 10px; }
.cycle-input-group .unit-text { margin-left: 5px; color: var(--text-light); font-size: 0.9em; }
.form-buttons { margin-top: 25px; }
.form-buttons .el-button .btn-ch { font-weight: 500; }
.form-buttons .el-button .btn-en { font-size: 0.8em; opacity: 0.8; margin-left: 4px; }

/* Results Grid */
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); /* Responsive columns */
  gap: 25px;
  margin-top: 25px;
}
.probability-card .section-title, .risk-stratification-card .section-title {
  margin-bottom: 15px; /* Reduced margin for titles inside result cards */
}
.probability-chart-container { height: 220px; margin-bottom:10px;} /* Adjusted height */
.probability-summary { text-align: center; font-size: 1rem; color: var(--text-light); }
.probability-summary strong { color: var(--text-color); }
.probability-summary-placeholder { text-align: center; color: #aaa; padding: 20px; font-style: italic;}


/* Risk Stratification Card */
.risk-display-area { display: flex; gap: 20px; align-items: flex-start; margin-top:10px; }
.risk-scale { flex-basis: 120px; display: flex; flex-direction: column-reverse; gap: 2px; }
.risk-scale-segment { display: flex; align-items: center; gap: 8px; }
.risk-percentage-label { font-size: 0.75rem; color: var(--text-extra-light); width: 30px; text-align: right; }
.risk-bar { height: 12px; flex-grow: 1; border-radius: 3px; }
.bar-high { background-color: var(--danger-color); }
.bar-medium { background-color: var(--warning-color); }
.bar-low { background-color: var(--success-color); }

.risk-result-box-container { flex-grow: 1; display: flex; flex-direction: column; align-items: center; }
.risk-result-box {
  padding: 15px; border-radius: 8px; text-align: center; width: 100%; max-width: 200px;
  margin-bottom: 15px; border: 2px solid;
}
.risk-result-box-placeholder { color: #aaa; padding: 20px; font-style: italic; text-align: center; border: 1px dashed #ccc; border-radius: 8px; }

.result-box-title-ch { font-weight: 600; margin-bottom: 5px; font-size: 0.9rem; }
.result-box-value { font-size: 1.4rem; font-weight: 700; margin-bottom: 8px; }
.result-box-title-en { font-size: 0.8rem; color: var(--text-light); }
.result-box-value-en { font-size: 0.9rem; font-weight: 500; color: var(--text-light); }

.high-risk { border-color: var(--danger-color); background-color: rgba(208, 2, 27, 0.05); }
.high-risk .result-box-value { color: var(--danger-color); }
.medium-risk { border-color: var(--warning-color); background-color: rgba(248, 231, 28, 0.08); }
.medium-risk .result-box-value { color: #b3a100; } /* Darker yellow for text */
.low-risk { border-color: var(--success-color); background-color: rgba(126, 211, 33, 0.05); }
.low-risk .result-box-value { color: var(--success-color); }

.risk-legend { font-size: 0.8rem; display: flex; flex-direction: column; gap: 5px; align-self: flex-start; margin-top: 5px; }
.legend-color { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: middle;}
.legend-high { background-color: var(--danger-color); }
.legend-medium { background-color: var(--warning-color); }
.legend-low { background-color: var(--success-color); }

/* Stats Sidebar */
.stats-sidebar { display: flex; flex-direction: column; gap: 25px; }
.ranking-list { list-style-type: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 15px; }
.ranking-item { font-size: 0.9rem; display: grid; grid-template-columns: auto 1fr auto; gap: 5px 10px; align-items: center; }
.ranking-item .rank { font-weight: 600; color: var(--primary-color); }
.ranking-item .location { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text-light);}
.ranking-item .count { font-weight: 500; color: var(--text-color); }
.progress-bar-bg {
  grid-column: 1 / -1; /* Span all columns */
  height: 12px; background-color: #eef1f5; border-radius: 6px; overflow: hidden;
}
.progress-bar-fill { height: 100%; background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)); border-radius: 6px; transition: width 0.5s ease-in-out; }

/* Batch Modal */
.batch-modal-content { display: flex; flex-direction: column; gap: 20px; }
.batch-uploader .el-upload-dragger { padding: 20px; /* Smaller padding */ }
.upload-icon-img { width: 50px; height: 50px; margin-bottom: 10px; }
.batch-results-table { margin-top: 15px; }
.batch-results-table h4 { margin-bottom: 10px; font-size: 1.1rem; }

/* Footer */
.app-footer {
  background-color: #2c3e50; /* Dark footer */
  color: #bdc3c7; /* Light gray text */
  padding: 30px 0;
  text-align: center;
  font-size: 0.85rem;
  margin-top: auto; /* Pushes footer to bottom */
}
.app-footer p { margin: 5px 0; }
.app-footer .departfont { font-weight: 500; color: #ecf0f1; }
.app-footer .engfont { font-size: 0.8rem; opacity: 0.8; }

/* Element Plus Overrides (if needed) */
.el-form--label-top .el-form-item__label { padding-bottom: 4px; line-height: 1.4; }
.el-input-group__append { background-color: #f5f7fa; color: var(--text-light); }

/* Responsive Adjustments */
@media (max-width: 1024px) {
  .content-grid {
    grid-template-columns: 1fr; /* Stack main content and sidebar */
  }
  .stats-sidebar {
    order: -1; /* Optionally move stats to top on smaller screens */
  }
   .map-container { height: 350px; }
}

@media (max-width: 768px) {
  .container { width: 95%; }
  .header-title h1 { font-size: 1.5rem; }
  .header-title p { font-size: 0.8rem; }
  .section-title { font-size: 1.15rem; }
  .map-container { height: 300px; }
  .results-grid { grid-template-columns: 1fr; } /* Stack result cards */
  .risk-display-area { flex-direction: column; align-items: stretch;}
  .risk-scale { flex-basis: auto; margin-bottom: 15px; }
  .risk-result-box-container { align-items: center; }
  .form-buttons { flex-direction: column; gap: 10px; }
  .form-buttons .el-button { width: 100%; }
}

</style>