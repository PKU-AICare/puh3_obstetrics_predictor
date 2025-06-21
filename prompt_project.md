please update the following project with Vue 3 setup syntax and fastapi (in single file is better -- e.g. backend.py, App.vue.

- 其主要功能：根据用户输入的信息，计算对应的预测概率——网站的核心展示内容预测结果的可视化统计图，在网站的核心位置展示14个疾病预测结果的饼图（圆圈状饼图，中间展示概率），图标题展示疾病的名字。即，需要改成用圆圈饼状图的形式，而不是打表格。
- 并能根据用户IP，统计各地区访问次数和使用次数
    - 对于国内的IP地址，精确到省或直辖市级（如北京市、上海市、浙江省……）；对于海外的IP地址，则精确到国家或地区级
    - 基于IP地址的地理位置，需要实现两个功能，①总访问频次，累计有多人少使用了这个网站，②有多少个国家的用户使用了这个网站，统计国家总数，③按地区访问次数排序，只展示国家维度，即给个国家的访问次数排名，展示排名前10的国家
- 注意要有IP记录的功能，先把地图可视化的功能砍掉 (不要有地图的功能区域)
- 网站多个孕妇预测的部分，需要增加一个下拉框，能实现选择哪个孕妇，可以可视化对应孕妇的预测结果。
- 设计出中英双语，搞出多语言的支持（可以使用i18n）。
- 不需要在系统上手动输入各个特征，而是实现上传excel （单个patient的xlsx (excel file，其有2个sheets: '数据上传表_实验室检查' & '数据上传表_基线特征') / 批量预测（上传形式是zip包（也需要可以支持rar, 7z 等压缩包支持），里面有很多个patient的xlsx files））。Excel template从前端下载，给出Excel template的设计（它有2 sheets，需要中英双语，不懂中文的人也能看懂该表，不需要展示变量名，然后前后端的代码能对应得上）。对于单个patient的预测结果要能直接展示在前端中，而批量预测的则支持预测完成后 下载导出预测结果的excel文件 (单个上传的excel的文件名是patient_id，预测结果导出对单个/批量都要有，输出的形式都是excel 只不过是单个patient/多个patients。rows are patients predicted results, columns are predicted result.)
- 要以最高质量去完成！样式styles要全面更新，需要modern 现代化、美观、和谐、大方!
- 数据的持久化可以先用sqlite
- fix and return updated complete code（我使用uv for python development, and I'm using fnm, pnpm for node environment)

backend/.python-version

```
3.12

```

backend/backend.py

```python
import io
import json
import math
import os
import zipfile
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import requests
from fastapi import (Depends, FastAPI, File, HTTPException, Request,
                   UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String,
                        Text, create_engine, func)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# --- Project Configuration ---
PROJECT_TITLE = "Assessment of Pregnancy-Related Disease Risks in Repeat Pregnancies"
SQLALCHEMY_DATABASE_URL = "sqlite:///./predictions.db"

# --- Database Setup ---
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models ---
class PredictionRecord(Base):
    __tablename__ = "prediction_records"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    ip_address = Column(String)
    location = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    results_json = Column(Text)

class VisitRecord(Base):
    __tablename__ = "visit_records"
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String, index=True)
    location = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# --- Pydantic Models ---
class PredictionResultItem(BaseModel):
    disease_abbr: str
    disease_name_cn: str
    probability: float

class SinglePredictionResponse(BaseModel):
    patient_id: str
    predictions: List[PredictionResultItem]

class LocationStat(BaseModel):
    location: str
    count: int

# --- Disease Formulas & Mappings ---
DISEASE_FORMULAS = {
    "DM": {"name_cn": "妊娠合并糖尿病", "coeffs": {"age_1st": 0.1419173920, "weight_gain_1st": -0.1698330926, "pre_bmi_1st": 0.0938179096, "nation_1stminority": 0.7494179418, "gestational_age_1st": -0.4444326646, "birth_weight_1st": 0.0009155465, "NICU_1styes": -16.2611736194, "PROM_1styes": -0.6551172461, "PE_1styes": 0.8977422306, "PT_1st_f": -0.3014261392, "Fib_1st_f": 0.6122690819, "ALP_1st_f": 0.0338571978, "ALP_1st_t": -0.0104502311, "Urea_1st_t": 0.5477669654, "RBC_1st_t": 0.9667897398}},
    "GDM": {"name_cn": "妊娠期糖尿病", "coeffs": {"age_1st": 0.0677375675, "pre_bmi_1st": 0.0303027085, "gestational_age_1st": -0.1469573905, "birth_weight_1st": 0.0003267695, "PROM_1styes": -0.1539029255, "ALT_1st_f": -0.0055781165, "TB_1st_f": -0.0215863388, "UA_1st_f": 0.0023808548, "P_1st_f": -0.6177690664, "WBC_1st_f": 0.0449929394, "RBC_1st_f": 0.3360527792, "AST_1st_t": -0.0232362771, "TP_1st_t": -0.0207373228, "Urea_1st_t": 0.1726151069, "NE_1st_t": -0.0476341578, "Hb_1st_t": 0.0175449679}},
    "HDP": {"name_cn": "妊娠期高血压疾病", "coeffs": {"weight_gain_1st": 0.054062796, "pre_bmi_1st": 0.193458307, "birth_weight_1st": -0.001137972, "NICU_1styes": -3.256899663, "PE_1styes": 3.858058925, "PT_1st_f": -0.300597209, "ALP_1st_f": 0.015503811, "DB_1st_f": 0.086570104, "WBC_1st_f": 0.077348983, "LY_pctn_1st_f": 0.300798922, "NE_pctn_1st_f": 0.274686573, "MO_pctn_1st_f": 0.207167696, "Hb_1st_f": 0.015825290, "PLT_1st_f": 0.002672225, "TT_1st_t": 0.167903201, "TBA_1st_t": -0.106703815, "Urea_1st_t": 0.458734254, "UA_1st_t": 0.004494712, "MO_pctn_1st_t": -0.143291082}},
    "HYPOT": {"name_cn": "妊娠合并甲状腺功能减退", "coeffs": {"age_1st": 0.0521641548, "gender_1stmale": -0.3170674447, "DM_1styes": -13.5156229492, "PROM_1styes": -0.3242030304, "AST_1st_f": 0.0204702243, "TP_1st_f": 0.0449350360, "Cr_1st_f": 0.0006603378, "TSH_1st_f": 0.7445658561, "Cr_1st_t": 0.0215867070}},
    "LBW": {"name_cn": "低出生体重儿", "coeffs": {"age_1st": 0.146896121, "pre_bmi_1st": 0.034844950, "nation_1stminority": 0.365470305, "gestational_age_1st": -0.099539808, "NICU_1styes": -0.902257912, "P_1st_f": -0.774395884, "RBC_1st_f": 0.319913915, "UA_1st_t": 0.002766374, "RDW_CV_1st_t": 0.097493680}},
    "LGA": {"name_cn": "大于胎龄儿", "coeffs": {"gestational_age_1st": -0.156033061, "birth_weight_1st": -0.001687150, "hysteromyoma_1styes": 0.517217070, "NICU_1styes": -2.457124930, "DM_1styes": 1.277533141, "PE_1styes": -1.937494470, "Placenta_Previa_1styes": 0.806095894, "TP_1st_f": 0.058642857, "MO_1st_f": 1.780582862, "MCV_1st_f": -0.044757368, "TT_1st_t": 0.175639553, "UA_1st_t": 0.003122707, "NE_1st_t": 0.121070269, "BAS_pctn_1st_t": 1.547644475}},
    "MYO": {"name_cn": "妊娠合并子宫肌瘤", "coeffs": {"weight_gain_1st": 0.029515956, "pre_bmi_1st": 0.050811565, "gestational_age_1st": -0.349786153, "birth_weight_1st": 0.002634555, "gender_1stmale": -0.354175523, "GDM_1styes": 0.234191248, "PT_1st_f": -0.137972195, "Fib_1st_f": 0.160828605, "UA_1st_f": 0.002396116, "Urea_1st_t": -0.141457178, "P_1st_t": 0.560103021, "RDW_SD_1st_t": 0.020350271}},
    "NICU": {"name_cn": "新生儿重症监护室", "coeffs": {"gestational_age_1st": -0.1725791002, "birth_weight_1st": -0.0007617624, "birth_length_1st": 0.1266719889, "hysteromyoma_1styes": 0.3451214024, "Delivery_method_1styes": 0.2854435335, "NICU_1styes": -14.4775925471, "GDM_1styes": 0.3088215468, "PPH_1styes": -0.4588597948, "ALP_1st_f": 0.0111946657, "DB_1st_f": -0.1173074496, "PT_1st_t": 0.4375821739, "TT_1st_t": 3.4309459560, "PTT_1st_t": -48.8590837230, "TP_1st_t": 0.0910124136, "ALB_1st_t": -0.1210340278, "RDW_SD_1st_t": -0.0412889009}},
    "PE": {"name_cn": "子痫前期", "coeffs": {"pre_bmi_1st": 0.1588087979, "birth_weight_1st": -0.0007503861, "gender_1stmale": -0.3531043810, "NICU_1styes": -2.0281813488, "PE_1styes": 2.2003865692, "PPH_1styes": -0.6496623637, "PT_1st_f": -0.4299248270, "ALP_1st_f": 0.0172409639, "BAS_pctn_1st_f": -1.4708132939, "PCT_1st_f": 3.4370473770, "Urea_1st_t": 0.3385603590}},
    "PP": {"name_cn": "前置胎盘", "coeffs": {"age_1st": 0.05105457, "birth_length_1st": 0.08991213, "Delivery_method_1styes": 0.82968740, "DM_1styes": -13.78513157, "PE_1styes": 0.70760017, "Placenta_Previa_1styes": 0.45416943, "APTT_1st_t": 0.07662945, "PTT_1st_t": -2.46431367, "ALB_1st_t": 0.05806045, "TB_1st_t": -0.07309937, "BAS_1st_t": -11.71195544, "RDW_CV_1st_t": 0.09310345}},
    "PPH": {"name_cn": "产后出血", "coeffs": {"age_1st": 0.0304508163, "pre_bmi_1st": 0.0399908747, "nation_1stminority": -0.3793425995, "gestational_age_1st": -0.1463766559, "birth_weight_1st": 0.0004501316, "gender_1stmale": -0.2668906219, "Delivery_method_1styes": 0.2848715084, "PPH_1styes": 0.6550008395, "Fib_1st_t": -0.2234355038, "ALP_1st_t": -0.0035479574, "PCT_1st_t": -3.7543409206}},
    "PROM": {"name_cn": "胎膜早破", "coeffs": {"pre_bmi_1st": -0.0277162724, "gestational_age_1st": -0.1455379006, "birth_weight_1st": 0.0003261704, "hysteromyoma_1styes": 0.3963949921, "Delivery_method_1styes": 0.8802072975, "PROM_1styes": 0.5543519130, "PE_1styes": -0.5436138359, "BAS_pctn_1st_t": 0.8631150211}},
    "PTB": {"name_cn": "早产", "coeffs": {"age_1st": 0.046681151, "pre_bmi_1st": 0.039969249, "gestational_age_1st": -0.474764824, "hysteromyoma_1styes": 0.518263100, "Delivery_method_1styes": -0.386175618, "NICU_1styes": -2.759545470, "PE_1styes": -0.906306622, "MCV_1st_f": -0.067096292, "PCT_1st_f": 3.676376419, "UA_1st_t": 0.002516177, "WBC_1st_t": 0.091190784, "BAS_pctn_1st_t": 1.204259664}},
    "SGA": {"name_cn": "小于胎龄儿", "coeffs": {"pre_bmi_1st": -0.064826410, "gestational_age_1st": 0.354456399, "birth_weight_1st": -0.002978985, "PE_1styes": -1.244787130, "Ca_1st_f": 2.337452681, "PCT_1st_f": 3.658695711}},
}

# --- FastAPI App Initialization ---
app = FastAPI(
    title=PROJECT_TITLE,
    description="An API for assessing pregnancy-related disease risks.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Dependency & Helpers ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_location_from_ip(ip: str) -> str:
    if not ip or ip in ("127.0.0.1", "::1") or ip.startswith(("192.168.", "10.", "172.16.")):
        return "Local Network"
    try:
        response = requests.get(f"https://ipapi.co/{ip}/json/", timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("error"): return "Unknown"

        city = data.get("city", "")
        region = data.get("region", "")
        country = data.get("country_name", "Unknown")

        if country == "China":
            return f"中国 {region}".strip() if region else "中国"

        return ", ".join(filter(None, [city, region, country]))
    except requests.exceptions.RequestException:
        return "Unknown"

def parse_excel_to_features(contents: bytes) -> Dict[str, float]:
    try:
        xls = pd.ExcelFile(io.BytesIO(contents))
        required_sheets = ['数据上传表_基线特征', '数据上传表_实验室检查']
        if not all(sheet in xls.sheet_names for sheet in required_sheets):
            raise HTTPException(status_code=400, detail=f"Excel must contain sheets: {', '.join(required_sheets)}")

        df_baseline = pd.read_excel(xls, sheet_name=required_sheets[0])
        df_lab = pd.read_excel(xls, sheet_name=required_sheets[1])
        features = {}

        # Process baseline features (Column positions: 1 for value, 3 for formula variable)
        for _, row in df_baseline.iterrows():
            if len(row) > 3 and pd.notna(row.iloc[1]) and pd.notna(row.iloc[3]):
                formula_var = str(row.iloc[3])
                try:
                    features[formula_var] = float(row.iloc[1])
                except (ValueError, TypeError):
                    pass

        # Process lab features (Value and VarName are paired by index)
        # Pairs: (Val_f: 3, Var_f: 6), (Val_s: 4, Var_s: 7), (Val_t: 5, Var_t: 8)
        periods = [(3, 6), (4, 7), (5, 8)]
        for _, row in df_lab.iterrows():
            for val_idx, var_idx in periods:
                if len(row) > var_idx and pd.notna(row.iloc[val_idx]) and pd.notna(row.iloc[var_idx]):
                    formula_var = str(row.iloc[var_idx])
                    if formula_var != 'nan':
                        try:
                            features[formula_var] = float(row.iloc[val_idx])
                        except (ValueError, TypeError):
                            pass
        return features
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error parsing Excel file: {e}")

def calculate_probabilities(features: Dict[str, float]) -> List[Dict[str, Any]]:
    results = []
    for abbr, data in DISEASE_FORMULAS.items():
        logit = 0.0
        for var, coeff in data["coeffs"].items():
            logit += coeff * features.get(var, 0.0)
        try:
            probability = 1 / (1 + math.exp(-logit))
        except OverflowError:
            probability = 0.0 if logit < 0 else 1.0
        results.append({
            "disease_abbr": abbr,
            "disease_name_cn": data["name_cn"],
            "probability": probability
        })
    return results

# --- Event Handlers & Middleware ---
@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)

@app.middleware("http")
async def log_visits_middleware(request: Request, call_next):
    if request.method == "OPTIONS" or any(p in str(request.url) for p in ["/docs", "/openapi.json", ".ico"]):
        return await call_next(request)

    client_ip = request.headers.get("x-forwarded-for") or request.client.host
    # Only log a visit once per session (or every few hours) could be an improvement.
    # For now, log every non-API request.
    if "/api/" not in str(request.url):
        with SessionLocal() as db:
            try:
                location = get_location_from_ip(client_ip)
                visit = VisitRecord(ip_address=client_ip, location=location)
                db.add(visit)
                db.commit()
            except Exception:
                db.rollback()
    return await call_next(request)

# --- API Endpoints ---
@app.post("/api/predict-single", response_model=SinglePredictionResponse)
async def predict_single(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .xlsx file.")

    patient_id = os.path.splitext(file.filename)[0]
    contents = await file.read()

    features = parse_excel_to_features(contents)
    predictions = calculate_probabilities(features)

    # Log the successful prediction
    client_ip = request.headers.get("x-forwarded-for") or request.client.host
    location = get_location_from_ip(client_ip)
    results_for_db = {p["disease_abbr"]: p["probability"] for p in predictions}
    db_record = PredictionRecord(
        patient_id=patient_id, ip_address=client_ip, location=location, results_json=json.dumps(results_for_db)
    )
    db.add(db_record)
    db.commit()

    return SinglePredictionResponse(patient_id=patient_id, predictions=predictions)

@app.post("/api/predict-batch")
async def predict_batch(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .zip file.")

    contents = await file.read()
    client_ip = request.headers.get("x-forwarded-for") or request.client.host
    location = get_location_from_ip(client_ip)
    all_results = []

    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as z:
            for filename in z.namelist():
                if filename.endswith('.xlsx') and not filename.startswith('__MACOSX'):
                    patient_id = os.path.splitext(os.path.basename(filename))[0]
                    with z.open(filename) as xlsx_file:
                        features = parse_excel_to_features(xlsx_file.read())
                        predictions = calculate_probabilities(features)

                        row_data = {"patient_id": patient_id}
                        for p in predictions:
                           row_data[f'{p["disease_abbr"]}_prob'] = p["probability"]
                        all_results.append(row_data)

                        results_for_db = {p["disease_abbr"]: p["probability"] for p in predictions}
                        db.add(PredictionRecord(patient_id=patient_id, ip_address=client_ip, location=location, results_json=json.dumps(results_for_db)))
        db.commit()
    except zipfile.BadZipFile:
        db.rollback()
        raise HTTPException(status_code=400, detail="Invalid or corrupted ZIP file.")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Batch processing error: {e}")

    if not all_results:
        raise HTTPException(status_code=400, detail="No valid .xlsx files found in the ZIP archive.")

    df_results = pd.DataFrame(all_results)
    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        df_results.to_excel(writer, index=False, sheet_name="Batch Predictions")
    output_buffer.seek(0)

    filename = f"batch_prediction_results_{datetime.now().strftime('%Y%m%d%H%M')}.xlsx"
    return StreamingResponse(
        output_buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/api/stats/usage", response_model=List[LocationStat])
async def get_usage_stats(db: Session = Depends(get_db)):
    query = db.query(
        PredictionRecord.location, func.count(PredictionRecord.id).label("count")
    ).group_by(PredictionRecord.location).order_by(func.count(PredictionRecord.id).desc()).limit(10).all()
    return [LocationStat(location=loc, count=count) for loc, count in query if loc]

@app.get("/api/stats/visits", response_model=List[LocationStat])
async def get_visit_stats(db: Session = Depends(get_db)):
    query = db.query(
        VisitRecord.location, func.count(VisitRecord.id).label("count")
    ).group_by(VisitRecord.location).order_by(func.count(VisitRecord.id).desc()).limit(10).all()
    return [LocationStat(location=loc, count=count) for loc, count in query if loc]

@app.get("/api/download-template")
async def download_template():
    # --- Baseline Features Sheet ---
    baseline_data = {
        '中文名称 (Chinese Name)': ['年龄', '孕期体重增长', '孕前BMI', '民族', '甲状腺功能减退症', '分娩孕周', '出生体重', '出生身长', '新生儿性别', '子宫肌瘤', '分娩方式', '新生儿转NICU', '孕前糖尿病', '胎膜早破', '妊娠期糖尿病', '子痫前期', '前置胎盘', '产后出血'],
        '输入数值 (Value)': ['' for _ in range(18)],
        '英文名称 (English Name)': ['Age', 'Weight gain during pregnancy', 'Pre-pregnancy BMI', 'Ethnicity (1 for Han, 2 for Minority)', 'Hypothyroidism (1 for Yes, 0 for No)', 'Gestational age (e.g., 38+1/7 -> 38.142)', 'Birth weight (g)', 'Birth length (cm)', 'Gender of newborn (1 for Male, 0 for Female)', 'Hysteromyoma (1 for Yes, 0 for No)', 'Delivery method (1 for Cesarean, 0 for Vaginal)', 'NICU admission (1 for Yes, 0 for No)', 'Pre-gestational DM (1 for Yes, 0 for No)', 'PROM (1 for Yes, 0 for No)', 'GDM (1 for Yes, 0 for No)', 'Pre-eclampsia (PE) (1 for Yes, 0 for No)', 'Placenta previa (1 for Yes, 0 for No)', 'Postpartum hemorrhage (PPH) (1 for Yes, 0 for No)'],
        'formula_variable': ['age_1st', 'weight_gain_1st', 'pre_bmi_1st', 'nation_1st', 'Hypothyroidism_1st', 'gestational_age_1st', 'birth_weight_1st', 'birth_length_1st', 'gender_1st', 'hysteromyoma_1st', 'Delivery_method_1st', 'NICU_1st', 'DM_1st', 'PROM_1st', 'GDM_1st', 'PE_1st', 'Placenta_Previa_1st', 'PPH_1st']
    }
    df_baseline = pd.DataFrame(baseline_data)

    # --- Lab Results Sheet ---
    # FIX: Define the list of variables first to avoid UnboundLocalError
    lab_variables = ['PT','APTT','TT','Fib','ALT','AST','GGT','LDH','ALP','TP','ALB','GLB','TB','DB','TBA','PA','Urea','Cr','UA','CysC','B2MG','CO2','Na','K','CL','Ca','P','Mg','CK','CKMB','GLU','HbA1c','TCHO','TG','HDLC','LDLC','ApoA1','ApoB','Lpa','TSH','T4','T3','FT4','FT3','TPOAb','TGAb','TMA','CRP','USCRP','WBC','LY','NE','MO','BAS','EOS','LY_pctn','NE_pctn','MO_pctn','BAS_pctn','EOS_pctn','RBC','Hb','Hct','MCV','MCH','MCHC','RDW_CV','RDW_SD','PLT','MPV','PCT','PDW','m_dbp','m_sbp','m_nowweight2']

    lab_data = {
        '输入变量 (Variable)': lab_variables,
        '中文标签 (Chinese Label)': ['凝血酶原时间','活化部分凝血活酶','凝血酶时间','纤维蛋白原','丙氨酸氨基转移酶','天冬氨酸氨基转移酶','快速γ谷氨酰转肽酶','乳酸脱氢酶','碱性磷酸酶','总蛋白','白蛋白','球蛋白','总胆红素','直接胆红素','总胆汁酸','前白蛋白','快速尿素','肌酐','尿酸','胱抑素C','β2微球蛋白','快速总二氧化碳','钠','钾','氯','钙','磷','镁','肌酸激酶','肌酸激酶同工酶','葡萄糖','糖化血红蛋白A1c','总胆固醇','甘油三酯','高密度脂蛋白胆固醇','低密度脂蛋白胆固醇','载脂蛋白A1','载脂蛋白B','脂蛋白a','促甲状腺素','总甲状腺素','总三碘甲状腺原氨酸','游离甲状腺素','游离三碘甲状腺原氨酸','抗甲状腺过氧化物酶抗体','抗甲状腺球蛋白抗体','抗甲状腺微粒体抗体','快速C-反应蛋白','超敏C反应蛋白','白细胞','淋巴细胞绝对值','中性粒细胞绝对值','单核细胞绝对值','嗜碱性粒细胞','嗜酸性粒细胞','淋巴细胞百分数','嗜中性粒细胞百分比','单核细胞百分比','嗜碱性粒细胞百分比','嗜酸性粒细胞百分比','红细胞','血红蛋白','红细胞压积','平均红细胞体积','平均血红蛋白含量','平均血红蛋白浓度','红细胞分布宽度CV','红细胞分布宽度SD','血小板','平均血小板体积','血小板压积','血小板分布宽度','舒张压','收缩压','本次门诊的体重'],
        '英文标签 (English Label)': ['Prothrombin time','Activated partial thromboplastin time','Thrombin time','Fibrinogen','Alanine aminotransferase','Aspartate aminotransferase','Gamma-glutamyl transferase','Lactate dehydrogenase','Alkaline phosphatase','Total protein','Albumin','Globulin','Total bilirubin','Direct bilirubin','Total bile acid','Prealbumin','Urea','Creatinine','Uric acid','Cystatin C','β2-microglobulin','Total carbon dioxide','Sodium','Potassium','Chloride','Calcium','Phosphorus','Magnesium','Creatine kinase','Creatine kinase-MB','Glucose','Glycated hemoglobin A1c','Total cholesterol','Triglyceride','High-density lipoprotein cholesterol','Low-density lipoprotein cholesterol','Apolipoprotein A1','Apolipoprotein B','Lipoprotein(a)','Thyroid-stimulating hormone','Total thyroxine','Total triiodothyronine','Free thyroxine','Free triiodothyronine','Anti-thyroid peroxidase antibody','Anti-thyroglobulin antibody','Anti-thyroid microsomal antibody','C-reactive protein','High-sensitivity C-reactive protein','White blood cell','Lymphocyte absolute count','Neutrophil absolute count','Monocyte absolute count','Basophil','Eosinophil','Lymphocyte percentage','Neutrophil percentage','Monocyte percentage','Basophil percentage','Eosinophil percentage','Red blood cell','Hemoglobin','Hematocrit','Mean corpuscular volume','Mean corpuscular hemoglobin','Mean corpuscular hemoglobin concentration','Red cell distribution width-CV','Red cell distribution width-SD','Platelet','Mean platelet volume','Plateletcrit','Platelet distribution width','Diastolic blood pressure','Systolic blood pressure','Current body weight'],
        '早孕期值 (Early P.)': ['' for _ in range(len(lab_variables))],
        '中孕期值 (Mid P.)': ['' for _ in range(len(lab_variables))],
        '晚孕期值 (Late P.)': ['' for _ in range(len(lab_variables))],
    }
    # FIX: Add the derived columns after the dictionary is created
    lab_data['var_f'] = [f'{v}_1st_f' for v in lab_variables]
    lab_data['var_s'] = [f'{v}_1st_s' for v in lab_variables]
    lab_data['var_t'] = [f'{v}_1st_t' for v in lab_variables]

    df_lab = pd.DataFrame(lab_data)

    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        df_baseline.to_excel(writer, sheet_name='数据上传表_基线特征', index=False)
        df_lab.to_excel(writer, sheet_name='数据上传表_实验室检查', index=False)
    output_buffer.seek(0)

    return StreamingResponse(
        output_buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=prediction_template.xlsx"}
    )
```

backend/.gitignore

```
# ===================================================================
# .gitignore for the FastAPI Backend
# ===================================================================
# Most Python and IDE rules are handled by the root .gitignore file.
# This file is for files generated specifically by THIS backend application.

# SQLite Database
# -----------------------------------
# Explicitly ignore the database file generated by this app.
predictions.db
predictions.db-journal

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# UV
#   Similar to Pipfile.lock, it is generally recommended to include uv.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#uv.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock
#poetry.toml

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#pdm.lock
#   pdm stores project-wide configurations in .pdm.toml, but it is recommended to not include it
#   in version control.
#   https://pdm.fming.dev/latest/usage/project/#working-with-version-control
.pdm.toml
.pdm-python
.pdm-build/

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#  and can be added to the global gitignore or merged into this file.  For a more nuclear
#  option (not recommended) you can uncomment the following to ignore the entire idea folder.
#.idea/

# Abstra
# Abstra is an AI-powered process automation framework.
# Ignore directories containing user credentials, local state, and settings.
# Learn more at https://abstra.io/docs
.abstra/

# Visual Studio Code
#  Visual Studio Code specific template is maintained in a separate VisualStudioCode.gitignore
#  that can be found at https://github.com/github/gitignore/blob/main/Global/VisualStudioCode.gitignore
#  and can be added to the global gitignore or merged into this file. However, if you prefer,
#  you could uncomment the following to ignore the entire vscode folder
# .vscode/

# Ruff stuff:
.ruff_cache/

# PyPI configuration file
.pypirc

# Cursor
#  Cursor is an AI-powered code editor. `.cursorignore` specifies files/directories to
#  exclude from AI features like autocomplete and code analysis. Recommended for sensitive data
#  refer to https://docs.cursor.com/context/ignore-files
.cursorignore
.cursorindexingignore

```

backend/pyproject.toml

```toml
[project]
name = "puh3-obstetrics-predictor"
version = "0.1.0"
description = "Assessment of Pregnancy-Related Disease Risks in Repeat Pregnancies"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "aiofiles>=24.1.0",
    "fastapi[standard]>=0.115.12",
    "geoip2>=5.1.0",
    "openpyxl>=3.1.5",
    "pandas>=2.2.3",
    "python-multipart>=0.0.20",
    "requests>=2.32.3",
    "sqlalchemy>=2.0.41",
]

```

.gitignore

```
run.sh
result_frontend_backend.txt

# ===================================================================
# Root .gitignore for the PUH3 Obstetrics Predictor Project
# ===================================================================

# IDE and Editor Configuration
# -----------------------------------
.idea/
.vscode/
# Keep shared settings, ignore user-specific ones
!.vscode/settings.json
!.vscode/extensions.json
!.vscode/launch.json
*.swp

# Operating System Files
# -----------------------------------
.DS_Store
Thumbs.db
._*

# Log files
# -----------------------------------
*.log
npm-debug.log*
yarn-debug.log*
pnpm-debug.log*
lerna-debug.log*

# Environment Variables
# -----------------------------------
# Ignore all .env files, but keep the example template.
.env*
!.env.example

# Build and Dependency Directories
# -----------------------------------
/node_modules/
/build/
/dist/
/.output/
/.vite/

# Python Cache, Build, and Virtual Environment Files
# ----------------------------------------------------
# Byte-code
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
*.egg-info/
.eggs/
*.egg

# Virtual environments
.venv/
venv/
env/
.env/
ENV/

# Caches from common tools
.pytest_cache/
.mypy_cache/
.ruff_cache/
.tox/

# Database files
# -----------------------------------
*.db
*.sqlite
*.sqlite3

# Test and Coverage Reports
# -----------------------------------
htmlcov/
.coverage

```

run.sh

```bash
#!/bin/bash

# 定义输出文件名
OUTPUT_FILE="result_frontend_backend.txt"

# 在开始时清空输出文件
> "$OUTPUT_FILE"

# 查找所有符合条件的文件并处理
# -print0 和 read -d '' 是为了安全处理包含空格或特殊字符的文件名
find . \
    \( -path './.git' -o -path '*/__pycache__' -o -path '*/node_modules' -o -path '*/.venv' \) -prune \
    -o \
    \( -type f \
       ! -name '*.ico' \
       ! -name 'pnpm-lock.yaml' \
       ! -name 'uv.lock' \
       ! -name 'package-lock.json' \
       ! -name 'result_frontend_backend.txt' \
       ! -name '*.md' \
       -print0 \) \
| while IFS= read -r -d '' filepath; do

    # 去除文件路径开头的 './' 以获得更干净的相对路径
    clean_filepath="${filepath#./}"

    # 检查文件是否为空，如果为空则跳过
    if [ ! -s "$filepath" ]; then
        continue
    fi

    # 检查文件是否为文本文件。如果不是，则跳过，避免输出二进制乱码。
    # (注意: 'file' 命令可能在某些极简的Docker环境中不存在)
    # pcos_database.db 和 uv.lock (如果是二进制) 会在这里被跳过
    if ! file -b --mime-type "$filepath" | grep -q -e '^text/' -e 'application/json'; then
        echo "Skipping non-text file: $clean_filepath" >> /dev/stderr
        continue
    fi

    # 将文件相对路径写入输出文件
    echo "$clean_filepath" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE" # 添加一个空行

    # 将文件内容包裹在```中并写入输出文件
    # 从文件扩展名推断语言，用于语法高亮
    extension="${clean_filepath##*.}"
    case "$extension" in
        js) lang="javascript" ;;
        py) lang="python" ;;
        md) lang="markdown" ;;
        json) lang="json" ;;
        html) lang="html" ;;
        vue) lang="vue" ;;
        sh) lang="bash" ;;
        yaml) lang="yaml" ;;
        toml) lang="toml" ;;
        *) lang="" ;; # 其他文件类型不指定语言
    esac

    echo "\`\`\`$lang" >> "$OUTPUT_FILE"
    cat "$filepath" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE" # 确保文件末尾有换行符
    echo "\`\`\`" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE" # 在文件块之间添加一个空行

done

echo "✅ All done. The result has been saved to '$OUTPUT_FILE'."
```

index.html

```html
<html lang="en"><head>
        <meta charset="utf-8"> <meta http-equiv="x-ua-compatible" content="ie=edge"> <meta name="keywords" content="CMS，信息技术开发，数据库，java"> <meta name="description" content="信息技术开发，数据库，Java"> <meta name="viewport" content="width=device-width,initial-scale=1"> <!--[if lt IE 9]>
      <script src="http://html5shim.googlecode.com/svn/trunk/html5.js"></script>
<![endif]-->
        <title>首页-多囊卵巢综合征(PCOS)预测计算工具</title>
    <link rel="shortcut icon" href="http://121.43.113.123:8888/favicon.ico"><link href="http://121.43.113.123:8888/css/common.css?633b3271948fc93c498c" rel="stylesheet"><link href="http://121.43.113.123:8888/css/index.css?633b3271948fc93c498c" rel="stylesheet"><style type="text/css">@font-face {
  font-family: "xm-iconfont";
  src: url('//at.alicdn.com/t/font_792691_ptvyboo0bno.eot?t=1574048839056');
  /* IE9 */
  src: url('//at.alicdn.com/t/font_792691_ptvyboo0bno.eot?t=1574048839056#iefix') format('embedded-opentype'), /* IE6-IE8 */ url('data:application/x-font-woff2;charset=utf-8;base64,d09GMgABAAAAAAksAAsAAAAAEYAAAAjeAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHEIGVgCEUgqTXI8lATYCJAM0CxwABCAFhG0HgTwbZQ4jEbaCkVIj+4sD3sS6BFAp9ka91ulVG4leTC/+h+3V+zyRYCTyREKkcZ+D5/u137lPdveLGJBMunoiNPOQPBMq0/FQtEKIkMRDZng69d+hOiQumAr7bJdBOEzMTU77s78mhbI58aCg7ebCs4LBTgCk+cD/4ZqWUHebipp7al3tyKOjwCV/hVyw9PdzaktxI7IMQs26/1N8gV4DI0bVut3UhCaflGGgwM3oTXg1IfRMbCsmrEnriJVeYM2eXHII4KdMMzL4OoACHgZBCTasITcReDUBE8kWPLMTCGoQaDV+eKpUPQI49r8vP6BTPIDCaiBSml3oOQX0voNPebv/u2P0AUfP1w0s5EADzYBZsNdByylo2eVq/NtRdgFpovQR5x2CIwmIZeik6/u0T/m/A7RJP00sCmmyksj/kwc+LC5BFBqDEMDDjwPiANDB9MpJTXwHmsO3YyBwWDA4OFwwJLRcRgAOBUYMDg0mHRwGTAYozsV0AgWYruDwwExDHfzwKWf4OurQ9jzQDtoF+wpistfBfluQ5bQiiJa4ZQoKhShLiMayBbyg05AIkYBoIBJEEApQy/FwYv4HchADIUBXl61dW6mpwIgyp7p8PrHddieSjhY9oqTxyPB/FGNYDklpfYh8VtaoqSgb0bKoGB17CuVUp9Ll2nS2UpNGMSw9hyirA7C6+QLyByIQS0sSSmxvArC5odZmYZMxZSiBR5OkQl0uiufxMH5eL8t3u0d4XKyuq6EMdcpNe2+oXA8p9yPa+4T1PM7+A54tc7tpl2vcAHAftnhZj2chy1CyaCRFsyMqQ5nkNnskEt2yxxZinPsOZjFm4+XWvKqLkfCGS1k4MNP82isxSMf7ZsGYvQVCNAeSSVtzWCxRdXGxyZlA2CvCEevuO7y9M2z2NWH8icydzq/qAJSp1lGvDWFp6Nw3xChJowPD+76nU+upQk6Kw9jI0Rgym9Ct8VlxMI3CSIaDCZja5tDYt0/EYra4tn0Kp3v8Rdezk8svcy1mKhoSvNcZz3LKlUe777Gmval0s7bzAc0k13LGk896V9DuvNn34N0ebKgItkQgOomuJtgQPChNI4cwa7CEWCvfk5QjJFlem6i3SfVShWi5LTFRG+JwdCNpSqbpRFwrtb1TbcRkJi/AbJJQOmfCdnswLNGVM7qqSRO1zO0Q0j5Vr3cYQ07HB0MX6KoIZhx+D9Djs2C5bXtVwvbgJHtSCIL7hjFJme4sZDdS5IlJdKUO1Qt8opn0trBafz3AX933kmCRgyMEWGZjMAkRKhwmIHJGR4ruwFCdWKYzrap2R/mvd2UKajzRAZu88pGAD90Y+02kTFCKrBSXwGGJ3wRcPCdIppTxSmHOfESRwIli0S5J/8AYDCxTGh4XZua4xvfvGx320rDK2qA8g5FlS7pWNLx71+BwgA/KZ5I0aeKmNeCNoNPl8qNHu8uHHzqaKc86fHi4vPuRI4ny+I/vjxw+clh4HXVCFvVnVFx07EHZwVhSRliTTMWSEi0h6YuS6DxCRmiin0B3L4ry6cvR0ijYexFdBL3wGQM0YOrUAZCBkLOBBtQ+xdk7omfgUv+u++admyUeXduyxLM+r/+49rPfhgEZor6GymToNYksNsZyC7ntwAH0928UpgMpxpF0ydNlsMMBw7QsxTCmu0Hf3F+/+vb99Yumhb+e9R0LBNm+4O+hu7lQ5bGjI9j5G88qQ5SLFyuEC7cwd25xoYo2j4eA4bhpM7TZhPtmc+uhVEVSMYXLWh0bfjI8dvUpvDUocPZmU4kwwOfc83wB5wPehrpD3waApbwW+fgRrZXcxw+mB/3woZT+8JFMYwRMIy2k/18qhqcKpjYeYSnIACaUoRDu0e3kQFh98R5fiI8oJqwwGZSJDSbehLzZs7zIeWTQ4UGOIs2c4j2/Q/tn7n7j9juO33On6WhURCT/wO6Y3QdmWFY0Ef6JUeGRggO7ZbtaZlh5RYKWXbLPBLc3l/5h4A0mu3ZXTZ+u6t6VHMAzZhxak50T+24NnRuaOmehRkXlqVR5lIpuwezUUDUdCuJysv8Z/0/8uNE1s7jIJIubFWnI/x7g4nAZx79yYpFoAOU3a9iwT1O/GxUxPY0ljVPv9EukI3qNrl/So2YfzasqHCroNjS0+w0tlPlsYfC6v/01ixquizJH1Kd/VK+OS3iS3rTJWmqsMPdU3B3oFyC9RSumWE/0gG36IjTysfH51IJ/5oOgNYu6p4yb5Fdufhr/Kjtu0oSyYP/WJQrz35aNFnMhtFcwb55NlNnH8Wdu1b+XZA9zqlZrhdPo/V3uBhiUlQ66h0LhbAmFYIncdFOpVMh6Fl7peqy5Z2ZdQBITO2x1Asj1dRFjIBMC3hbuUh8Ooc4W03EjAdo8UL/t0oUfyU8630bmMcw/vqDNAsC9BQD4OqCgH+ljy0UhJB8AAJA+8EmArxk5gnRLik90AElf8rBm+IMvBTWnucb3+0o0ARk+r0ZBv8sU01nnSmP45/H8Dp8C8X+iE9e+ZvXymK/sQJ5/DuqhYKebPnKmPqLYuDcIMWS2/Rjxp2s8Do821LVn6A/xMK1RKvBLK5gyDsZ5uQ6bYusmx2yqLFe4lECHDPcFhojmckuAbnCI6Cn308RI6AAJdtCICQLQyBHKhSgX5YowN6BBPIEB8VxuSfNncpAuutzPnCSiDHDEo+DsKQBPoJi4MpRktepIs2zjO5h84IEMM3ffECKSZU1ZHxfewEI4h494MuuUNNOBjuw18QKHAzEXaAcylS3m3baq9MpnKenYmfEUgCdbXTHEtTVKsvruNGv9/DuYfOAhcuKu9TeEiA9nNJTUDOUbbVkn3sv2eDJrEnVrpvcHOjJeqRsOcpYYLuxoBzKVtCOm3ZaKbtJcurw+e/zN6c7Pd6r4gqUo0WLEiiOueOITvwQkKCEJM9nO3F60y5HkqLhdqUyXZtK3lqwReQ+G40O92UhOt0x/KmKM+u7LTPMzoEBOCYtiUPfSjODiuFXjSDm2idzAoc4Tj9bs2eJYDOU7HQA=') format('woff2'), url('//at.alicdn.com/t/font_792691_ptvyboo0bno.woff?t=1574048839056') format('woff'), url('//at.alicdn.com/t/font_792691_ptvyboo0bno.ttf?t=1574048839056') format('truetype'), /* chrome, firefox, opera, Safari, Android, iOS 4.2+ */ url('//at.alicdn.com/t/font_792691_ptvyboo0bno.svg?t=1574048839056#iconfont') format('svg');
  /* iOS 4.1- */
}
.xm-iconfont {
  font-family: "xm-iconfont" !important;
  font-size: 16px;
  font-style: normal;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.xm-icon-quanxuan:before {
  content: "\e62c";
}
.xm-icon-caidan:before {
  content: "\e610";
}
.xm-icon-fanxuan:before {
  content: "\e837";
}
.xm-icon-pifu:before {
  content: "\e668";
}
.xm-icon-qingkong:before {
  content: "\e63e";
}
.xm-icon-sousuo:before {
  content: "\e600";
}
.xm-icon-danx:before {
  content: "\e62b";
}
.xm-icon-duox:before {
  content: "\e613";
}
.xm-icon-close:before {
  content: "\e601";
}
.xm-icon-expand:before {
  content: "\e641";
}
.xm-icon-banxuan:before {
  content: "\e60d";
}
</style><style type="text/css">@-webkit-keyframes xm-upbit {
  from {
    -webkit-transform: translate3d(0, 30px, 0);
    opacity: 0.3;
  }
  to {
    -webkit-transform: translate3d(0, 0, 0);
    opacity: 1;
  }
}
@keyframes xm-upbit {
  from {
    transform: translate3d(0, 30px, 0);
    opacity: 0.3;
  }
  to {
    transform: translate3d(0, 0, 0);
    opacity: 1;
  }
}
@-webkit-keyframes loader {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(360deg);
    transform: rotate(360deg);
  }
}
@keyframes loader {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(360deg);
    transform: rotate(360deg);
  }
}
xm-select {
  background-color: #FFF;
  position: relative;
  border: 1px solid #E6E6E6;
  border-radius: 2px;
  display: block;
  width: 100%;
  cursor: pointer;
  outline: none;
}
xm-select * {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  font-size: 14px;
  font-weight: 400;
  text-overflow: ellipsis;
  user-select: none;
  -ms-user-select: none;
  -moz-user-select: none;
  -webkit-user-select: none;
}
xm-select:hover,
xm-select:focus {
  border-color: #C0C4CC;
}
xm-select > .xm-tips {
  color: #999999;
  padding: 0 10px;
  position: absolute;
  display: flex;
  height: 100%;
  align-items: center;
}
xm-select > .xm-icon {
  display: inline-block;
  overflow: hidden;
  position: absolute;
  width: 0;
  height: 0;
  right: 10px;
  top: 50%;
  margin-top: -3px;
  cursor: pointer;
  border: 6px dashed transparent;
  border-top-color: #C2C2C2;
  border-top-style: solid;
  transition: all 0.3s;
  -webkit-transition: all 0.3s;
}
xm-select > .xm-icon-expand {
  margin-top: -9px;
  transform: rotate(180deg);
}
xm-select > .xm-label.single-row {
  position: absolute;
  top: 0;
  bottom: 0px;
  left: 0px;
  right: 30px;
  overflow: auto hidden;
}
xm-select > .xm-label.single-row .scroll {
  overflow-y: hidden;
}
xm-select > .xm-label.single-row .label-content {
  flex-wrap: nowrap;
  white-space: nowrap;
}
xm-select > .xm-label.auto-row .label-content {
  flex-wrap: wrap;
  padding-right: 30px !important;
}
xm-select > .xm-label.auto-row .xm-label-block > span {
  white-space: unset;
  height: 100%;
}
xm-select > .xm-label .scroll .label-content {
  display: flex;
  padding: 3px 10px;
}
xm-select > .xm-label .xm-label-block {
  display: flex;
  position: relative;
  padding: 0px 5px;
  margin: 2px 5px 2px 0;
  border-radius: 3px;
  align-items: baseline;
  color: #FFF;
}
xm-select > .xm-label .xm-label-block > span {
  display: flex;
  color: #FFF;
  white-space: nowrap;
}
xm-select > .xm-label .xm-label-block > i {
  color: #FFF;
  margin-left: 8px;
  font-size: 12px;
  cursor: pointer;
  display: flex;
}
xm-select > .xm-label .xm-label-block.disabled {
  background-color: #C2C2C2 !important;
  cursor: no-drop !important;
}
xm-select > .xm-label .xm-label-block.disabled > i {
  cursor: no-drop !important;
}
xm-select > .xm-body {
  position: absolute;
  left: 0;
  top: 42px;
  padding: 5px 0;
  z-index: 999;
  width: 100%;
  min-width: fit-content;
  border: 1px solid #E6E6E6;
  background-color: #fff;
  border-radius: 2px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.12);
  animation-name: xm-upbit;
  animation-duration: 0.3s;
  animation-fill-mode: both;
}
xm-select > .xm-body .scroll-body {
  overflow-x: hidden;
  overflow-y: auto;
}
xm-select > .xm-body .scroll-body::-webkit-scrollbar {
  width: 8px;
}
xm-select > .xm-body .scroll-body::-webkit-scrollbar-track {
  -webkit-border-radius: 2em;
  -moz-border-radius: 2em;
  -ms-border-radius: 2em;
  border-radius: 2em;
  background-color: #FFF;
}
xm-select > .xm-body .scroll-body::-webkit-scrollbar-thumb {
  -webkit-border-radius: 2em;
  -moz-border-radius: 2em;
  -ms-border-radius: 2em;
  border-radius: 2em;
  background-color: #C2C2C2;
}
xm-select > .xm-body.up {
  top: auto;
  bottom: 42px;
}
xm-select > .xm-body.relative {
  position: relative;
  display: block !important;
  top: 0;
  box-shadow: none;
  border: none;
  animation-name: none;
  animation-duration: 0;
  min-width: 100%;
}
xm-select > .xm-body .xm-group {
  cursor: default;
}
xm-select > .xm-body .xm-group-item {
  display: inline-block;
  cursor: pointer;
  padding: 0 10px;
  color: #999;
  font-size: 12px;
}
xm-select > .xm-body .xm-option {
  display: flex;
  align-items: center;
  position: relative;
  padding: 0 10px;
  cursor: pointer;
}
xm-select > .xm-body .xm-option-icon {
  color: transparent;
  display: flex;
  border: 1px solid #E6E6E6;
  border-radius: 3px;
  justify-content: center;
  align-items: center;
}
xm-select > .xm-body .xm-option-icon.xm-custom-icon {
  color: unset;
  border: unset;
}
xm-select > .xm-body .xm-option-icon-hidden {
  margin-right: -10px;
}
xm-select > .xm-body .xm-option-icon.xm-icon-danx {
  border-radius: 100%;
}
xm-select > .xm-body .xm-option-content {
  display: flex;
  position: relative;
  padding-left: 15px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  ;
  width: calc(100% - 20px);
}
xm-select > .xm-body .xm-option.hide-icon .xm-option-content {
  padding-left: 0;
}
xm-select > .xm-body .xm-option.selected.hide-icon .xm-option-content {
  color: #FFF !important;
}
xm-select > .xm-body .xm-option .loader {
  width: 0.8em;
  height: 0.8em;
  margin-right: 6px;
  color: #C2C2C2;
}
xm-select > .xm-body .xm-select-empty {
  text-align: center;
  color: #999;
}
xm-select > .xm-body .disabled {
  cursor: no-drop;
}
xm-select > .xm-body .disabled:hover {
  background-color: #FFF;
}
xm-select > .xm-body .disabled .xm-option-icon {
  border-color: #C2C2C2 !important;
}
xm-select > .xm-body .disabled .xm-option-content {
  color: #C2C2C2 !important;
}
xm-select > .xm-body .disabled.selected > .xm-option-icon {
  color: #C2C2C2 !important;
}
xm-select > .xm-body .xm-search {
  background-color: #FFF !important;
  position: relative;
  padding: 0 10px;
  margin-bottom: 5px;
  cursor: pointer;
}
xm-select > .xm-body .xm-search > i {
  position: absolute;
  color: ;
}
xm-select > .xm-body .xm-search-input {
  border: none;
  border-bottom: 1px solid #E6E6E6;
  padding-left: 27px;
  cursor: text;
}
xm-select > .xm-body .xm-paging {
  padding: 0 10px;
  display: flex;
  margin-top: 5px;
}
xm-select > .xm-body .xm-paging > span:first-child {
  border-radius: 2px 0 0 2px;
}
xm-select > .xm-body .xm-paging > span:last-child {
  border-radius: 0 2px 2px 0;
}
xm-select > .xm-body .xm-paging > span {
  display: flex;
  flex: auto;
  justify-content: center;
  vertical-align: middle;
  margin: 0 -1px 0 0;
  background-color: #fff;
  color: #333;
  font-size: 12px;
  border: 1px solid #e2e2e2;
  flex-wrap: nowrap;
  width: 100%;
  overflow: hidden;
  min-width: 50px;
}
xm-select > .xm-body .xm-toolbar {
  padding: 0 10px;
  display: flex;
  margin: -3px 0;
  cursor: default;
}
xm-select > .xm-body .xm-toolbar .toolbar-tag {
  cursor: pointer;
  display: flex;
  margin-right: 20px;
  color: ;
  align-items: baseline;
}
xm-select > .xm-body .xm-toolbar .toolbar-tag:hover {
  opacity: 0.8;
}
xm-select > .xm-body .xm-toolbar .toolbar-tag:active {
  opacity: 1;
}
xm-select > .xm-body .xm-toolbar .toolbar-tag > i {
  margin-right: 2px;
  font-size: 14px;
}
xm-select > .xm-body .xm-toolbar .toolbar-tag:last-child {
  margin-right: 0;
}
xm-select > .xm-body .xm-body-custom {
  line-height: initial;
  cursor: default;
}
xm-select > .xm-body .xm-body-custom * {
  box-sizing: initial;
}
xm-select > .xm-body .xm-tree {
  position: relative;
}
xm-select > .xm-body .xm-tree-icon {
  display: inline-block;
  margin-right: 3px;
  cursor: pointer;
  border: 6px dashed transparent;
  border-left-color: #C2C2C2;
  border-left-style: solid;
  transition: all 0.3s;
  -webkit-transition: all 0.3s;
  z-index: 2;
  visibility: hidden;
}
xm-select > .xm-body .xm-tree-icon.expand {
  margin-top: 3px;
  margin-right: 5px;
  margin-left: -2px;
  transform: rotate(90deg);
}
xm-select > .xm-body .xm-tree-icon.xm-visible {
  visibility: visible;
}
xm-select > .xm-body .xm-tree .left-line {
  position: absolute;
  left: 13px;
  width: 0;
  z-index: 1;
  border-left: 1px dotted #c0c4cc !important;
}
xm-select > .xm-body .xm-tree .top-line {
  position: absolute;
  left: 13px;
  height: 0;
  z-index: 1;
  border-top: 1px dotted #c0c4cc !important;
}
xm-select > .xm-body .xm-tree .xm-tree-icon + .top-line {
  margin-left: 1px;
}
xm-select > .xm-body .scroll-body > .xm-tree > .xm-option > .top-line,
xm-select > .xm-body .scroll-body > .xm-option > .top-line {
  width: 0 !important;
}
xm-select > .xm-body .xm-cascader-box {
  position: absolute;
  left: 0;
  right: 0;
  top: 0;
  bottom: 0;
  padding: 5px 0;
  border: 1px solid #E6E6E6;
  background-color: #fff;
  border-radius: 2px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.12);
  margin: -1px;
}
xm-select > .xm-body .xm-cascader-box::before {
  content: ' ';
  position: absolute;
  width: 0;
  height: 0;
  border: 6px solid transparent;
  border-right-color: #E6E6E6;
  top: 10px;
  left: -12px;
}
xm-select > .xm-body .xm-cascader-box::after {
  content: ' ';
  position: absolute;
  width: 0;
  height: 0;
  border: 6px solid transparent;
  border-right-color: #fff;
  top: 10px;
  left: -11px;
}
xm-select > .xm-body .xm-cascader-scroll {
  height: 100%;
  overflow-x: hidden;
  overflow-y: auto;
}
xm-select > .xm-body.cascader {
  width: unset;
  min-width: unset;
}
xm-select > .xm-body.cascader .xm-option-content {
  padding-left: 8px;
}
xm-select > .xm-body.cascader .disabled .xm-right-arrow {
  color: #C2C2C2 !important;
}
xm-select > .xm-body.cascader .hide-icon.disabled .xm-right-arrow {
  color: #999 !important;
}
xm-select .xm-input {
  cursor: pointer;
  border-radius: 2px;
  border-width: 1px;
  border-style: solid;
  border-color: #E6E6E6;
  display: block;
  width: 100%;
  box-sizing: border-box;
  background-color: #FFF;
  line-height: 1.3;
  padding-left: 10px;
  outline: 0;
  user-select: text;
  -ms-user-select: text;
  -moz-user-select: text;
  -webkit-user-select: text;
}
xm-select .dis {
  display: none;
}
xm-select .loading {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(255, 255, 255, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
}
xm-select .loader {
  border: 0.2em dotted currentcolor;
  border-radius: 50%;
  -webkit-animation: 1s loader linear infinite;
  animation: 1s loader linear infinite;
  display: inline-block;
  width: 1em;
  height: 1em;
  color: inherit;
  vertical-align: middle;
  pointer-events: none;
}
xm-select .xm-select-default {
  position: absolute;
  width: 100%;
  height: 100%;
  border: none;
  visibility: hidden;
}
xm-select .xm-select-disabled {
  position: absolute;
  left: 0;
  right: 0;
  top: 0;
  bottom: 0;
  cursor: no-drop;
  z-index: 2;
  opacity: 0.3;
  background-color: #FFF;
}
xm-select .item--divided {
  border-top: 1px solid #ebeef5;
  width: calc(100% - 20px);
  cursor: initial;
}
xm-select .xm-right-arrow {
  position: absolute;
  color: ;
  right: 5px;
  top: -1px;
  font-weight: 700;
  transform: scale(0.6, 1);
}
xm-select .xm-right-arrow::after {
  content: '>';
}
xm-select[size='large'] {
  min-height: 40px;
  line-height: 40px;
}
xm-select[size='large'] .xm-input {
  height: 40px;
}
xm-select[size='large'] .xm-label .scroll .label-content {
  line-height: 34px;
}
xm-select[size='large'] .xm-label .xm-label-block {
  height: 30px;
  line-height: 30px;
}
xm-select[size='large'] .xm-body .xm-option .xm-option-icon {
  height: 20px;
  width: 20px;
  font-size: 20px;
}
xm-select[size='large'] .xm-paging > span {
  height: 34px;
  line-height: 34px;
}
xm-select[size='large'] .xm-tree .left-line {
  height: 100%;
  bottom: 20px;
}
xm-select[size='large'] .xm-tree .left-line-group {
  height: calc(100% - 40px);
}
xm-select[size='large'] .xm-tree .xm-tree-icon.xm-hidden + .top-line {
  top: 19px;
}
xm-select[size='large'] .item--divided {
  margin: 10px;
}
xm-select {
  min-height: 36px;
  line-height: 36px;
}
xm-select .xm-input {
  height: 36px;
}
xm-select .xm-label .scroll .label-content {
  line-height: 30px;
}
xm-select .xm-label .xm-label-block {
  height: 26px;
  line-height: 26px;
}
xm-select .xm-body .xm-option .xm-option-icon {
  height: 18px;
  width: 18px;
  font-size: 18px;
}
xm-select .xm-paging > span {
  height: 30px;
  line-height: 30px;
}
xm-select .xm-tree .left-line {
  height: 100%;
  bottom: 18px;
}
xm-select .xm-tree .left-line-group {
  height: calc(100% - 36px);
}
xm-select .xm-tree .xm-tree-icon.xm-hidden + .top-line {
  top: 17px;
}
xm-select .item--divided {
  margin: 9px;
}
xm-select[size='small'] {
  min-height: 32px;
  line-height: 32px;
}
xm-select[size='small'] .xm-input {
  height: 32px;
}
xm-select[size='small'] .xm-label .scroll .label-content {
  line-height: 26px;
}
xm-select[size='small'] .xm-label .xm-label-block {
  height: 22px;
  line-height: 22px;
}
xm-select[size='small'] .xm-body .xm-option .xm-option-icon {
  height: 16px;
  width: 16px;
  font-size: 16px;
}
xm-select[size='small'] .xm-paging > span {
  height: 26px;
  line-height: 26px;
}
xm-select[size='small'] .xm-tree .left-line {
  height: 100%;
  bottom: 16px;
}
xm-select[size='small'] .xm-tree .left-line-group {
  height: calc(100% - 32px);
}
xm-select[size='small'] .xm-tree .xm-tree-icon.xm-hidden + .top-line {
  top: 15px;
}
xm-select[size='small'] .item--divided {
  margin: 8px;
}
xm-select[size='mini'] {
  min-height: 28px;
  line-height: 28px;
}
xm-select[size='mini'] .xm-input {
  height: 28px;
}
xm-select[size='mini'] .xm-label .scroll .label-content {
  line-height: 22px;
}
xm-select[size='mini'] .xm-label .xm-label-block {
  height: 18px;
  line-height: 18px;
}
xm-select[size='mini'] .xm-body .xm-option .xm-option-icon {
  height: 14px;
  width: 14px;
  font-size: 14px;
}
xm-select[size='mini'] .xm-paging > span {
  height: 22px;
  line-height: 22px;
}
xm-select[size='mini'] .xm-tree .left-line {
  height: 100%;
  bottom: 14px;
}
xm-select[size='mini'] .xm-tree .left-line-group {
  height: calc(100% - 28px);
}
xm-select[size='mini'] .xm-tree .xm-tree-icon.xm-hidden + .top-line {
  top: 13px;
}
xm-select[size='mini'] .item--divided {
  margin: 7px;
}
.layui-form-pane xm-select {
  margin: -1px -1px -1px 0;
}
</style></head>
    <body>
        <div id="header" class="container-fluid"> <div class="row head"> <div class="col-lg-6 col-md-6 col-xs-12 leftlogo"> <img src="http://121.43.113.123:8888/resource/logo.png" class="img-responsiv leftimg"> <div class="logotext"> <div class="logo_en">多囊卵巢综合征筛查工具</div> <div class="logo_en_lo"> PCOSt: Polycystic Ovary Syndrome Screening Tool </div> </div> </div> <div class="visible-lg rightlogo"> <img src="http://121.43.113.123:8888/resource/right.svg" class="rightimg"> </div> </div> </div>
        <div class="container-fluid">
            <div class="row">
                <div class="col-lg-12 col-md-12 col-xs-12 tipsarea" style="z-index: 1;"><span class="tipsch">世界浏览记录实时监测</span>&nbsp;&nbsp;<span class="tipsen">browsing records</span></div>
            </div>F

            <div class="row">
                <!-- <div id="worldmap1" class="col-lg-12 col-md-12 col-xs-12 " style="height: 600px; width: 100%;position: absolute;"></div> -->
                <div class="col-lg-12 col-md-12 col-xs-12 main">
                    <div id="worldmap" class="worldmap" style="-webkit-tap-highlight-color: transparent; user-select: none;" _echarts_instance_="ec_1748373295974"><div style="position: relative; width: 1090px; height: 500px; padding: 0px; margin: 0px; border-width: 0px; cursor: default;"><canvas data-zr-dom-id="zr_0" width="1635" height="750" style="position: absolute; left: 0px; top: 0px; width: 1090px; height: 500px; user-select: none; -webkit-tap-highlight-color: rgba(0, 0, 0, 0); padding: 0px; margin: 0px; border-width: 0px;"></canvas><canvas data-zr-dom-id="zr_2" width="1635" height="750" style="position: absolute; left: 0px; top: 0px; width: 1090px; height: 500px; user-select: none; -webkit-tap-highlight-color: rgba(0, 0, 0, 0); padding: 0px; margin: 0px; border-width: 0px;"></canvas></div></div>
                    <div class="float_div" style="z-index: 1;">
                            <div style="margin: 5px 0;"><button type="button" class="btn btn-primary btn-lg" id="bigbtu" style="opacity: 0.3; width: 58px;">+</button></div>
                            <div style="margin: 5px 0;"><button type="button" class="btn btn-primary btn-lg" id="smallbtu" style="opacity: 0.3; width: 58px;">-</button></div>
                    </div>
                </div>
            </div>
            <div class="row">
              <div id="calculate" class="col-lg-6 col-md-12 col-xs-12">
                <div class="row">
                    <div class="col-lg-12 col-xs-12 areabottom">
                        <div class="card">
                            <div class="card-body" style="display: flex;justify-content: space-between;align-items: center">
                                <div>
                                    <div><img class="logoimg" src="http://121.43.113.123:8888/resource/logo.png" alt="logo"><span class="calculatetch">多囊卵巢综合征(PCOS)预测计算工具</span></div>
                                    <div class="calculateten">Calculation Tool For Polycystic Ovarian Syndrome(PCOS)</div>
                                </div>
<!--                                <div id="batchCalcBtn">批量计算</div>-->
                            </div>
                            <div class="card-body">
                                <div class="form-group row">
                                    <div class="col-6 col-lg-6 col-md-6 col-xs-12  col-form-label text-left">
                                        <label class="labelen">AMH</label>&nbsp;&nbsp;<label class="labelch">抗缪勒管激素</label>
                                    </div>
                                    <div class="col-2 col-lg-6 col-md-6 col-xs-12 input-group">
                                        <input id="amh" placeholder="" class="form-control inputstyle" onkeyup="value=value.replace(/[^\d.]/g,'')"><span class="input-group-addon labelch">ng/ml</span>
                                    </div>
                                </div>
                                <div class="form-group row">
                                    <div class="col-lg-6 col-md-6 col-xs-12 col-form-label text-left">
                                        <span class="labelen">Menstrual cycle days</span>&nbsp;&nbsp;<span class="labelch">月经周期天数</span>
                                    </div>
                                    <div class="col-lg-6 col-md-6 col-xs-12 input-group">
                                        <input id="startday" placeholder="" class="form-control inputstyle" onkeyup="this.value=this.value.replace(/\D/g,'')">
                                        <span class="input-group-addon labelch">天(days)-</span>
                                        <input id="endday" placeholder="" class="form-control inputstyle" onkeyup="this.value=this.value.replace(/\D/g,'')">
                                        <span class="input-group-addon labelch">天(days)</span>
                                    </div>
                                </div>
                                <div class="form-group row">
                                    <div class="col-lg-6 col-md-6 col-xs-12 col-form-label text-left">
                                        <span class="labelen">BMI</span>&nbsp;&nbsp;<span class="labelch">体重指数</span>
                                    </div>
                                    <div class="col-lg-6 col-md-6 col-xs-12 input-group">
                                        <input id="bmi" placeholder="" class="form-control inputstyle" onkeyup="value=value.replace(/[^\d.]/g,'')"><span class="input-group-addon labelch">kg/㎡</span>
                                    </div>
                                </div>
                                <div class="form-group row">
                                    <div class="col-lg-6 col-md-6 col-xs-12 col-form-label text-left">
                                        <span class="labelen">Androstenedione</span>&nbsp;&nbsp;<span class="labelch">雄烯二酮</span>
                                    </div>
                                    <div class="col-lg-6 col-md-6 col-xs-12 input-group">
                                        <input id="androstenedione" placeholder="" class="form-control inputstyle" onkeyup="value=value.replace(/[^\d.]/g,'')"><span class="input-group-addon labelch">nmol/L</span>
                                    </div>
                                </div>
                                <div class="form-group row">
                                    <div class="col-lg-6 col-md-6 col-xs-6">
                                        <button type="submit" id="batchCalcBtn">
                                            <span class="butch">批量计算</span>&nbsp;&nbsp;<span class="buten">Batch calculations</span>
                                        </button>
                                    </div>
                                    <div class="col-lg-6 col-md-6 col-xs-6">
                                        <button type="submit" id="calcBtn">
                                            <span class="butch">点击计算</span>  <span class="buten">Calculate</span>
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-lg-6 col-md-6 col-xs-12 areabottom">
                        <div class="card">
                            <div class="new-card-body">
                                <div id="title" class="pcostitletops">
                                    <img class="logoimg" src="http://121.43.113.123:8888/resource/logo.png" alt="logo">
                                    <span class="calculatetch">患病概率</span><span class="calculateten">PCOS probability</span>
                                </div>
                               <div id="pcoscharts" class="pocsarea" style="-webkit-tap-highlight-color: transparent; user-select: none;" _echarts_instance_="ec_1748373295975"><div style="position: relative; width: 501px; height: 295px; padding: 0px; margin: 0px; border-width: 0px; cursor: default;"><canvas data-zr-dom-id="zr_0" width="751" height="442" style="position: absolute; left: 0px; top: 0px; width: 501px; height: 295px; user-select: none; -webkit-tap-highlight-color: rgba(0, 0, 0, 0); padding: 0px; margin: 0px; border-width: 0px;"></canvas></div></div>
                               <div class="pocsdes" id="pocsdes"> <p>您的患病概率为11.947%，</p><p>属于中危人群。</p></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-6 col-md-6 col-xs-12 areabottom">
                        <div class="card">
                            <div class="new-card-body">
                                <div id="risktitile" class="risktitle">
                                    <img class="logoimg" src="http://121.43.113.123:8888/resource/logo.png" alt="logo">
                                    <span class="calculatetch">危险分层模型</span><span class="calculateten">Risk group</span>
                                </div>
                                <div class="riskarea">
                                    <div id="riskcharts" class="riskleft">
                                        <div class="part1">
                                            <div class="part11"><span class="riskpre">100%</span></div>
                                            <div class="part12"></div>
                                            <div class="part13"></div>
                                        </div>
                                        <div class="part1">
                                            <div class="part11"><span class="riskpre">90%</span></div>
                                            <div class="part12"></div>
                                            <div class="part13"></div>
                                        </div>
                                        <div class="part1">
                                            <div class="part11"><span class="riskpre">80%</span></div>
                                            <div class="part12"></div>
                                            <div class="part13"></div>
                                        </div>
                                        <div class="part1">
                                            <div class="part11"><span class="riskpre">70%</span></div>
                                            <div class="part12"></div>
                                            <div class="part13"></div>
                                        </div>
                                        <div class="part1">
                                            <div class="part11"><span class="riskpre">60%</span></div>
                                            <div class="part12"></div>
                                            <div class="part13"></div>
                                        </div>
                                        <div class="part2">
                                            <div class="part11"><span class="riskpre">50%</span></div>
                                            <div class="part12"></div>
                                            <div class="part23"></div>
                                        </div>
                                        <div class="part2">
                                            <div class="part11"><span class="riskpre">40%</span></div>
                                            <div class="part12"></div>
                                            <div class="part23"></div>
                                        </div>
                                        <div class="part2">
                                            <div class="part11"><span class="riskpre">30%</span></div>
                                            <div class="part12"></div>
                                            <div class="part23"></div>
                                        </div>
                                        <div class="part2">
                                            <div class="part11"><span class="riskpre">20%</span></div>
                                            <div class="part12"></div>
                                            <div class="part23"></div>
                                        </div>
                                        <div class="part3">
                                            <div class="part11">
                                                <div class="riskpre">
                                                    <div>10%</div>
                                                    <div style="margin-top:1.5em">0%</div>
                                                </div>
                                            </div>
                                            <div class="part32"></div>
                                            <div class="part33"></div>
                                        </div>
                                    </div>
                                    <div class="riskright">
                                        <div class="testResult" id="testResult" style="background-color: rgba(245, 240, 117, 0.3); border: 2px solid rgb(245, 240, 117);">
                                            <div class="resultpart" id="resultpartdiv"><div class="resulttitlediv"><p><span class="resulttitlech">检查结果</span></p></div><div class="resultdiv"><p><span class="resultch">中危</span></p></div><div class="resultendiv"><p><span class="resulten">Test Result: </span></p></div><div class="resultendiv"><p><span class="resulten">Medium-Risk Group</span></p></div></div>
                                        </div>
                                        <div class="resultmeanarea" style="height:20%">
                                            <div class="resultmean" style="height: 100%;">
                                                <div class="resultgroup">
                                                    <div class="highstyle"></div><span class="resultdes">高危组&nbsp;&nbsp;&nbsp;&nbsp;High-risk group</span>
                                                </div>
                                                <div class="resultgroup">
                                                    <div class="medistyle"></div> <span class="resultdes">中危组&nbsp;&nbsp;&nbsp;&nbsp;Medium-risk group</span>
                                                </div>
                                                <div class="resultgroup">
                                                    <div class="lowstyle"></div> <span class="resultdes">低危组&nbsp;&nbsp;&nbsp;&nbsp;Low-risk group</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
              </div>
              <div id="usage" class="col-lg-6 col-md-12 col-xs-12 rowarea">
                <div class="row">
                    <div class="col-lg-12 col-md-6  col-xs-12 areabottom">
                        <div class="card">
                            <div class="new-card-body">
                                <div id="usagetitle" class="pcostitletops">
                                    <img class="logoimg" src="http://121.43.113.123:8888/resource/logo.png" alt="logo">
                                    <span class="calculatetch">使用量排名</span><span class="calculateten">Usage ranking</span>
                                </div>
                                <div id="usagecharts" class="usagearea"> <div> <span class="usagecountry">NO.1&nbsp;中国浙江&nbsp;&nbsp;Zhejiang,China</span> <span style="color:rgba(255,255,255,.7);font-size:12px;font-weight:lighter;float:right">406</span> </div> <div class="progress l_left" id="usagevader1" style="width:100%;padding:0;margin-top:4px;border-radius:10px"><span class="barControl" style="width:30%;"><div class="barContro_space5"><span class="vader5" style="height: 20px; width: 22.3077%;" id="barusagevader1"></span></div></span></div>  <div> <span class="usagecountry">NO.2&nbsp;中国广东&nbsp;&nbsp;Guangdong,China</span> <span style="color:rgba(255,255,255,.7);font-size:12px;font-weight:lighter;float:right">307</span> </div> <div class="progress l_left" id="usagevader2" style="width:100%;padding:0;margin-top:4px;border-radius:10px"><span class="barControl" style="width:30%;"><div class="barContro_space5"><span class="vader5" style="height: 20px; width: 16.8681%;" id="barusagevader2"></span></div></span></div>  <div> <span class="usagecountry">NO.3&nbsp;中国北京&nbsp;&nbsp;Beijing,China</span> <span style="color:rgba(255,255,255,.7);font-size:12px;font-weight:lighter;float:right">246</span> </div> <div class="progress l_left" id="usagevader3" style="width:100%;padding:0;margin-top:4px;border-radius:10px"><span class="barControl" style="width:30%;"><div class="barContro_space5"><span class="vader5" style="height: 20px; width: 13.5165%;" id="barusagevader3"></span></div></span></div>  <div> <span class="usagecountry">NO.4&nbsp;中国天津&nbsp;&nbsp;Tianjin,China</span> <span style="color:rgba(255,255,255,.7);font-size:12px;font-weight:lighter;float:right">206</span> </div> <div class="progress l_left" id="usagevader4" style="width:100%;padding:0;margin-top:4px;border-radius:10px"><span class="barControl" style="width:30%;"><div class="barContro_space5"><span class="vader5" style="height: 20px; width: 11.3187%;" id="barusagevader4"></span></div></span></div>  <div> <span class="usagecountry">NO.5&nbsp;中国上海&nbsp;&nbsp;Shanghai,China</span> <span style="color:rgba(255,255,255,.7);font-size:12px;font-weight:lighter;float:right">140</span> </div> <div class="progress l_left" id="usagevader5" style="width:100%;padding:0;margin-top:4px;border-radius:10px"><span class="barControl" style="width:30%;"><div class="barContro_space5"><span class="vader5" style="height: 20px; width: 7.69231%;" id="barusagevader5"></span></div></span></div>  <div> <span class="usagecountry">NO.6&nbsp;中国山东&nbsp;&nbsp;Shandong,China</span> <span style="color:rgba(255,255,255,.7);font-size:12px;font-weight:lighter;float:right">68</span> </div> <div class="progress l_left" id="usagevader6" style="width:100%;padding:0;margin-top:4px;border-radius:10px"><span class="barControl" style="width:30%;"><div class="barContro_space5"><span class="vader5" style="height: 20px; width: 3.73626%;" id="barusagevader6"></span></div></span></div> </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-12 col-md-6 col-xs-12 areabottom">
                        <div class="card">
                            <div class="new-card-body">
                                <div id="visittitle" class="pcostitletops">
                                    <img class="logoimg" src="http://121.43.113.123:8888/resource/logo.png" alt="logo">
                                    <span class="calculatetch">访问量排名</span><span class="calculateten">Visit ranking</span>
                                </div>
                                <div id="visitcharts" class="usagearea"> <div> <span class="usagecountry">NO.1&nbsp;中国浙江&nbsp;&nbsp;Zhejiang,China</span> <span style="color:rgba(255,255,255,.7);font-size:12px;font-weight:lighter;float:right">1203</span> </div> <div class="progress l_left" id="visitVader1" style="width:100%;padding:0;margin-top:4px;border-radius:10px"><span class="barControl" style="width:30%;"><div class="barContro_space5"><span class="vader5" style="height: 20px; width: 29.4998%;" id="barvisitVader1"></span></div></span></div>  <div> <span class="usagecountry">NO.2&nbsp;美国&nbsp;&nbsp;America</span> <span style="color:rgba(255,255,255,.7);font-size:12px;font-weight:lighter;float:right">521</span> </div> <div class="progress l_left" id="visitVader2" style="width:100%;padding:0;margin-top:4px;border-radius:10px"><span class="barControl" style="width:30%;"><div class="barContro_space5"><span class="vader5" style="height: 20px; width: 12.7759%;" id="barvisitVader2"></span></div></span></div>  <div> <span class="usagecountry">NO.3&nbsp;中国北京&nbsp;&nbsp;Beijing,China</span> <span style="color:rgba(255,255,255,.7);font-size:12px;font-weight:lighter;float:right">433</span> </div> <div class="progress l_left" id="visitVader3" style="width:100%;padding:0;margin-top:4px;border-radius:10px"><span class="barControl" style="width:30%;"><div class="barContro_space5"><span class="vader5" style="height: 20px; width: 10.6179%;" id="barvisitVader3"></span></div></span></div>  <div> <span class="usagecountry">NO.4&nbsp;中国广东&nbsp;&nbsp;Guangdong,China</span> <span style="color:rgba(255,255,255,.7);font-size:12px;font-weight:lighter;float:right">378</span> </div> <div class="progress l_left" id="visitVader4" style="width:100%;padding:0;margin-top:4px;border-radius:10px"><span class="barControl" style="width:30%;"><div class="barContro_space5"><span class="vader5" style="height: 20px; width: 9.26925%;" id="barvisitVader4"></span></div></span></div>  <div> <span class="usagecountry">NO.5&nbsp;中国陕西&nbsp;&nbsp;Shanxi,China</span> <span style="color:rgba(255,255,255,.7);font-size:12px;font-weight:lighter;float:right">243</span> </div> <div class="progress l_left" id="visitVader5" style="width:100%;padding:0;margin-top:4px;border-radius:10px"><span class="barControl" style="width:30%;"><div class="barContro_space5"><span class="vader5" style="height: 20px; width: 5.9588%;" id="barvisitVader5"></span></div></span></div>  <div> <span class="usagecountry">NO.6&nbsp;中国上海&nbsp;&nbsp;Shanghai,China</span> <span style="color:rgba(255,255,255,.7);font-size:12px;font-weight:lighter;float:right">219</span> </div> <div class="progress l_left" id="visitVader6" style="width:100%;padding:0;margin-top:4px;border-radius:10px"><span class="barControl" style="width:30%;"><div class="barContro_space5"><span class="vader5" style="height: 20px; width: 5.37028%;" id="barvisitVader6"></span></div></span></div> </div>
                            </div>
                        </div>
                    </div>
                </div>
              </div>
            </div>
        </div>
        <div id="batchCalcModal" style="background: rgb(47, 49, 49); display: none;" class="">
            <div class="batchCalcModal_con">
                <div class="batchCalcModal_con_download_temp" id="downloadTemp">点击此处下载模版</div>
                <div class="batchCalcModal_con_upload" id="batchCalcUpload">
                    <img class="upload_img" id="uploadImgEle" src="http://121.43.113.123:8888/resource/upload.png" alt="">
                    <div class="batchCalcModal_con_upload_title">点击此处上传文件或拖拽</div>
                    <div class="batchCalcModal_con_upload_tip">最多上传1个文件</div>

                </div><input class="layui-upload-file" type="file" accept=".xlsx,.xls" name="file">
            </div>
        </div>
        <footer> <div class="container"> <div class="row"> <div class="hidden-lg col-md-12 col-lg-12 col-xs-12" style="margin-bottom:1em"> <div><img src="http://121.43.113.123:8888/resource/logo7.svg" style="width:100%;height:100%"></div> </div> <div class="visible-lg col-md-12 col-lg-12 col-xs-12 desc"> <div><span class="departfont">国家妇产疾病临床医学研究中心 北京大学第三医院妇产科生殖医学中心</span></div> <div><span class="engfont">center for reproductive medicine,department of obstetrics and gynecology,national clinical research center for obstetrics and gynecology,peking university third hospital,beijing,china</span></div> </div> </div> </div> </footer> <script src="//static.vecverse.com/pcos/js/jquery.min.js"></script> <script src="//static.vecverse.com/pcos/js/echarts.min.js"></script> <script src="//static.vecverse.com/pcos/js/echarts-gl.min.js"></script> <script src="//static.vecverse.com/pcos/js/worldchina.js"></script> <script src="//static.vecverse.com/pcos/js/ProgressBarWars.js"></script>
    <script type="text/javascript" src="http://121.43.113.123:8888/js/base.js?633b3271948fc93c498c"></script><script type="text/javascript" src="http://121.43.113.123:8888/js/index.js?633b3271948fc93c498c"></script>

<script src="https://unpkg.com/axios/dist/axios.min.js">

</script><div class="layui-layer-move" id="layui-layer-move" style="cursor: move; display: none;"></div></body></html>
```

frontend/src/main.js

```javascript
import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

import App from './App.vue'

const app = createApp(App)

app.use(ElementPlus)
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

app.mount('#app')
```

frontend/src/App.vue

```vue
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
```

frontend/.vscode/extensions.json

```json
{
  "recommendations": ["Vue.volar"]
}

```

frontend/.vscode/settings.json

```json
{
  "explorer.fileNesting.enabled": true,
  "explorer.fileNesting.patterns": {
    "tsconfig.json": "tsconfig.*.json, env.d.ts",
    "vite.config.*": "jsconfig*, vitest.config.*, cypress.config.*, playwright.config.*",
    "package.json": "package-lock.json, pnpm*, .yarnrc*, yarn*, .eslint*, eslint*, .oxlint*, oxlint*, .prettier*, prettier*, .editorconfig"
  }
}

```

frontend/.gitignore

```
# Logs
logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*
lerna-debug.log*

node_modules
.DS_Store
dist
dist-ssr
coverage
*.local

/cypress/videos/
/cypress/screenshots/

# Editor directories and files
.vscode/*
!.vscode/extensions.json
.idea
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?

*.tsbuildinfo

```

frontend/jsconfig.json

```json
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "exclude": ["node_modules", "dist"]
}

```

frontend/index.html

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8">
    <link rel="icon" href="/favicon.ico">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>再次妊娠孕期疾病发生风险评估</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.js"></script>
  </body>
</html>
```

frontend/vite.config.js

```javascript
import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    port: 5173,
    // proxy: {
    //   '/api': {
    //     target: 'http://127.0.0.1:8000',
    //     changeOrigin: true,
    //   }
    // }
  }
})
```

frontend/package.json

```json
{
  "name": "frontend",
  "version": "0.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@element-plus/icons-vue": "^2.3.1",
    "axios": "^1.9.0",
    "echarts": "^5.6.0",
    "element-plus": "^2.9.11",
    "vue": "^3.5.13",
    "vue-echarts": "^7.0.3"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.2.3",
    "vite": "^6.2.4",
    "vite-plugin-vue-devtools": "^7.7.2"
  }
}

```
