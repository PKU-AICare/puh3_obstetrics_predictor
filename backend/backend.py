# backend/main.py
import os
import io
import math
import zipfile
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func, Text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from starlette.background import BackgroundTask

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
    # Store all prediction results as a JSON string for flexibility
    results_json = Column(Text)

class VisitRecord(Base):
    __tablename__ = "visit_records"
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String)
    location = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# --- Pydantic Models for API ---
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

# --- Disease Formulas and Mappings ---
# This dictionary holds the coefficients for each variable in the logit formula.
# logit = b0 + b1*x1 + b2*x2 + ...  (We assume there is an intercept of 0 if not provided)
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
    allow_origins=["*"],  # Allow all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Helper Functions ---
def get_location_from_ip(ip: str) -> str:
    if not ip or ip in ("127.0.0.1", "::1") or ip.startswith(("192.168.", "10.", "172.16.")):
        return "Local Network"
    try:
        # Using a free, no-key-required IP geolocation service
        response = requests.get(f"https://ipapi.co/{ip}/json/", timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            return "Unknown"
        city = data.get("city", "")
        region = data.get("region", "")
        country_name = data.get("country_name", "Unknown")

        if country_name == "China":
            return f"中国 {region}".strip()
        return ", ".join(filter(None, [city, region, country_name]))

    except requests.exceptions.RequestException:
        return "Unknown"

def parse_excel_to_features(contents: bytes) -> Dict[str, float]:
    try:
        xls = pd.ExcelFile(io.BytesIO(contents))
        if '数据上传表_基线特征' not in xls.sheet_names or '数据上传表_实验室检查' not in xls.sheet_names:
            raise HTTPException(status_code=400, detail="Excel file must contain sheets: '数据上传表_基线特征' and '数据上传表_实验室检查'")

        df_baseline = pd.read_excel(xls, sheet_name='数据上传表_基线特征')
        df_lab = pd.read_excel(xls, sheet_name='数据上传表_实验室检查')

        features = {}

        # Process baseline features
        # Assuming the table has '变量名' and '公式里面的变量名' columns as per spec
        for _, row in df_baseline.iterrows():
            var_name_cn = str(row.iloc[0]) # The value/feature name
            formula_var = str(row.iloc[3]) # The variable name used in formulas
            value = row.iloc[1] if len(row) > 1 else None  # Assuming value is in the second column
            if pd.notna(value):
                features[formula_var] = float(value)

        # Process lab features
        # This is complex because the layout is stacked.
        # '对应公式里面的变量名' are in columns 6, 7, 8
        for _, row in df_lab.iterrows():
            # early, mid, late pregnancy values and their corresponding variable names
            periods = [(3, 6), (4, 7), (5, 8)] # (value_col_idx, var_name_col_idx)
            for val_idx, var_idx in periods:
                if len(row) > var_idx:
                    formula_var = str(row.iloc[var_idx])
                    value = row.iloc[val_idx]
                    if pd.notna(value) and formula_var != 'nan':
                       try:
                           features[formula_var] = float(value)
                       except (ValueError, TypeError):
                           # Ignore non-numeric values
                           pass

        return features
    except Exception as e:
        # Catch any pandas or processing error
        raise HTTPException(status_code=400, detail=f"Error parsing Excel file: {e}")

def calculate_probabilities(features: Dict[str, float]) -> List[Dict[str, Any]]:
    results = []
    for abbr, data in DISEASE_FORMULAS.items():
        logit = 0.0
        for var, coeff in data["coeffs"].items():
            feature_value = features.get(var, 0.0) # Default to 0 if not present
            logit += coeff * feature_value

        try:
            probability = 1 / (1 + math.exp(-logit))
        except OverflowError:
            # If logit is very large or small, exp will overflow.
            probability = 0.0 if logit < 0 else 1.0

        results.append({
            "disease_abbr": abbr,
            "disease_name_cn": data["name_cn"],
            "probability": probability
        })
    return results

async def cleanup_temp_file(filepath: str):
    if os.path.exists(filepath):
        os.remove(filepath)

# --- FastAPI Event Handlers & Middleware ---
@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)

@app.middleware("http")
async def log_visits_middleware(request: Request, call_next):
    # Exclude backend-specific paths and static file requests
    if request.method == "OPTIONS" or any(p in request.url.path for p in ["/docs", "/openapi.json", ".js", ".css"]):
        return await call_next(request)

    # Identify client IP
    client_ip = request.headers.get("X-Forwarded-For") or request.client.host

    # We log the visit regardless of the endpoint path for general traffic analysis
    db = SessionLocal()
    try:
        # Check if a recent visit from this IP exists to avoid flooding the DB
        # This is a simple de-duplication strategy
        # A more robust solution might use Redis or a different approach
        is_api_call = "/api/" in request.url.path
        if not is_api_call:
            location = get_location_from_ip(client_ip)
            visit = VisitRecord(ip_address=client_ip, location=location)
            db.add(visit)
            db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error logging visit for IP {client_ip}: {e}")
    finally:
        db.close()

    response = await call_next(request)
    return response


# --- API Endpoints ---
@app.post("/api/predict-single", response_model=SinglePredictionResponse)
async def predict_single(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .xlsx file.")

    patient_id = os.path.splitext(file.filename)[0]
    contents = await file.read()

    features = parse_excel_to_features(contents)
    predictions = calculate_probabilities(features)

    # Log the successful calculation
    client_ip = request.headers.get("X-Forwarded-For") or request.client.host
    location = get_location_from_ip(client_ip)

    # Convert predictions to a JSON string for storage
    import json
    results_for_db = {p["disease_abbr"]: p["probability"] for p in predictions}

    db_record = PredictionRecord(
        patient_id=patient_id,
        ip_address=client_ip,
        location=location,
        results_json=json.dumps(results_for_db)
    )
    db.add(db_record)
    db.commit()

    return SinglePredictionResponse(patient_id=patient_id, predictions=predictions)


@app.post("/api/predict-batch")
async def predict_batch(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .zip file.")

    contents = await file.read()
    client_ip = request.headers.get("X-Forwarded-For") or request.client.host
    location = get_location_from_ip(client_ip)

    all_results = []

    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as z:
            for filename in z.namelist():
                if filename.endswith('.xlsx') and not filename.startswith('__MACOSX'):
                    patient_id = os.path.splitext(os.path.basename(filename))[0]
                    with z.open(filename) as xlsx_file:
                        xlsx_contents = xlsx_file.read()
                        features = parse_excel_to_features(xlsx_contents)
                        predictions = calculate_probabilities(features)

                        # Prepare data for DataFrame
                        row_data = {"patient_id": patient_id}
                        for p in predictions:
                           row_data[f'{p["disease_abbr"]}_prob'] = p["probability"]
                        all_results.append(row_data)

                        # Log each prediction to DB
                        import json
                        results_for_db = {p["disease_abbr"]: p["probability"] for p in predictions}
                        db_record = PredictionRecord(
                            patient_id=patient_id,
                            ip_address=client_ip,
                            location=location,
                            results_json=json.dumps(results_for_db)
                        )
                        db.add(db_record)
        db.commit()
    except zipfile.BadZipFile:
        db.rollback()
        raise HTTPException(status_code=400, detail="Invalid or corrupted ZIP file.")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred during batch processing: {e}")

    if not all_results:
        raise HTTPException(status_code=400, detail="No valid .xlsx files found in the ZIP archive.")

    # Create Excel response
    df_results = pd.DataFrame(all_results)
    output_buffer = io.BytesIO()
    df_results.to_excel(output_buffer, index=False, sheet_name="Batch Predictions")
    output_buffer.seek(0)

    return StreamingResponse(
        output_buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=batch_prediction_results_{datetime.now().strftime('%Y%m%d%H%M')}.xlsx"}
    )

@app.get("/api/stats/usage", response_model=List[LocationStat])
async def get_usage_stats(db: Session = Depends(get_db)):
    """Returns the top locations based on prediction counts."""
    query = (
        db.query(PredictionRecord.location, func.count(PredictionRecord.id).label("count"))
        .group_by(PredictionRecord.location)
        .order_by(func.count(PredictionRecord.id).desc())
        .limit(10)
    ).all()
    return [LocationStat(location=loc, count=count) for loc, count in query]

@app.get("/api/stats/visits", response_model=List[LocationStat])
async def get_visit_stats(db: Session = Depends(get_db)):
    """Returns the top locations based on visit counts."""
    query = (
        db.query(VisitRecord.location, func.count(VisitRecord.id).label("count"))
        .group_by(VisitRecord.location)
        .order_by(func.count(VisitRecord.id).desc())
        .limit(10)
    ).all()
    return [LocationStat(location=loc, count=count) for loc, count in query]


@app.get("/api/download-template")
async def download_template():
    """Generates a two-sheet Excel template for data upload."""
    baseline_cols = {
        '变量名': ['age', 'weight_gain', 'pre_bmi', 'nation', 'Hypothyroidism', 'gestational_age', 'birth_weight', 'birth_length', 'gender', 'hysteromyoma', 'Delivery_method', 'NICU', 'DM', 'PROM', 'GDM', 'PE', 'Placenta Previa', 'PPH'],
        '输入数值': ['' for _ in range(18)],
        '中文名称': ['年龄', '孕期体重增长', '孕前BMI', '民族（汉族 vs 少数民族）', '甲状腺功能减退症', '分娩孕周', '出生体重', '出生身长', '新生儿性别', '子宫肌瘤', '分娩方式', '新生儿转NICU', '孕前糖尿病', '胎膜早破', '妊娠期糖尿病', '子痫前期', '前置胎盘', '产后出血'],
        '公式里面的变量名': ['age_1st', 'weight_gain_1st', 'pre_bmi_1st', 'nation_1st', 'Hypothyroidism_1st', 'gestational_age_1st', 'birth_weight_1st', 'birth_length_1st', 'gender_1st', 'hysteromyoma_1st', 'Delivery_method_1st', 'NICU_1st', 'DM_1st', 'PROM_1st', 'GDM_1st', 'PE_1st', 'Placenta_Previa_1st', 'PPH_1st']
    }
    df_baseline = pd.DataFrame(baseline_cols)

    lab_cols = {
        '输入变量': ['PT', 'APTT', 'TT', 'Fib', 'ALT', 'AST', 'ALP', 'TP', 'ALB', 'TB', 'DB', 'TBA', 'Urea', 'Cr', 'UA', 'Ca', 'P', 'TSH', 'WBC', 'LY', 'NE', 'MO', 'BAS', 'LY_pctn', 'NE_pctn', 'MO_pctn', 'BAS_pctn', 'RBC', 'Hb', 'MCV', 'RDW_CV', 'RDW_SD', 'PLT', 'PCT'],
        '中文标签': ['凝血酶原时间', '活化部分凝血活酶', '凝血酶时间', '纤维蛋白原', '丙氨酸氨基转移酶', '天冬氨酸氨基转移酶', '碱性磷酸酶', '总蛋白', '白蛋白', '总胆红素', '直接胆红素', '总胆汁酸', '快速尿素', '肌酐', '尿酸', '钙', '磷', '促甲状腺素', '白细胞', '淋巴细胞绝对值', '中性粒细胞绝对值', '单核细胞绝对值', '嗜碱性粒细胞', '淋巴细胞百分数', '嗜中性粒细胞百分比', '单核细胞百分比', '嗜碱性粒细胞百分比', '红细胞', '血红蛋白', '平均红细胞体积', '红细胞分布宽度CV', '红细胞分布宽度SD', '血小板', '血小板压积'],
        '英文标签': ['Prothrombin time', 'Activated partial thromboplastin time', ...],
        '早孕期值': ['' for _ in range(34)],
        '中孕期值': ['' for _ in range(34)],
        '晚孕期值': ['' for _ in range(34)],
        '对应公式里面的变量名': ['PT_1st_f', 'APTT_1st_f', 'TT_1st_f', ...],
        'Unnamed: 7': ['PT_1st_s', 'APTT_1st_s', 'TT_1st_s', ...], # Corresponding mid-preg var names
        'Unnamed: 8': ['PT_1st_t', 'APTT_1st_t', 'TT_1st_t', ...], # Corresponding late-preg var names
    }
    # To save space, I will generate the full template programmatically.
    # The dictionary above is a sample.
    from itertools import repeat
    lab_data = pd.read_csv(io.StringIO("""
    输入变量,中文标签,早孕期值,中孕期值,晚孕期值,早孕期变量,中孕期变量,晚孕期变量
    PT,凝血酶原时间,,,PT_1st_f,PT_1st_s,PT_1st_t
    APTT,活化部分凝血活酶,,,APTT_1st_f,APTT_1st_s,APTT_1st_t
    TT,凝血酶时间,,,TT_1st_f,TT_1st_s,TT_1st_t
    Fib,纤维蛋白原,,,Fib_1st_f,Fib_1st_s,Fib_1st_t
    ALT,丙氨酸氨基转移酶,,,ALT_1st_f,ALT_1st_s,ALT_1st_t
    AST,天冬氨酸氨基转移酶,,,AST_1st_f,AST_1st_s,AST_1st_t
    ALP,碱性磷酸酶,,,ALP_1st_f,ALP_1st_s,ALP_1st_t
    TP,总蛋白,,,TP_1st_f,TP_1st_s,TP_1st_t
    ALB,白蛋白,,,ALB_1st_f,ALB_1st_s,ALB_1st_t
    TB,总胆红素,,,TB_1st_f,TB_1st_s,TB_1st_t
    DB,直接胆红素,,,DB_1st_f,DB_1st_s,DB_1st_t
    TBA,总胆汁酸,,,TBA_1st_f,TBA_1st_s,TBA_1st_t
    Urea,快速尿素,,,Urea_1st_f,Urea_1st_s,Urea_1st_t
    Cr,肌酐,,,Cr_1st_f,Cr_1st_s,Cr_1st_t
    UA,尿酸,,,UA_1st_f,UA_1st_s,UA_1st_t
    Ca,钙,,,Ca_1st_f,Ca_1st_s,Ca_1st_t
    P,磷,,,P_1st_f,P_1st_s,P_1st_t
    TSH,促甲状腺素,,,TSH_1st_f,TSH_1st_s,TSH_1st_t
    WBC,白细胞,,,WBC_1st_f,WBC_1st_s,WBC_1st_t
    LY,淋巴细胞绝对值,,,LY_1st_f,LY_1st_s,LY_1st_t
    NE,中性粒细胞绝对值,,,NE_1st_f,NE_1st_s,NE_1st_t
    MO,单核细胞绝对值,,,MO_1st_f,MO_1st_s,MO_1st_t
    BAS,嗜碱性粒细胞,,,BAS_1st_f,BAS_1st_s,BAS_1st_t
    LY_pctn,淋巴细胞百分数,,,LY_pctn_1st_f,LY_pctn_1st_s,LY_pctn_1st_t
    NE_pctn,嗜中性粒细胞百分比,,,NE_pctn_1st_f,NE_pctn_1st_s,NE_pctn_1st_t
    MO_pctn,单核细胞百分比,,,MO_pctn_1st_f,MO_pctn_1st_s,MO_pctn_1st_t
    BAS_pctn,嗜碱性粒细胞百分比,,,BAS_pctn_1st_f,BAS_pctn_1st_s,BAS_pctn_1st_t
    RBC,红细胞,,,RBC_1st_f,RBC_1st_s,RBC_1st_t
    Hb,血红蛋白,,,Hb_1st_f,Hb_1st_s,Hb_1st_t
    MCV,平均红细胞体积,,,MCV_1st_f,MCV_1st_s,MCV_1st_t
    RDW_CV,红细胞分布宽度CV,,,RDW_CV_1st_f,RDW_CV_1st_s,RDW_CV_1st_t
    RDW_SD,红细胞分布宽度SD,,,RDW_SD_1st_f,RDW_SD_1st_s,RDW_SD_1st_t
    PLT,血小板,,,PLT_1st_f,PLT_1st_s,PLT_1st_t
    PCT,血小板压积,,,PCT_1st_f,PCT_1st_s,PCT_1st_t
    """))
    df_lab = pd.DataFrame(lab_data).rename(columns={'早孕期变量': '对应公式里面的变量名', '中孕期变量': '对应公式里面的变量名_中', '晚孕期变量': '对应公式里面的变量名_晚'})
    df_lab_template = df_lab[['输入变量','中文标签','早孕期值','中孕期值','晚孕期值','对应公式里面的变量名','对应公式里面的变量名_中','对应公式里面的变量名_晚']]


    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        df_lab_template.to_excel(writer, sheet_name='数据上传表_实验室检查', index=False)
        df_baseline.to_excel(writer, sheet_name='数据上传表_基线特征', index=False)
    output_buffer.seek(0)

    return StreamingResponse(
        output_buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=prediction_template.xlsx"}
    )
