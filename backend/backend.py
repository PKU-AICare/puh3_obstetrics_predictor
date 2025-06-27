import io
import json
import math
import os
import zipfile
import rarfile
import py7zr
from datetime import datetime
from typing import Any, Dict, List, Optional

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
PROJECT_TITLE = "Assessment of Pregnancy-Related Disease Risks"
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
    country = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    input_features_json = Column(Text) # Store all input features
    results_json = Column(Text) # Store prediction probabilities

class VisitRecord(Base):
    __tablename__ = "visit_records"
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String, unique=False, index=True)
    location = Column(String)
    country = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- Pydantic Models (Data Transfer Objects) ---
class PredictionItem(BaseModel):
    disease_abbr: str
    disease_name_cn: str
    disease_name_en: str
    probability: float

class PatientPredictionResult(BaseModel):
    patient_id: str
    predictions: List[PredictionItem]

class LocationStat(BaseModel):
    location: str
    count: int

class StatsResponse(BaseModel):
    total_visits: int
    total_predictions: int
    unique_countries_count: int
    usage_ranking_by_country: List[LocationStat]
    visit_ranking_by_country: List[LocationStat]

# --- Disease Formulas & Mappings ---
# Correctly transcribed from the provided tables.
DISEASE_FORMULAS = {
    "DM": {"name_cn": "妊娠合并糖尿病", "name_en": "Diabetes Mellitus", "coeffs": {"age_1st": 0.1419173920, "weight_gain_1st": -0.1698330926, "pre_bmi_1st": 0.0938179096, "nation_1stminority": 0.7494179418, "gestational_age_1st": -0.4444326646, "birth_weight_1st": 0.0009155465, "NICU_1styes": -16.2611736194, "PROM_1styes": -0.6551172461, "PE_1styes": 0.8977422306, "PT_1st_f": -0.3014261392, "Fib_1st_f": 0.6122690819, "ALP_1st_f": 0.0338571978, "ALP_1st_t": -0.0104502311, "Urea_1st_t": 0.5477669654, "RBC_1st_t": 0.9667897398}},
    "GDM": {"name_cn": "妊娠期糖尿病", "name_en": "Gestational Diabetes Mellitus", "coeffs": {"age_1st": 0.0677375675, "pre_bmi_1st": 0.0303027085, "gestational_age_1st": -0.1469573905, "birth_weight_1st": 0.0003267695, "PROM_1styes": -0.1539029255, "ALT_1st_f": -0.0055781165, "TB_1st_f": -0.0215863388, "UA_1st_f": 0.0023808548, "P_1st_f": -0.6177690664, "WBC_1st_f": 0.0449929394, "RBC_1st_f": 0.3360527792, "AST_1st_t": -0.0232362771, "TP_1st_t": -0.0207373228, "Urea_1st_t": 0.1726151069, "NE_1st_t": -0.0476341578, "Hb_1st_t": 0.0175449679}},
    "HDP": {"name_cn": "妊娠期高血压疾病", "name_en": "Hypertensive Disorders of Pregnancy", "coeffs": {"weight_gain_1st": 0.054062796, "pre_bmi_1st": 0.193458307, "birth_weight_1st": -0.001137972, "NICU_1styes": -3.256899663, "PE_1styes": 3.858058925, "PT_1st_f": -0.300597209, "ALP_1st_f": 0.015503811, "DB_1st_f": 0.086570104, "WBC_1st_f": 0.077348983, "LY_pctn_1st_f": 0.300798922, "NE_pctn_1st_f": 0.274686573, "MO_pctn_1st_f": 0.207167696, "Hb_1st_f": 0.015825290, "PLT_1st_f": 0.002672225, "TT_1st_t": 0.167903201, "TBA_1st_t": -0.106703815, "Urea_1st_t": 0.458734254, "UA_1st_t": 0.004494712, "MO_pctn_1st_t": -0.143291082}},
    "HYPOT": {"name_cn": "妊娠合并甲状腺功能减退", "name_en": "Hypothyroidism in Pregnancy", "coeffs": {"age_1st": 0.0521641548, "gender_1stmale": -0.3170674447, "DM_1styes": -13.5156229492, "PROM_1styes": -0.3242030304, "AST_1st_f": 0.0204702243, "TP_1st_f": 0.0449350360, "Cr_1st_f": 0.0006603378, "TSH_1st_f": 0.7445658561, "Cr_1st_t": 0.0215867070}},
    "LBW": {"name_cn": "低出生体重儿", "name_en": "Low Birth Weight", "coeffs": {"age_1st": 0.146896121, "pre_bmi_1st": 0.034844950, "nation_1stminority": 0.365470305, "gestational_age_1st": -0.099539808, "NICU_1styes": -0.902257912, "P_1st_f": -0.774395884, "RBC_1st_f": 0.319913915, "UA_1st_t": 0.002766374, "RDW_CV_1st_t": 0.097493680}},
    "LGA": {"name_cn": "大于胎龄儿", "name_en": "Large for Gestational Age", "coeffs": {"gestational_age_1st": -0.156033061, "birth_weight_1st": -0.001687150, "hysteromyoma_1styes": 0.517217070, "NICU_1styes": -2.457124930, "DM_1styes": 1.277533141, "PE_1styes": -1.937494470, "Placenta_Previa_1styes": 0.806095894, "TP_1st_f": 0.058642857, "MO_1st_f": 1.780582862, "MCV_1st_f": -0.044757368, "TT_1st_t": 0.175639553, "UA_1st_t": 0.003122707, "NE_1st_t": 0.121070269, "BAS_pctn_1st_t": 1.547644475}},
    "MYO": {"name_cn": "妊娠合并子宫肌瘤", "name_en": "Hysteromyoma", "coeffs": {"weight_gain_1st": 0.029515956, "pre_bmi_1st": 0.050811565, "gestational_age_1st": -0.349786153, "birth_weight_1st": 0.002634555, "gender_1stmale": -0.354175523, "GDM_1styes": 0.234191248, "PT_1st_f": -0.137972195, "Fib_1st_f": 0.160828605, "UA_1st_f": 0.002396116, "Urea_1st_t": -0.141457178, "P_1st_t": 0.560103021, "RDW_SD_1st_t": 0.020350271}},
    "NICU": {"name_cn": "新生儿重症监护室", "name_en": "Neonatal Intensive Care Unit", "coeffs": {"gestational_age_1st": -0.1725791002, "birth_weight_1st": -0.0007617624, "birth_length_1st": 0.1266719889, "hysteromyoma_1styes": 0.3451214024, "Delivery_method_1styes": 0.2854435335, "NICU_1styes": -14.4775925471, "GDM_1styes": 0.3088215468, "PPH_1styes": -0.4588597948, "ALP_1st_f": 0.0111946657, "DB_1st_f": -0.1173074496, "PT_1st_t": 0.4375821739, "TT_1st_t": 3.4309459560, "PTT_1st_t": -48.8590837230, "TP_1st_t": 0.0910124136, "ALB_1st_t": -0.1210340278, "RDW_SD_1st_t": -0.0412889009}},
    "PE": {"name_cn": "子痫前期", "name_en": "Preeclampsia", "coeffs": {"pre_bmi_1st": 0.1588087979, "birth_weight_1st": -0.0007503861, "gender_1stmale": -0.3531043810, "NICU_1styes": -2.0281813488, "PE_1styes": 2.2003865692, "PPH_1styes": -0.6496623637, "PT_1st_f": -0.4299248270, "ALP_1st_f": 0.0172409639, "BAS_pctn_1st_f": -1.4708132939, "PCT_1st_f": 3.4370473770, "Urea_1st_t": 0.3385603590}},
    "PP": {"name_cn": "前置胎盘", "name_en": "Placenta Previa", "coeffs": {"age_1st": 0.05105457, "birth_length_1st": 0.08991213, "Delivery_method_1styes": 0.82968740, "DM_1styes": -13.78513157, "PE_1styes": 0.70760017, "Placenta_Previa_1styes": 0.45416943, "APTT_1st_t": 0.07662945, "PTT_1st_t": -2.46431367, "ALB_1st_t": 0.05806045, "TB_1st_t": -0.07309937, "BAS_1st_t": -11.71195544, "RDW_CV_1st_t": 0.09310345}},
    "PPH": {"name_cn": "产后出血", "name_en": "Postpartum Hemorrhage", "coeffs": {"age_1st": 0.0304508163, "pre_bmi_1st": 0.0399908747, "nation_1stminority": -0.3793425995, "gestational_age_1st": -0.1463766559, "birth_weight_1st": 0.0004501316, "gender_1stmale": -0.2668906219, "Delivery_method_1styes": 0.2848715084, "PPH_1styes": 0.6550008395, "Fib_1st_t": -0.2234355038, "ALP_1st_t": -0.0035479574, "PCT_1st_t": -3.7543409206}},
    "PROM": {"name_cn": "胎膜早破", "name_en": "Premature Rupture of Membranes", "coeffs": {"pre_bmi_1st": -0.0277162724, "gestational_age_1st": -0.1455379006, "birth_weight_1st": 0.0003261704, "hysteromyoma_1styes": 0.3963949921, "Delivery_method_1styes": 0.8802072975, "PROM_1styes": 0.5543519130, "PE_1styes": -0.5436138359, "BAS_pctn_1st_t": 0.8631150211}},
    "PTB": {"name_cn": "早产", "name_en": "Preterm Birth", "coeffs": {"age_1st": 0.046681151, "pre_bmi_1st": 0.039969249, "gestational_age_1st": -0.474764824, "hysteromyoma_1styes": 0.518263100, "Delivery_method_1styes": -0.386175618, "NICU_1styes": -2.759545470, "PE_1styes": -0.906306622, "MCV_1st_f": -0.067096292, "PCT_1st_f": 3.676376419, "UA_1st_t": 0.002516177, "WBC_1st_t": 0.091190784, "BAS_pctn_1st_t": 1.204259664}},
    "SGA": {"name_cn": "小于胎龄儿", "name_en": "Small for Gestational Age", "coeffs": {"pre_bmi_1st": -0.064826410, "gestational_age_1st": 0.354456399, "birth_weight_1st": -0.002978985, "PE_1styes": -1.244787130, "Ca_1st_f": 2.337452681, "PCT_1st_f": 3.658695711}},
}

# --- FastAPI App Initialization ---
app = FastAPI(
    title=PROJECT_TITLE,
    description="A modern API for assessing pregnancy-related disease risks based on first pregnancy data.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins, adjust in production for security
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

def get_ip_info(request: Request) -> Dict[str, str]:
    """Extracts client IP and fetches geographic location."""
    ip = request.headers.get("x-forwarded-for") or request.client.host
    default_location = {"ip": ip, "location": "Local Network", "country": "N/A"}
    if not ip or ip in ("127.0.0.1", "::1") or ip.startswith(("192.168.", "10.", "172.")):
        return default_location
    try:
        response = requests.get(f"https://ipapi.co/{ip}/json/", timeout=3)
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            return {"ip": ip, "location": "Unknown", "country": "Unknown"}

        country_name = data.get("country_name", "Unknown")
        region = data.get("region", "")

        location = f"中国 {region}".strip() if data.get("country_code") == "CN" and region else country_name

        return {"ip": ip, "location": location, "country": country_name}
    except requests.exceptions.RequestException:
        return {"ip": ip, "location": "Unknown", "country": "Unknown"}

@app.middleware("http")
async def log_visits_middleware(request: Request, call_next):
    """Logs a visit record for non-API GET requests to the root."""
    response = await call_next(request)
    # Log only the initial page load, not API calls or static assets.
    if request.method == "GET" and request.url.path == "/" and response.status_code == 200:
        db: Session = next(get_db())
        try:
            ip_info = get_ip_info(request)
            visit = VisitRecord(
                ip_address=ip_info["ip"],
                location=ip_info["location"],
                country=ip_info["country"]
            )
            db.add(visit)
            db.commit()
        except Exception as e:
            print(f"Error logging visit: {e}")
            db.rollback()
        finally:
            db.close()
    return response

# --- Core Logic Functions ---
def parse_excel_to_features(contents: bytes) -> Dict[str, float]:
    """Parses the bilingual Excel template into a feature dictionary using hidden variable names."""
    try:
        xls = pd.ExcelFile(io.BytesIO(contents))
        sheet_names = ['数据上传表_基线特征', '数据上传表_实验室检查']
        if not all(sheet in xls.sheet_names for sheet in sheet_names):
            raise ValueError(f"Excel must contain sheets: {', '.join(sheet_names)}")

        df_baseline = pd.read_excel(xls, sheet_name=sheet_names[0], header=0).fillna(0)
        df_lab = pd.read_excel(xls, sheet_name=sheet_names[1], header=0).fillna(0)

        features = {}

        for _, row in df_baseline.iterrows():
            var_name = row.get('variable_name')
            value = row.get('Value / 数值', 0)
            if var_name:
                features[var_name] = float(value)

        for _, row in df_lab.iterrows():
            if pd.notna(row['variable_name_f']) and pd.notna(row['Early P. / 早孕期']):
                features[row['variable_name_f']] = float(row['Early P. / 早孕期'])
            if pd.notna(row['variable_name_t']) and pd.notna(row['Late P. / 晚孕期']):
                features[row['variable_name_t']] = float(row['Late P. / 晚孕期'])

        return features
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error parsing Excel file: {e}")

def calculate_probabilities(features: Dict[str, float]) -> List[Dict[str, Any]]:
    """Calculates all 14 disease probabilities."""
    results = []
    for abbr, data in DISEASE_FORMULAS.items():
        logit = 0.0
        for var, coeff in data["coeffs"].items():
            logit += coeff * features.get(var, 0.0)
        try:
            probability = 1 / (1 + math.exp(-logit))
        except OverflowError:
            probability = 1.0 if logit > 0 else 0.0

        results.append({
            "disease_abbr": abbr,
            "disease_name_cn": data["name_cn"],
            "disease_name_en": data["name_en"],
            "probability": probability
        })
    return results

def process_and_log_prediction(db: Session, ip_info: dict, patient_id: str, features: dict):
    """Calculates predictions and logs the event to the database."""
    predictions = calculate_probabilities(features)
    results_for_db = {p["disease_abbr"]: p["probability"] for p in predictions}

    db_record = PredictionRecord(
        patient_id=patient_id,
        ip_address=ip_info["ip"],
        location=ip_info["location"],
        country=ip_info["country"],
        input_features_json=json.dumps(features),
        results_json=json.dumps(results_for_db)
    )
    db.add(db_record)
    return predictions

# --- API Endpoints ---
@app.post("/api/predict-single", response_model=PatientPredictionResult)
async def predict_single(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.lower().endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .xlsx file.")

    ip_info = get_ip_info(request)
    patient_id = os.path.splitext(file.filename)[0]
    contents = await file.read()

    features = parse_excel_to_features(contents)
    predictions = process_and_log_prediction(db, ip_info, patient_id, features)

    db.commit()
    return PatientPredictionResult(patient_id=patient_id, predictions=predictions)

@app.post("/api/predict-batch", response_model=List[PatientPredictionResult])
async def predict_batch(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_ext = file.filename.lower().split('.')[-1]
    if file_ext not in ['zip', 'rar', '7z']:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .zip, .rar, or .7z archive.")

    ip_info = get_ip_info(request)
    all_results: List[PatientPredictionResult] = []

    try:
        archive_bytes = io.BytesIO(await file.read())

        if file_ext == 'zip':
            with zipfile.ZipFile(archive_bytes) as archive:
                for filename in archive.namelist():
                    if filename.lower().endswith('.xlsx') and not filename.startswith('__MACOSX'):
                        patient_id = os.path.splitext(os.path.basename(filename))[0]
                        with archive.open(filename) as xlsx_file:
                            contents = xlsx_file.read()
                            features = parse_excel_to_features(contents)
                            predictions = process_and_log_prediction(db, ip_info, patient_id, features)
                            all_results.append(PatientPredictionResult(patient_id=patient_id, predictions=predictions))

        elif file_ext == 'rar':
            with rarfile.RarFile(archive_bytes) as archive:
                for filename in archive.namelist():
                    if filename.lower().endswith('.xlsx') and not filename.startswith('__MACOSX'):
                        patient_id = os.path.splitext(os.path.basename(filename))[0]
                        with archive.open(filename) as xlsx_file:
                            contents = xlsx_file.read()
                            features = parse_excel_to_features(contents)
                            predictions = process_and_log_prediction(db, ip_info, patient_id, features)
                            all_results.append(PatientPredictionResult(patient_id=patient_id, predictions=predictions))

        elif file_ext == '7z':
            with py7zr.SevenZipFile(archive_bytes) as archive:
                for filename in archive.getnames():
                    if filename.lower().endswith('.xlsx') and not filename.startswith('__MACOSX'):
                        patient_id = os.path.splitext(os.path.basename(filename))[0]
                        extracted_data = archive.read([filename])
                        if filename in extracted_data:
                            contents = extracted_data[filename].getvalue()  # BytesIO object
                            features = parse_excel_to_features(contents)
                            predictions = process_and_log_prediction(db, ip_info, patient_id, features)
                            all_results.append(PatientPredictionResult(patient_id=patient_id, predictions=predictions))

        db.commit()
    except zipfile.BadZipFile:
        db.rollback()
        raise HTTPException(status_code=400, detail="Invalid or corrupted ZIP file.")
    except rarfile.BadRarFile if rarfile else Exception:
        db.rollback()
        raise HTTPException(status_code=400, detail="Invalid or corrupted RAR file.")
    except py7zr.Bad7zFile if py7zr else Exception:
        db.rollback()
        raise HTTPException(status_code=400, detail="Invalid or corrupted 7z file.")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Batch processing error: {e}")

    if not all_results:
        raise HTTPException(status_code=400, detail="No valid .xlsx files found in the archive.")
    return all_results

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats(db: Session = Depends(get_db)):
    total_visits = db.query(func.count(VisitRecord.id)).scalar() or 0
    total_predictions = db.query(func.count(PredictionRecord.id)).scalar() or 0

    valid_country_filter = PredictionRecord.country.isnot(None) & ~PredictionRecord.country.in_(["Unknown", "N/A", "Local Network"])

    unique_countries_count = db.query(func.count(func.distinct(PredictionRecord.country))).filter(valid_country_filter).scalar() or 0

    usage_ranking_query = (
        db.query(PredictionRecord.country, func.count(PredictionRecord.id).label("count"))
        .filter(valid_country_filter)
        .group_by(PredictionRecord.country)
        .order_by(func.count(PredictionRecord.id).desc())
        .limit(10).all()
    )

    visit_ranking_query = (
        db.query(VisitRecord.country, func.count(VisitRecord.id).label("count"))
        .filter(VisitRecord.country.isnot(None) & ~VisitRecord.country.in_(["Unknown", "N/A", "Local Network"]))
        .group_by(VisitRecord.country)
        .order_by(func.count(VisitRecord.id).desc())
        .limit(10).all()
    )

    return StatsResponse(
        total_visits=total_visits,
        total_predictions=total_predictions,
        unique_countries_count=unique_countries_count,
        usage_ranking_by_country=[LocationStat(location=loc, count=c) for loc, c in usage_ranking_query],
        visit_ranking_by_country=[LocationStat(location=loc, count=c) for loc, c in visit_ranking_query],
    )

@app.get("/api/download-template")
async def download_template():
    """Generates and serves the bilingual Excel template with hidden variable names."""
    baseline_data = {
        'Feature Name (CN / EN)': [
            '年龄 / Age', '孕期体重增长 / Weight gain during pregnancy', '孕前BMI / Pre-pregnancy BMI', '民族 / Ethnicity',
            '首次妊娠甲状腺功能减退症 / Hypothyroidism', '首次分娩孕周 / Gestational age', '首次分娩新生儿出生体重 / Birth weight',
            '首次分娩新生儿出生身长 / Birth length', '首次分娩新生儿性别 / Gender of newborn', '首次妊娠合并子宫肌瘤 / Hysteromyoma',
            '首次分娩方式 / Delivery method', '首次分娩新生儿转NICU / NICU admission', '首次妊娠前合并糖尿病 / Pre-gestational DM',
            '首次妊娠胎膜早破 / PROM', '首次妊娠期糖尿病 / GDM', '首次妊娠子痫前期 / Preeclampsia (PE)',
            '首次妊娠前置胎盘 / Placenta Previa', '首次分娩产后出血 / Postpartum Hemorrhage (PPH)'
        ], 'Value / 数值': [''] * 18,
        'Notes / 说明': [
            'Years / 岁', 'kg', 'kg/m²', '1: Han, 2: Minority / 1: 汉族, 2: 少数民族', '1: Yes, 0: No / 1: 是, 0: 否',
            'Weeks (e.g., 38+1/7 -> 38.14)', 'g / 克', 'cm / 厘米', '1: Male, 0: Female / 1: 男, 0: 女', '1: Yes, 0: No / 1: 是, 0: 否',
            '1: Cesarean, 0: Vaginal / 1: 剖宫产, 0: 顺产', '1: Yes, 0: No / 1: 是, 0: 否', '1: Yes, 0: No / 1: 是, 0: 否',
            '1: Yes, 0: No / 1: 是, 0: 否', '1: Yes, 0: No / 1: 是, 0: 否', '1: Yes, 0: No / 1: 是, 0: 否',
            '1: Yes, 0: No / 1: 是, 0: 否', '1: Yes, 0: No / 1: 是, 0: 否'
        ],
        'variable_name': [
            'age_1st', 'weight_gain_1st', 'pre_bmi_1st', 'nation_1st', 'Hypothyroidism_1st', 'gestational_age_1st',
            'birth_weight_1st', 'birth_length_1st', 'gender_1st', 'hysteromyoma_1st', 'Delivery_method_1st', 'NICU_1st',
            'DM_1st', 'PROM_1st', 'GDM_1st', 'PE_1st', 'Placenta_Previa_1st', 'PPH_1st'
        ]
    }
    # Remap binary features to match formula variable names
    baseline_data['variable_name'] = [v.replace('_1st', '_1styes') if v not in ['age_1st', 'weight_gain_1st', 'pre_bmi_1st', 'nation_1st', 'gestational_age_1st', 'birth_weight_1st', 'birth_length_1st'] else v for v in baseline_data['variable_name']]
    baseline_data['variable_name'][3] = 'nation_1stminority' # Special case
    baseline_data['variable_name'][8] = 'gender_1stmale' # Special case

    lab_vars = [
        ('PT', '凝血酶原时间', 'Prothrombin time'), ('APTT', '活化部分凝血活酶', 'Activated partial thromboplastin time'), ('TT', '凝血酶时间', 'Thrombin time'),
        ('Fib', '纤维蛋白原', 'Fibrinogen'), ('ALT', '丙氨酸氨基转移酶', 'Alanine aminotransferase'), ('AST', '天冬氨酸氨基转移酶', 'Aspartate aminotransferase'),
        ('ALP', '碱性磷酸酶', 'Alkaline phosphatase'), ('TP', '总蛋白', 'Total protein'), ('ALB', '白蛋白', 'Albumin'), ('TB', '总胆红素', 'Total bilirubin'),
        ('DB', '直接胆红素', 'Direct bilirubin'), ('TBA', '总胆汁酸', 'Total bile acid'), ('Urea', '快速尿素', 'Urea'), ('Cr', '肌酐', 'Creatinine'),
        ('UA', '尿酸', 'Uric acid'), ('Ca', '钙', 'Calcium'), ('P', '磷', 'Phosphorus'), ('TSH', '促甲状腺素', 'Thyroid-stimulating hormone'),
        ('WBC', '白细胞', 'White blood cell'), ('LY', '淋巴细胞绝对值', 'Lymphocyte absolute count'), ('NE', '中性粒细胞绝对值', 'Neutrophil absolute count'),
        ('MO', '单核细胞绝对值', 'Monocyte absolute count'), ('BAS', '嗜碱性粒细胞', 'Basophil'), ('LY_pctn', '淋巴细胞百分数', 'Lymphocyte percentage'),
        ('NE_pctn', '嗜中性粒细胞百分比', 'Neutrophil percentage'), ('MO_pctn', '单核细胞百分比', 'Monocyte percentage'), ('BAS_pctn', '嗜碱性粒细胞百分比', 'Basophil percentage'),
        ('RBC', '红细胞', 'Red blood cell'), ('Hb', '血红蛋白', 'Hemoglobin'), ('MCV', '平均红细胞体积', 'Mean corpuscular volume'),
        ('RDW_CV', '红细胞分布宽度CV', 'Red cell distribution width-CV'), ('RDW_SD', '红细胞分布宽度SD', 'Red cell distribution width-SD'),
        ('PLT', '血小板', 'Platelet'), ('PCT', '血小板压积', 'Plateletcrit')
    ]
    lab_data = {
        'Lab Test (CN / EN)': [f"{cn} / {en}" for abbr, cn, en in lab_vars], 'Early P. / 早孕期': [''] * len(lab_vars),
        'Late P. / 晚孕期': [''] * len(lab_vars), 'variable_name_f': [f'{abbr}_1st_f' for abbr, _, _ in lab_vars],
        'variable_name_t': [f'{abbr}_1st_t' for abbr, _, _ in lab_vars]
    }

    df_baseline = pd.DataFrame(baseline_data)
    df_lab = pd.DataFrame(lab_data)

    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        df_baseline.to_excel(writer, sheet_name='数据上传表_基线特征', index=False)
        df_lab.to_excel(writer, sheet_name='数据上传表_实验室检查', index=False)

        ws_baseline = writer.sheets['数据上传表_基线特征']
        ws_baseline.column_dimensions['D'].hidden = True
        ws_lab = writer.sheets['数据上传表_实验室检查']
        ws_lab.column_dimensions['D'].hidden = True
        ws_lab.column_dimensions['E'].hidden = True

    output_buffer.seek(0)
    return StreamingResponse(
        output_buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=Prediction_Template_Bilingual.xlsx"}
    )