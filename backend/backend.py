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
    """Stores each prediction event."""
    __tablename__ = "prediction_records"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    ip_address = Column(String)
    location = Column(String)
    country = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    results_json = Column(Text) # Store prediction probabilities as JSON

class VisitRecord(Base):
    """Stores each unique visit to the site."""
    __tablename__ = "visit_records"
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String, index=True)
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
    visit_ranking_by_country: List[LocationStat]
    usage_ranking_by_country: List[LocationStat]

# --- Disease Formulas & Mappings (with English names) ---
# NOTE: Intercepts are assumed to be 0 or embedded in coefficients as per the original structure.
DISEASE_FORMULAS = {
    "DM": {"name_cn": "妊娠合并糖尿病", "name_en": "Diabetes Mellitus in Pregnancy", "coeffs": {"age_1st": 0.1419173920, "weight_gain_1st": -0.1698330926, "pre_bmi_1st": 0.0938179096, "nation_1stminority": 0.7494179418, "gestational_age_1st": -0.4444326646, "birth_weight_1st": 0.0009155465, "NICU_1styes": -16.2611736194, "PROM_1styes": -0.6551172461, "PE_1styes": 0.8977422306, "PT_1st_f": -0.3014261392, "Fib_1st_f": 0.6122690819, "ALP_1st_f": 0.0338571978, "ALP_1st_t": -0.0104502311, "Urea_1st_t": 0.5477669654, "RBC_1st_t": 0.9667897398}},
    "GDM": {"name_cn": "妊娠期糖尿病", "name_en": "Gestational Diabetes Mellitus", "coeffs": {"age_1st": 0.0677375675, "pre_bmi_1st": 0.0303027085, "gestational_age_1st": -0.1469573905, "birth_weight_1st": 0.0003267695, "PROM_1styes": -0.1539029255, "ALT_1st_f": -0.0055781165, "TB_1st_f": -0.0215863388, "UA_1st_f": 0.0023808548, "P_1st_f": -0.6177690664, "WBC_1st_f": 0.0449929394, "RBC_1st_f": 0.3360527792, "AST_1st_t": -0.0232362771, "TP_1st_t": -0.0207373228, "Urea_1st_t": 0.1726151069, "NE_1st_t": -0.0476341578, "Hb_1st_t": 0.0175449679}},
    "HDP": {"name_cn": "妊娠期高血压疾病", "name_en": "Hypertensive Disorders of Pregnancy", "coeffs": {"weight_gain_1st": 0.054062796, "pre_bmi_1st": 0.193458307, "birth_weight_1st": -0.001137972, "NICU_1styes": -3.256899663, "PE_1styes": 3.858058925, "PT_1st_f": -0.300597209, "ALP_1st_f": 0.015503811, "DB_1st_f": 0.086570104, "WBC_1st_f": 0.077348983, "LY_pctn_1st_f": 0.300798922, "NE_pctn_1st_f": 0.274686573, "MO_pctn_1st_f": 0.207167696, "Hb_1st_f": 0.015825290, "PLT_1st_f": 0.002672225, "TT_1st_t": 0.167903201, "TBA_1st_t": -0.106703815, "Urea_1st_t": 0.458734254, "UA_1st_t": 0.004494712, "MO_pctn_1st_t": -0.143291082}},
    "HYPOT": {"name_cn": "妊娠合并甲状腺功能减退", "name_en": "Hypothyroidism in Pregnancy", "coeffs": {"age_1st": 0.0521641548, "gender_1stmale": -0.3170674447, "DM_1styes": -13.5156229492, "PROM_1styes": -0.3242030304, "AST_1st_f": 0.0204702243, "TP_1st_f": 0.0449350360, "Cr_1st_f": 0.0006603378, "TSH_1st_f": 0.7445658561, "Cr_1st_t": 0.0215867070}},
    "LBW": {"name_cn": "低出生体重儿", "name_en": "Low Birth Weight", "coeffs": {"age_1st": 0.146896121, "pre_bmi_1st": 0.034844950, "nation_1stminority": 0.365470305, "gestational_age_1st": -0.099539808, "NICU_1styes": -0.902257912, "P_1st_f": -0.774395884, "RBC_1st_f": 0.319913915, "UA_1st_t": 0.002766374, "RDW_CV_1st_t": 0.097493680}},
    "LGA": {"name_cn": "大于胎龄儿", "name_en": "Large for Gestational Age", "coeffs": {"gestational_age_1st": -0.156033061, "birth_weight_1st": -0.001687150, "hysteromyoma_1styes": 0.517217070, "NICU_1styes": -2.457124930, "DM_1styes": 1.277533141, "PE_1styes": -1.937494470, "Placenta_Previa_1styes": 0.806095894, "TP_1st_f": 0.058642857, "MO_1st_f": 1.780582862, "MCV_1st_f": -0.044757368, "TT_1st_t": 0.175639553, "UA_1st_t": 0.003122707, "NE_1st_t": 0.121070269, "BAS_pctn_1st_t": 1.547644475}},
    "MYO": {"name_cn": "妊娠合并子宫肌瘤", "name_en": "Myoma in Pregnancy", "coeffs": {"weight_gain_1st": 0.029515956, "pre_bmi_1st": 0.050811565, "gestational_age_1st": -0.349786153, "birth_weight_1st": 0.002634555, "gender_1stmale": -0.354175523, "GDM_1styes": 0.234191248, "PT_1st_f": -0.137972195, "Fib_1st_f": 0.160828605, "UA_1st_f": 0.002396116, "Urea_1st_t": -0.141457178, "P_1st_t": 0.560103021, "RDW_SD_1st_t": 0.020350271}},
    "NICU": {"name_cn": "新生儿重症监护室", "name_en": "NICU Admission", "coeffs": {"gestational_age_1st": -0.1725791002, "birth_weight_1st": -0.0007617624, "birth_length_1st": 0.1266719889, "hysteromyoma_1styes": 0.3451214024, "Delivery_method_1styes": 0.2854435335, "NICU_1styes": -14.4775925471, "GDM_1styes": 0.3088215468, "PPH_1styes": -0.4588597948, "ALP_1st_f": 0.0111946657, "DB_1st_f": -0.1173074496, "PT_1st_t": 0.4375821739, "TT_1st_t": 3.4309459560, "PTT_1st_t": -48.8590837230, "TP_1st_t": 0.0910124136, "ALB_1st_t": -0.1210340278, "RDW_SD_1st_t": -0.0412889009}},
    "PE": {"name_cn": "子痫前期", "name_en": "Pre-eclampsia", "coeffs": {"pre_bmi_1st": 0.1588087979, "birth_weight_1st": -0.0007503861, "gender_1stmale": -0.3531043810, "NICU_1styes": -2.0281813488, "PE_1styes": 2.2003865692, "PPH_1styes": -0.6496623637, "PT_1st_f": -0.4299248270, "ALP_1st_f": 0.0172409639, "BAS_pctn_1st_f": -1.4708132939, "PCT_1st_f": 3.4370473770, "Urea_1st_t": 0.3385603590}},
    "PP": {"name_cn": "前置胎盘", "name_en": "Placenta Previa", "coeffs": {"age_1st": 0.05105457, "birth_length_1st": 0.08991213, "Delivery_method_1styes": 0.82968740, "DM_1styes": -13.78513157, "PE_1styes": 0.70760017, "Placenta_Previa_1styes": 0.45416943, "APTT_1st_t": 0.07662945, "PTT_1st_t": -2.46431367, "ALB_1st_t": 0.05806045, "TB_1st_t": -0.07309937, "BAS_1st_t": -11.71195544, "RDW_CV_1st_t": 0.09310345}},
    "PPH": {"name_cn": "产后出血", "name_en": "Postpartum Hemorrhage", "coeffs": {"age_1st": 0.0304508163, "pre_bmi_1st": 0.0399908747, "nation_1stminority": -0.3793425995, "gestational_age_1st": -0.1463766559, "birth_weight_1st": 0.0004501316, "gender_1stmale": -0.2668906219, "Delivery_method_1styes": 0.2848715084, "PPH_1styes": 0.6550008395, "Fib_1st_t": -0.2234355038, "ALP_1st_t": -0.0035479574, "PCT_1st_t": -3.7543409206}},
    "PROM": {"name_cn": "胎膜早破", "name_en": "Premature Rupture of Membranes", "coeffs": {"pre_bmi_1st": -0.0277162724, "gestational_age_1st": -0.1455379006, "birth_weight_1st": 0.0003261704, "hysteromyoma_1styes": 0.3963949921, "Delivery_method_1styes": 0.8802072975, "PROM_1styes": 0.5543519130, "PE_1styes": -0.5436138359, "BAS_pctn_1st_t": 0.8631150211}},
    "PTB": {"name_cn": "早产", "name_en": "Preterm Birth", "coeffs": {"age_1st": 0.046681151, "pre_bmi_1st": 0.039969249, "gestational_age_1st": -0.474764824, "hysteromyoma_1styes": 0.518263100, "Delivery_method_1styes": -0.386175618, "NICU_1styes": -2.759545470, "PE_1styes": -0.906306622, "MCV_1st_f": -0.067096292, "PCT_1st_f": 3.676376419, "UA_1st_t": 0.002516177, "WBC_1st_t": 0.091190784, "BAS_pctn_1st_t": 1.204259664}},
    "SGA": {"name_cn": "小于胎龄儿", "name_en": "Small for Gestational Age", "coeffs": {"pre_bmi_1st": -0.064826410, "gestational_age_1st": 0.354456399, "birth_weight_1st": -0.002978985, "PE_1styes": -1.244787130, "Ca_1st_f": 2.337452681, "PCT_1st_f": 3.658695711}},
}

# --- FastAPI App Initialization ---
app = FastAPI(
    title=PROJECT_TITLE,
    description="A modern API for assessing pregnancy-related disease risks.",
    version="3.0.0"
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

def get_location_from_ip(ip: str) -> Dict[str, str]:
    """Fetches geographic location from an IP address."""
    default_location = {"location": "Local Network", "country": "N/A"}
    # Filter out private/local IP addresses
    if not ip or ip in ("127.0.0.1", "::1") or ip.startswith(("192.168.", "10.", "172.16.")):
        return default_location
    try:
        # Using a free, no-key-required IP geolocation API
        response = requests.get(f"https://ipapi.co/{ip}/json/", timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            return {"location": "Unknown", "country": "Unknown"}

        country = data.get("country_name", "Unknown")
        region = data.get("region", "")

        # Specific formatting for China to include the province
        if data.get("country_code") == "CN":
            location = f"中国 {region}".strip() if region else "中国"
        else:
            location = country
        return {"location": location, "country": country}
    except requests.exceptions.RequestException:
        # Handle cases where the API is unreachable or times out
        return {"location": "Unknown", "country": "Unknown"}

def parse_excel_to_features(contents: bytes) -> Dict[str, float]:
    """Parses the bilingual Excel template into a feature dictionary."""
    try:
        xls = pd.ExcelFile(io.BytesIO(contents))
        required_sheets = ['数据上传表_基线特征', '数据上传表_实验室检查']
        if not all(sheet in xls.sheet_names for sheet in required_sheets):
            raise HTTPException(status_code=400, detail=f"Excel must contain sheets: {', '.join(required_sheets)}")

        df_baseline = pd.read_excel(xls, sheet_name=required_sheets[0], header=0)
        df_lab = pd.read_excel(xls, sheet_name=required_sheets[1], header=0)

        features = {}

        # Process baseline features using the hidden 'variable_name' column for robust mapping
        for _, row in df_baseline.iterrows():
            var_name = row.get('variable_name')
            value = row.get('Value / 数值')
            if pd.notna(var_name) and pd.notna(value):
                # Handle specific mappings as described in template notes
                if "nation_1stminority" in var_name:
                    features[var_name] = 1.0 if float(value) == 2 else 0.0 # 2 is minority
                elif "_1styes" in var_name or "male" in var_name: # Covers all 'yes' and 'male' flags
                    features[var_name] = 1.0 if float(value) == 1 else 0.0 # 1 is yes/male
                else: # Handle standard numeric values
                    features[var_name] = float(value)

        # Process lab features, mapping early/mid/late pregnancy values
        for _, row in df_lab.iterrows():
            # Early pregnancy (f)
            if 'variable_name_f' in df_lab.columns and pd.notna(row['variable_name_f']) and pd.notna(row['Early P. / 早孕期']):
                features[row['variable_name_f']] = float(row['Early P. / 早孕期'])
            # Mid pregnancy (s) - not currently used in formulas, but parsed for completeness
            if 'variable_name_s' in df_lab.columns and pd.notna(row['variable_name_s']) and pd.notna(row['Mid P. / 中孕期']):
                features[row['variable_name_s']] = float(row['Mid P. / 中孕期'])
            # Late pregnancy (t)
            if 'variable_name_t' in df_lab.columns and pd.notna(row['variable_name_t']) and pd.notna(row['Late P. / 晚孕期']):
                features[row['variable_name_t']] = float(row['Late P. / 晚孕期'])
        return features
    except Exception as e:
        # Catch-all for parsing errors (e.g., non-numeric data in value columns)
        raise HTTPException(status_code=422, detail=f"Error parsing Excel file: {e}")

def calculate_probabilities(features: Dict[str, float]) -> List[Dict[str, Any]]:
    """Calculates disease probabilities based on features and formulas."""
    results = []
    for abbr, data in DISEASE_FORMULAS.items():
        logit = 0.0
        # Sum the product of coefficients and their corresponding feature values
        # .get(var, 0.0) ensures that if a feature is missing from the input, it's treated as 0
        for var, coeff in data["coeffs"].items():
            logit += coeff * features.get(var, 0.0)

        # Calculate probability using the logistic function
        try:
            probability = 1 / (1 + math.exp(-logit))
        except OverflowError: # Handle extremely large or small logit values
            probability = 0.0 if logit < 0 else 1.0

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

    # Prepare a simplified dictionary for JSON storage in the database
    results_for_db = {p["disease_abbr"]: p["probability"] for p in predictions}

    db_record = PredictionRecord(
        patient_id=patient_id,
        ip_address=ip_info.get("ip_address"),
        location=ip_info.get("location"),
        country=ip_info.get("country"),
        results_json=json.dumps(results_for_db)
    )
    db.add(db_record)
    return predictions

# --- Middleware for Visit Logging ---
# A simple middleware to log initial site visits.
# For a Single Page Application (SPA), this typically logs the first load of index.html.
@app.middleware("http")
async def log_visits_middleware(request: Request, call_next):
    # Log visit if the root path is accessed. We assume this is the main app entry point.
    # We also check that it's not an API call to avoid double logging or logging internal calls.
    if request.method == "GET" and request.url.path == "/" and not request.url.path.startswith("/api/"):
        db: Session = next(get_db())
        try:
            # Get the real client IP, considering proxies
            client_ip = request.headers.get("x-forwarded-for") or request.client.host
            ip_info = get_location_from_ip(client_ip)

            visit = VisitRecord(
                ip_address=client_ip,
                location=ip_info.get("location"),
                country=ip_info.get("country")
            )
            db.add(visit)
            db.commit()
        except Exception:
            # If logging fails, roll back the transaction but don't crash the app
            db.rollback()
        finally:
            db.close()

    response = await call_next(request)
    return response

# --- API Endpoints ---

@app.post("/api/predict-single", response_model=PatientPredictionResult)
async def predict_single(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Endpoint for single patient prediction via .xlsx file upload."""
    if not file.filename.lower().endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .xlsx file.")

    client_ip = request.headers.get("x-forwarded-for") or request.client.host
    ip_info = get_location_from_ip(client_ip)
    ip_info["ip_address"] = client_ip

    patient_id = os.path.splitext(file.filename)[0]
    contents = await file.read()

    features = parse_excel_to_features(contents)
    predictions = process_and_log_prediction(db, ip_info, patient_id, features)

    db.commit()
    return PatientPredictionResult(patient_id=patient_id, predictions=predictions)

@app.post("/api/predict-batch", response_model=List[PatientPredictionResult])
async def predict_batch(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Endpoint for batch prediction via .zip file upload."""
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .zip archive.")

    # Note: For .rar or .7z support, you would need to install `rarfile` or `py7zr`
    # and add conditional logic based on the file extension. `zipfile` is in the standard library.

    client_ip = request.headers.get("x-forwarded-for") or request.client.host
    ip_info = get_location_from_ip(client_ip)
    ip_info["ip_address"] = client_ip

    all_results: List[PatientPredictionResult] = []

    try:
        with zipfile.ZipFile(io.BytesIO(await file.read())) as z:
            for filename in z.namelist():
                # Process only .xlsx files and ignore macOS resource fork folders
                if filename.lower().endswith('.xlsx') and not filename.startswith('__MACOSX'):
                    patient_id = os.path.splitext(os.path.basename(filename))[0]
                    with z.open(filename) as xlsx_file:
                        try:
                            contents = xlsx_file.read()
                            features = parse_excel_to_features(contents)
                            predictions = process_and_log_prediction(db, ip_info, patient_id, features)
                            all_results.append(PatientPredictionResult(patient_id=patient_id, predictions=predictions))
                        except Exception as e:
                            # Log and skip corrupted/invalid files within the batch
                            print(f"Skipping file {filename} in batch due to error: {e}")
                            continue
        db.commit()
    except zipfile.BadZipFile:
        db.rollback()
        raise HTTPException(status_code=400, detail="Invalid or corrupted ZIP file.")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during batch processing: {e}")

    if not all_results:
        raise HTTPException(status_code=400, detail="No valid .xlsx files were processed in the ZIP archive.")

    return all_results

@app.get("/api/stats", response_model=StatsResponse)
async def get_combined_stats(db: Session = Depends(get_db)):
    """Endpoint to retrieve usage and visit statistics."""
    total_visits = db.query(func.count(VisitRecord.id)).scalar() or 0
    total_predictions = db.query(func.count(PredictionRecord.id)).scalar() or 0
    unique_countries_count = db.query(func.count(func.distinct(PredictionRecord.country))).filter(PredictionRecord.country.isnot(None), PredictionRecord.country != "N/A").scalar() or 0

    # Top 10 countries by usage (predictions)
    usage_ranking_query = (
        db.query(PredictionRecord.country, func.count(PredictionRecord.id).label("count"))
        .filter(PredictionRecord.country.isnot(None), PredictionRecord.country.notin_(["Unknown", "N/A", "Local Network"]))
        .group_by(PredictionRecord.country)
        .order_by(func.count(PredictionRecord.id).desc())
        .limit(10)
        .all()
    )
    usage_ranking = [LocationStat(location=loc, count=count) for loc, count in usage_ranking_query]

    # Top 10 countries by visits
    visit_ranking_query = (
        db.query(VisitRecord.country, func.count(VisitRecord.id).label("count"))
        .filter(VisitRecord.country.isnot(None), VisitRecord.country.notin_(["Unknown", "N/A", "Local Network"]))
        .group_by(VisitRecord.country)
        .order_by(func.count(VisitRecord.id).desc())
        .limit(10)
        .all()
    )
    visit_ranking = [LocationStat(location=loc, count=count) for loc, count in visit_ranking_query]

    return StatsResponse(
        total_visits=total_visits,
        total_predictions=total_predictions,
        unique_countries_count=unique_countries_count,
        visit_ranking_by_country=visit_ranking,
        usage_ranking_by_country=usage_ranking,
    )

@app.get("/api/download-template")
async def download_template():
    """Generates and serves the bilingual Excel template."""
    # --- Baseline Features Sheet ---
    baseline_data = {
        'Feature Name (CN / EN)': [
            '年龄 / Age', '孕期体重增长 / Weight Gain During Pregnancy', '孕前BMI / Pre-pregnancy BMI',
            '民族 / Ethnicity', '首次妊娠甲状腺功能减退症 / Hypothyroidism in 1st Pregnancy', '首次分娩孕周 / Gestational Age at 1st Delivery',
            '首次分娩新生儿出生体重 / Birth Weight at 1st Delivery', '首次分娩新生儿出生身长 / Birth Length at 1st Delivery',
            '首次分娩新生儿性别 / Gender of Newborn at 1st Delivery', '首次妊娠合并子宫肌瘤 / Hysteromyoma in 1st Pregnancy',
            '首次分娩方式 / Delivery Method at 1st Delivery', '首次分娩新生儿转NICU / NICU Admission for 1st Newborn',
            '首次妊娠前合并糖尿病 / Pre-gestational DM in 1st Pregnancy', '首次妊娠胎膜早破 / PROM in 1st Pregnancy',
            '首次妊娠期糖尿病 / GDM in 1st Pregnancy', '首次妊娠子痫前期 / Pre-eclampsia (PE) in 1st Pregnancy',
            '首次妊娠前置胎盘 / Placenta Previa in 1st Pregnancy', '首次分娩产后出血 / Postpartum Hemorrhage (PPH) in 1st Delivery'
        ],
        'Value / 数值': ['' for _ in range(18)],
        'Notes / 说明': [
            'Years / 岁', 'kg / 公斤', 'kg/m²',
            '1 for Han, 2 for Minority / 1为汉族, 2为少数民族', '1 for Yes, 0 for No / 1是, 0否', 'e.g., 38+1/7 -> 38.14',
            'g / 克', 'cm / 厘米',
            '1 for Male, 0 for Female / 1男, 0女', '1 for Yes, 0 for No / 1是, 0否',
            '1 for Cesarean, 0 for Vaginal / 1剖宫产, 0顺产', '1 for Yes, 0 for No / 1是, 0否',
            '1 for Yes, 0 for No / 1是, 0否', '1 for Yes, 0 for No / 1是, 0否', '1 for Yes, 0 for No / 1是, 0否',
            '1 for Yes, 0 for No / 1是, 0否', '1 for Yes, 0 for No / 1是, 0否', '1 for Yes, 0 for No / 1是, 0否'
        ],
        'variable_name': [
            'age_1st', 'weight_gain_1st', 'pre_bmi_1st', 'nation_1stminority', 'Hypothyroidism_1styes',
            'gestational_age_1st', 'birth_weight_1st', 'birth_length_1st', 'gender_1stmale', 'hysteromyoma_1styes',
            'Delivery_method_1styes', 'NICU_1styes', 'DM_1styes', 'PROM_1styes', 'GDM_1styes', 'PE_1styes',
            'Placenta_Previa_1styes', 'PPH_1styes'
        ]
    }
    df_baseline = pd.DataFrame(baseline_data)

    # --- Lab Results Sheet ---
    lab_vars_raw = ['PT','APTT','TT','Fib','ALT','AST','GGT','LDH','ALP','TP','ALB','GLB','TB','DB','TBA','PA','Urea','Cr','UA','CysC','B2MG','CO2','Na','K','CL','Ca','P','Mg','CK','CKMB','GLU','HbA1c','TCHO','TG','HDLC','LDLC','ApoA1','ApoB','Lpa','TSH','T4','T3','FT4','FT3','TPOAb','TGAb','TMA','CRP','USCRP','WBC','LY','NE','MO','BAS','EOS','LY_pctn','NE_pctn','MO_pctn','BAS_pctn','EOS_pctn','RBC','Hb','Hct','MCV','MCH','MCHC','RDW_CV','RDW_SD','PLT','MPV','PCT','PDW','m_dbp','m_sbp','m_nowweight2']
    lab_vars_cn = ['凝血酶原时间','活化部分凝血活酶','凝血酶时间','纤维蛋白原','丙氨酸氨基转移酶','天冬氨酸氨基转移酶','快速γ谷氨酰转肽酶','乳酸脱氢酶','碱性磷酸酶','总蛋白','白蛋白','球蛋白','总胆红素','直接胆红素','总胆汁酸','前白蛋白','快速尿素','肌酐','尿酸','胱抑素C','β2微球蛋白','快速总二氧化碳','钠','钾','氯','钙','磷','镁','肌酸激酶','肌酸激酶同工酶','葡萄糖','糖化血红蛋白A1c','总胆固醇','甘油三酯','高密度脂蛋白胆固醇','低密度脂蛋白胆固醇','载脂蛋白A1','载脂蛋白B','脂蛋白a','促甲状腺素','总甲状腺素','总三碘甲状腺原氨酸','游离甲状腺素','游离三碘甲状腺原氨酸','抗甲状腺过氧化物酶抗体','抗甲状腺球蛋白抗体','抗甲状腺微粒体抗体','快速C-反应蛋白','超敏C反应蛋白','白细胞','淋巴细胞绝对值','中性粒细胞绝对值','单核细胞绝对值','嗜碱性粒细胞','嗜酸性粒细胞','淋巴细胞百分数','嗜中性粒细胞百分比','单核细胞百分比','嗜碱性粒细胞百分比','嗜酸性粒细胞百分比','红细胞','血红蛋白','红细胞压积','平均红细胞体积','平均血红蛋白含量','平均血红蛋白浓度','红细胞分布宽度CV','红细胞分布宽度SD','血小板','平均血小板体积','血小板压积','血小板分布宽度','舒张压','收缩压','本次门诊的体重']
    lab_vars_en = ['Prothrombin time','Activated partial thromboplastin time','Thrombin time','Fibrinogen','Alanine aminotransferase','Aspartate aminotransferase','Gamma-glutamyl transferase','Lactate dehydrogenase','Alkaline phosphatase','Total protein','Albumin','Globulin','Total bilirubin','Direct bilirubin','Total bile acid','Prealbumin','Urea','Creatinine','Uric acid','Cystatin C','β2-microglobulin','Total carbon dioxide','Sodium','Potassium','Chloride','Calcium','Phosphorus','Magnesium','Creatine kinase','Creatine kinase-MB','Glucose','Glycated hemoglobin A1c','Total cholesterol','Triglyceride','High-density lipoprotein cholesterol','Low-density lipoprotein cholesterol','Apolipoprotein A1','Apolipoprotein B','Lipoprotein(a)','Thyroid-stimulating hormone','Total thyroxine','Total triiodothyronine','Free thyroxine','Free triiodothyronine','Anti-thyroid peroxidase antibody','Anti-thyroglobulin antibody','Anti-thyroid microsomal antibody','C-reactive protein','High-sensitivity C-reactive protein','White blood cell','Lymphocyte absolute count','Neutrophil absolute count','Monocyte absolute count','Basophil','Eosinophil','Lymphocyte percentage','Neutrophil percentage','Monocyte percentage','Basophil percentage','Eosinophil percentage','Red blood cell','Hemoglobin','Hematocrit','Mean corpuscular volume','Mean corpuscular hemoglobin','Mean corpuscular hemoglobin concentration','Red cell distribution width-CV','Red cell distribution width-SD','Platelet','Mean platelet volume','Plateletcrit','Platelet distribution width','Diastolic blood pressure','Systolic blood pressure','Current body weight']

    lab_data = {
        'Lab Test (CN / EN)': [f"{cn} / {en}" for cn, en in zip(lab_vars_cn, lab_vars_en)],
        'Early P. / 早孕期': ['' for _ in lab_vars_raw],
        'Mid P. / 中孕期': ['' for _ in lab_vars_raw],
        'Late P. / 晚孕期': ['' for _ in lab_vars_raw],
        'variable_name_f': [f'{v}_1st_f' for v in lab_vars_raw],
        'variable_name_s': [f'{v}_1st_s' for v in lab_vars_raw],
        'variable_name_t': [f'{v}_1st_t' for v in lab_vars_raw]
    }
    df_lab = pd.DataFrame(lab_data)

    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        df_baseline.to_excel(writer, sheet_name='数据上传表_基线特征', index=False)
        df_lab.to_excel(writer, sheet_name='数据上传表_实验室检查', index=False)

        # Hide the variable name columns for a cleaner user experience
        ws_baseline = writer.sheets['数据上传表_基线特征']
        ws_baseline.column_dimensions['D'].hidden = True

        ws_lab = writer.sheets['数据上传表_实验室检查']
        for col_letter in ['E', 'F', 'G']: # Corresponds to variable_name columns
            ws_lab.column_dimensions[col_letter].hidden = True

    output_buffer.seek(0)

    return StreamingResponse(
        output_buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=prediction_template_bilingual.xlsx"}
    )