import io
import json
import math
import os
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import libarchive
import pandas as pd
import requests
from fastapi import (Body, Depends, FastAPI, File, Form, HTTPException,
                     Request, UploadFile, status)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import (Column, DateTime, ForeignKey, Integer, String, Text,
                        create_engine, func)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# --- Project Configuration ---
PROJECT_TITLE = "Assessment of Pregnancy-Related Disease Risks"
SQLALCHEMY_DATABASE_URL = "sqlite:///./predictions.db"
UPLOAD_DIRECTORY = Path("uploads")

# --- Create upload directory on startup ---
UPLOAD_DIRECTORY.mkdir(exist_ok=True)

# --- Database Setup ---
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Database Models ---
class VisitRecord(Base):
    """Logs every time a user visits the webpage."""
    __tablename__ = "visit_records"
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String, index=True)
    # Location for ranking: "中国 上海", "United States", "本地网络", "未知", etc.
    location_for_stats = Column(String, index=True)
    country = Column(String)  # Country name, e.g., "China", "本地", "未知"
    created_at = Column(DateTime, default=datetime.utcnow)

class PredictionUsageRecord(Base):
    """Logs each prediction event."""
    __tablename__ = "prediction_usage_records"
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String, index=True)
    location_for_stats = Column(String, index=True)
    country = Column(String)
    # 1 for single xlsx, N for N files in an archive
    prediction_count = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserDataRecord(Base):
    """Records information about user-uploaded files."""
    __tablename__ = "user_data_records"
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String, index=True)
    original_filename = Column(String)
    # Filename stored on the server, with timestamp
    saved_filename = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# --- Pydantic Models ---
class IpInfo(BaseModel):
    ip: str
    location_for_stats: str
    country: str

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
    unique_locations_count: int
    visit_ranking: List[LocationStat]
    usage_ranking: List[LocationStat]


# --- Disease Formulas & Mappings (Unchanged) ---
DISEASE_FORMULAS = {
    "DM": {"name_cn": "妊娠合并糖尿病","name_en": "Diabetes Mellitus","coeffs": {"age_1st": 0.1419173920,"weight_gain_1st": -0.1698330926,"pre_bmi_1st": 0.0938179096,"nation_1stminority": 0.7494179418,"gestational_age_1st": -0.4444326646,"birth_weight_1st": 0.0009155465,"NICU_1styes": -16.2611736194,"PROM_1styes": -0.6551172461,"PE_1styes": 0.8977422306,"PT_1st_f": -0.3014261392,"Fib_1st_f": 0.6122690819,"ALP_1st_f": 0.0338571978,"ALP_1st_t": -0.0104502311,"Urea_1st_t": 0.5477669654,"RBC_1st_t": 0.9667897398}},
    "GDM": {"name_cn": "妊娠期糖尿病","name_en": "Gestational Diabetes Mellitus","coeffs": {"age_1st": 0.0677375675,"pre_bmi_1st": 0.0303027085,"gestational_age_1st": -0.1469573905,"birth_weight_1st": 0.0003267695,"PROM_1styes": -0.1539029255,"ALT_1st_f": -0.0055781165,"TB_1st_f": -0.0215863388,"UA_1st_f": 0.0023808548,"P_1st_f": -0.6177690664,"WBC_1st_f": 0.0449929394,"RBC_1st_f": 0.3360527792,"AST_1st_t": -0.0232362771,"TP_1st_t": -0.0207373228,"Urea_1st_t": 0.1726151069,"NE_1st_t": -0.0476341578,"Hb_1st_t": 0.0175449679}},
    "HDP": {"name_cn": "妊娠期高血压疾病","name_en": "Hypertensive Disorders of Pregnancy","coeffs": {"weight_gain_1st": 0.054062796,"pre_bmi_1st": 0.193458307,"birth_weight_1st": -0.001137972,"NICU_1styes": -3.256899663,"PE_1styes": 3.858058925,"PT_1st_f": -0.300597209,"ALP_1st_f": 0.015503811,"DB_1st_f": 0.086570104,"WBC_1st_f": 0.077348983,"LY_pctn_1st_f": 0.300798922,"NE_pctn_1st_f": 0.274686573,"MO_pctn_1st_f": 0.207167696,"Hb_1st_f": 0.015825290,"PLT_1st_f": 0.002672225,"TT_1st_t": 0.167903201,"TBA_1st_t": -0.106703815,"Urea_1st_t": 0.458734254,"UA_1st_t": 0.004494712,"MO_pctn_1st_t": -0.143291082}},
    "HYPOT": {"name_cn": "妊娠合并甲状腺功能减退","name_en": "Hypothyroidism","coeffs": {"age_1st": 0.0521641548,"gender_1stmale": -0.3170674447,"DM_1styes": -13.5156229492,"PROM_1styes": -0.3242030304,"AST_1st_f": 0.0204702243,"TP_1st_f": 0.0449350360,"Cr_1st_f": 0.0006603378,"TSH_1st_f": 0.7445658561,"Cr_1st_t": 0.0215867070}},
    "LBW": {"name_cn": "低出生体重儿","name_en": "Low Birth Weight","coeffs": {"age_1st": 0.146896121,"pre_bmi_1st": 0.034844950,"nation_1stminority": 0.365470305,"gestational_age_1st": -0.099539808,"NICU_1styes": -0.902257912,"P_1st_f": -0.774395884,"RBC_1st_f": 0.319913915,"UA_1st_t": 0.002766374,"RDW_CV_1st_t": 0.097493680}},
    "LGA": {"name_cn": "大于胎龄儿","name_en": "Large for Gestational Age","coeffs": {"gestational_age_1st": -0.156033061,"birth_weight_1st": -0.001687150,"hysteromyoma_1styes": 0.517217070,"NICU_1styes": -2.457124930,"DM_1styes": 1.277533141,"PE_1styes": -1.937494470,"Placenta_Previa_1styes": 0.806095894,"TP_1st_f": 0.058642857,"MO_1st_f": 1.780582862,"MCV_1st_f": -0.044757368,"TT_1st_t": 0.175639553,"UA_1st_t": 0.003122707,"NE_1st_t": 0.121070269,"BAS_pctn_1st_t": 1.547644475}},
    "MYO": {"name_cn": "妊娠合并子宫肌瘤","name_en": "Hysteromyoma","coeffs": {"weight_gain_1st": 0.029515956,"pre_bmi_1st": 0.050811565,"gestational_age_1st": -0.349786153,"birth_weight_1st": 0.002634555,"gender_1stmale": -0.354175523,"GDM_1styes": 0.234191248,"PT_1st_f": -0.137972195,"Fib_1st_f": 0.160828605,"UA_1st_f": 0.002396116,"Urea_1st_t": -0.141457178,"P_1st_t": 0.560103021,"RDW_SD_1st_t": 0.020350271}},
    "NICU": {"name_cn": "新生儿重症监护室","name_en": "Neonatal Intensive Care Unit","coeffs": {"gestational_age_1st": -0.1725791002,"birth_weight_1st": -0.0007617624,"birth_length_1st": 0.1266719889,"hysteromyoma_1styes": 0.3451214024,"Delivery_method_1styes": 0.2854435335,"NICU_1styes": -14.4775925471,"GDM_1styes": 0.3088215468,"PPH_1styes": -0.4588597948,"ALP_1st_f": 0.0111946657,"DB_1st_f": -0.1173074496,"PT_1st_t": 0.4375821739,"TT_1st_t": 3.4309459560,"PTT_1st_t": -48.8590837230,"TP_1st_t": 0.0910124136,"ALB_1st_t": -0.1210340278,"RDW_SD_1st_t": -0.0412889009}},
    "PE": {"name_cn": "子痫前期","name_en": "Preeclampsia","coeffs": {"pre_bmi_1st": 0.1588087979,"birth_weight_1st": -0.0007503861,"gender_1stmale": -0.3531043810,"NICU_1styes": -2.0281813488,"PE_1styes": 2.2003865692,"PPH_1styes": -0.6496623637,"PT_1st_f": -0.4299248270,"ALP_1st_f": 0.0172409639,"BAS_pctn_1st_f": -1.4708132939,"PCT_1st_f": 3.4370473770,"Urea_1st_t": 0.3385603590}},
    "PP": {"name_cn": "前置胎盘","name_en": "Placenta Previa","coeffs": {"age_1st": 0.05105457,"birth_length_1st": 0.08991213,"Delivery_method_1styes": 0.82968740,"DM_1styes": -13.78513157,"PE_1styes": 0.70760017,"Placenta_Previa_1styes": 0.45416943,"APTT_1st_t": 0.07662945,"PTT_1st_t": -2.46431367,"ALB_1st_t": 0.05806045,"TB_1st_t": -0.07309937,"BAS_1st_t": -11.71195544,"RDW_CV_1st_t": 0.09310345}},
    "PPH": {"name_cn": "产后出血","name_en": "Postpartum Hemorrhage","coeffs": {"age_1st": 0.0304508163,"pre_bmi_1st": 0.0399908747,"nation_1stminority": -0.3793425995,"gestational_age_1st": -0.1463766559,"birth_weight_1st": 0.0004501316,"gender_1stmale": -0.2668906219,"Delivery_method_1styes": 0.2848715084,"PPH_1styes": 0.6550008395,"Fib_1st_t": -0.2234355038,"ALP_1st_t": -0.0035479574,"PCT_1st_t": -3.7543409206}},
    "PROM": {"name_cn": "胎膜早破","name_en": "Premature Rupture of Membranes","coeffs": {"pre_bmi_1st": -0.0277162724,"gestational_age_1st": -0.1455379006,"birth_weight_1st": 0.0003261704,"hysteromyoma_1styes": 0.3963949921,"Delivery_method_1styes": 0.8802072975,"PROM_1styes": 0.5543519130,"PE_1styes": -0.5436138359,"BAS_pctn_1st_t": 0.8631150211}},
    "PTB": {"name_cn": "早产","name_en": "Preterm Birth","coeffs": {"age_1st": 0.046681151,"pre_bmi_1st": 0.039969249,"gestational_age_1st": -0.474764824,"hysteromyoma_1styes": 0.518263100,"Delivery_method_1styes": -0.386175618,"NICU_1styes": -2.759545470,"PE_1styes": -0.906306622,"MCV_1st_f": -0.067096292,"PCT_1st_f": 3.676376419,"UA_1st_t": 0.002516177,"WBC_1st_t": 0.091190784,"BAS_pctn_1st_t": 1.204259664}},
    "SGA": {"name_cn": "小于胎龄儿","name_en": "Small for Gestational Age","coeffs": {"pre_bmi_1st": -0.064826410,"gestational_age_1st": 0.354456399,"birth_weight_1st": -0.002978985,"PE_1styes": -1.244787130,"Ca_1st_f": 2.337452681,"PCT_1st_f": 3.658695711}}
}


# --- FastAPI App Initialization ---
app = FastAPI(
    title=PROJECT_TITLE,
    description="A modern API for assessing pregnancy-related disease risks based on first pregnancy data.",
    version="3.2.0"
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

def get_ip_from_request(request: Request) -> str:
    """Extracts client IP from request headers."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host

def get_geo_info(ip: str) -> Dict[str, str]:
    """
    Fetches geographic location for a given IP.
    Handles localhost and unknown cases as per requirements.
    """
    # Handle localhost and private network IPs, which are valid for stats.
    # The ::1 is the IPv6 equivalent of 127.0.0.1.
    if not ip or ip in ("127.0.0.1", "localhost", "::1") or ip.startswith(("192.168.", "10.", "172.")):
        return {"ip": ip or "local", "location_for_stats": "本地网络", "country": "本地"}

    api_url = f"https://api.ip2location.io/?ip={ip}"
    try:
        # Set a reasonable timeout for the external API call
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()

        # The API might return a 200 OK with empty data for some IPs.
        if not data.get("country_name") and not data.get("country_code"):
            print(f"API returned no country data for {ip}")
            return {"ip": ip, "location_for_stats": "未知", "country": "未知"}

        country = data.get("country_name", "未知")
        region = data.get("region_name")  # Can be None if not found
        country_code = data.get("country_code")

        # Default location for stats is the country name
        location_for_stats = country

        # For China, be more specific for stats ranking.
        if country_code == "CN" and region:
            location_for_stats = f"中国 {region}"

        # If after all this, the country name is still unknown, unify the status.
        if country == "未知":
             location_for_stats = "未知"

        return {"ip": ip, "location_for_stats": location_for_stats, "country": country}

    except requests.exceptions.RequestException as e:
        print(f"Failed to get IP info for {ip}: {e}")
        return {"ip": ip, "location_for_stats": "未知", "country": "未知"}
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response for {ip}: {e}")
        return {"ip": ip, "location_for_stats": "未知", "country": "未知"}


# --- Core Logic Functions ---
def extract_archive(file_bytes: bytes, file_ext: str) -> List[tuple]:
    """Extracts .xlsx files from a given archive's bytes."""
    extracted_files = []
    # Using a temporary file is more robust for libarchive
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # libarchive is more versatile (rar, 7z, zip)
        with libarchive.file_reader(temp_path) as archive:
            for entry in archive:
                if (entry.name.lower().endswith('.xlsx') and
                    not entry.name.startswith(('__MACOSX', '.')) and not entry.isdir):
                    file_content = b''.join(entry.get_blocks())
                    filename = os.path.basename(entry.name)
                    patient_id = os.path.splitext(filename)[0]
                    extracted_files.append((patient_id, file_content))
    except Exception as e:
        # Fallback for zip files if libarchive fails for some reason
        print(f"libarchive extraction failed: {e}. Falling back to zipfile for .zip.")
        if file_ext == 'zip':
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                    for filename in zf.namelist():
                        if (filename.lower().endswith('.xlsx') and not filename.startswith(('__MACOSX', '.'))):
                            with zf.open(filename) as xlsx_file:
                                content = xlsx_file.read()
                                patient_id = os.path.splitext(os.path.basename(filename))[0]
                                extracted_files.append((patient_id, content))
            except Exception as zip_error:
                raise HTTPException(status_code=400, detail=f"Failed to extract zip archive: {zip_error}")
    finally:
        os.unlink(temp_path) # Clean up the temporary file

    if not extracted_files:
        raise HTTPException(status_code=400, detail="No .xlsx files found in the archive.")
    return extracted_files


def parse_excel_to_features(contents: bytes) -> Dict[str, float]:
    """Parses a single Excel file's bytes into a feature dictionary."""
    try:
        xls = pd.ExcelFile(io.BytesIO(contents))
        required_sheets = ['数据上传表_基线特征', '数据上传表_实验室检查']
        if not all(sheet in xls.sheet_names for sheet in required_sheets):
            raise ValueError(f"Excel must contain sheets: {', '.join(required_sheets)}")

        features = {}
        df_baseline = pd.read_excel(xls, sheet_name=required_sheets[0], header=0).fillna(0)
        for _, row in df_baseline.iterrows():
            var_name, value = row.get('variable_name'), row.get('Value / 数值', 0)
            if pd.notna(var_name):
                features[var_name.strip()] = float(value) if pd.notna(value) else 0.0

        df_lab = pd.read_excel(xls, sheet_name=required_sheets[1], header=0).fillna(0)
        for _, row in df_lab.iterrows():
            for period_key, var_key in [('Early P. / 早孕期', 'variable_name_f'), ('Mid P. / 中孕期', 'variable_name_s'), ('Late P. / 晚孕期', 'variable_name_t')]:
                var_name, value = row.get(var_key), row.get(period_key, 0)
                if pd.notna(var_name):
                    features[var_name.strip()] = float(value) if pd.notna(value) else 0.0
        return features
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error parsing Excel file: {str(e)}")


def calculate_probabilities(features: Dict[str, float]) -> List[Dict[str, Any]]:
    """Calculates disease probabilities and formats them as percentages."""
    results = []
    for abbr, data in DISEASE_FORMULAS.items():
        logit = sum(coeff * features.get(var, 0.0) for var, coeff in data["coeffs"].items())
        try:
            raw_probability = 1 / (1 + math.exp(-logit))
        except OverflowError:
            raw_probability = 1.0 if logit > 0 else 0.0

        percentage_probability = round(raw_probability * 100, 2)
        results.append({
            "disease_abbr": abbr,
            "disease_name_cn": data["name_cn"],
            "disease_name_en": data["name_en"],
            "probability": percentage_probability
        })
    return results

def save_uploaded_file(file: UploadFile, ip_info: IpInfo, db: Session):
    """Saves the file to disk with a timestamp and logs it to the database."""
    timestamp = int(time.time() * 1000)
    # Sanitize filename to prevent directory traversal
    safe_basename = Path(os.path.basename(file.filename))

    saved_filename_str = f"{safe_basename.stem}_{timestamp}{safe_basename.suffix}"
    saved_path = UPLOAD_DIRECTORY / saved_filename_str

    # Read file content once
    file_content = file.file.read()

    with open(saved_path, "wb") as buffer:
        buffer.write(file_content)

    # Reset file pointer in case it's needed again (e.g., for prediction logic)
    file.file.seek(0)

    # Log to UserDataRecord
    db_record = UserDataRecord(
        ip_address=ip_info.ip,
        original_filename=file.filename,
        saved_filename=saved_filename_str
    )
    db.add(db_record)


# --- API Endpoints ---

@app.get("/api/get-geo-info", response_model=IpInfo)
async def get_user_geo_info(request: Request):
    """
    Called by the frontend to get user's IP and geo-location for caching.
    """
    client_ip = get_ip_from_request(request)
    return get_geo_info(client_ip)

@app.post("/api/log-visit", status_code=status.HTTP_204_NO_CONTENT)
async def log_visit(ip_info: IpInfo, db: Session = Depends(get_db)):
    """Logs a visit record using data provided by the frontend."""
    try:
        visit = VisitRecord(
            ip_address=ip_info.ip,
            location_for_stats=ip_info.location_for_stats,
            country=ip_info.country
        )
        db.add(visit)
        db.commit()
    except Exception as e:
        db.rollback()
        # Don't raise an exception to the client for a simple logging failure
        print(f"Error logging visit for IP {ip_info.ip}: {e}")


@app.post("/api/predict-single", response_model=PatientPredictionResult)
async def predict_single(
    db: Session = Depends(get_db),
    ip_info_json: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .xlsx file.")

    try:
        ip_info = IpInfo.model_validate_json(ip_info_json)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid IP info format.")

    # --- Transactional Block ---
    try:
        # 1. Save uploaded file and log it in UserDataRecord
        save_uploaded_file(file, ip_info, db)

        # 2. Log prediction usage
        usage_record = PredictionUsageRecord(
            ip_address=ip_info.ip,
            location_for_stats=ip_info.location_for_stats,
            country=ip_info.country,
            prediction_count=1 # Single file
        )
        db.add(usage_record)

        # 3. Process prediction
        patient_id = os.path.splitext(file.filename)[0]
        contents = await file.read()
        features = parse_excel_to_features(contents)
        predictions = calculate_probabilities(features)

        db.commit() # Commit all changes if successful
    except Exception as e:
        db.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

    return PatientPredictionResult(patient_id=patient_id, predictions=predictions)


@app.post("/api/predict-batch", response_model=List[PatientPredictionResult])
async def predict_batch(
    db: Session = Depends(get_db),
    ip_info_json: str = Form(...),
    file: UploadFile = File(...)
):
    file_ext = file.filename.lower().split('.')[-1]
    if file_ext not in ['zip', 'rar', '7z']:
        raise HTTPException(status_code=400, detail="Invalid archive type. Use .zip, .rar, or .7z.")

    try:
        ip_info = IpInfo.model_validate_json(ip_info_json)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid IP info format.")

    all_results: List[PatientPredictionResult] = []

    # --- Transactional Block ---
    try:
        # 1. Save uploaded archive and log it in UserDataRecord
        save_uploaded_file(file, ip_info, db)

        # 2. Extract files from archive
        archive_bytes = await file.read()
        extracted_files = extract_archive(archive_bytes, file_ext)
        num_patients = len(extracted_files)

        if num_patients == 0:
            # The transaction will be rolled back, so no records are created
            raise HTTPException(status_code=400, detail="No processable .xlsx files found in archive.")

        # 3. Log prediction usage (one record for the whole batch)
        usage_record = PredictionUsageRecord(
            ip_address=ip_info.ip,
            location_for_stats=ip_info.location_for_stats,
            country=ip_info.country,
            prediction_count=num_patients
        )
        db.add(usage_record)

        # 4. Process each file
        for patient_id, xlsx_content in extracted_files:
            try:
                features = parse_excel_to_features(xlsx_content)
                predictions = calculate_probabilities(features)
                all_results.append(PatientPredictionResult(patient_id=patient_id, predictions=predictions))
            except Exception as e:
                # Log the error and continue with other files in the batch
                print(f"Skipping patient '{patient_id}' due to error: {e}")
                continue

        # If after processing, no patients were successful, we should inform the user.
        if not all_results:
            raise HTTPException(status_code=400, detail="No patients could be processed successfully from the archive.")


        db.commit() # Commit all changes if successful
    except Exception as e:
        db.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e}")

    return all_results


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats(db: Session = Depends(get_db)):
    """
    This endpoint correctly includes '本地网络' (Local Network) and '未知' (Unknown)
    in all statistics and rankings as requested.
    """
    total_visits = db.query(func.count(VisitRecord.id)).scalar() or 0

    total_predictions_scalar = db.query(func.sum(PredictionUsageRecord.prediction_count)).scalar()
    total_predictions = total_predictions_scalar if total_predictions_scalar is not None else 0

    # Count all distinct, non-null locations. This now includes "本地网络" and "未知"
    # as they are valid categories for statistics.
    unique_locations_count = (
        db.query(VisitRecord.location_for_stats)
        .filter(VisitRecord.location_for_stats.isnot(None))
        .distinct()
        .count()
    )

    # Ranking queries no longer filter out any specific location names,
    # allowing all recorded locations to appear in the rankings.
    visit_ranking_query = (
        db.query(VisitRecord.location_for_stats, func.count(VisitRecord.id).label("count"))
        .filter(VisitRecord.location_for_stats.isnot(None))
        .group_by(VisitRecord.location_for_stats)
        .order_by(func.count(VisitRecord.id).desc())
        .limit(10).all()
    )

    usage_ranking_query = (
        db.query(
            PredictionUsageRecord.location_for_stats,
            func.sum(PredictionUsageRecord.prediction_count).label("count")
        )
        .filter(PredictionUsageRecord.location_for_stats.isnot(None))
        .group_by(PredictionUsageRecord.location_for_stats)
        .order_by(func.sum(PredictionUsageRecord.prediction_count).desc())
        .limit(10).all()
    )

    return StatsResponse(
        total_visits=total_visits,
        total_predictions=total_predictions,
        unique_locations_count=unique_locations_count,
        visit_ranking=[LocationStat(location=loc, count=c) for loc, c in visit_ranking_query],
        usage_ranking=[LocationStat(location=loc, count=c) for loc, c in usage_ranking_query],
    )


@app.get("/api/download-template")
async def download_template():
    baseline_data = {'Feature Name (CN / EN)':['年龄 / Age (years)','孕期体重增长 / Weight gain during pregnancy (kg)','孕前BMI / Pre-pregnancy BMI (kg/m²)','民族 / Ethnicity (1: Minority, 0: Han)','甲状腺功能减退症 / Hypothyroidism (1: Yes, 0: No)','分娩孕周 / Gestational age at delivery (weeks)','出生体重 / Birth weight (g)','出生身长 / Birth length (cm)','新生儿性别 / Gender of newborn (1: Male, 0: Female)','子宫肌瘤 / Hysteromyoma (1: Yes, 0: No)','分娩方式 / Delivery method (1: Cesarean, 0: Vaginal)','新生儿转NICU / NICU admission (1: Yes, 0: No)','孕前糖尿病 / Pre-gestational DM (1: Yes, 0: No)','胎膜早破 / PROM (1: Yes, 0: No)','妊娠期糖尿病 / GDM (1: Yes, 0: No)','子痫前期 / Preeclampsia (1: Yes, 0: No)','前置胎盘 / Placenta Previa (1: Yes, 0: No)','产后出血 / Postpartum Hemorrhage (1: Yes, 0: No)'],'Value / 数值':[0]*18,'variable_name':['age_1st','weight_gain_1st','pre_bmi_1st','nation_1stminority','Hypothyroidism_1styes','gestational_age_1st','birth_weight_1st','birth_length_1st','gender_1stmale','hysteromyoma_1styes','Delivery_method_1styes','NICU_1styes','DM_1styes','PROM_1styes','GDM_1styes','PE_1styes','Placenta_Previa_1styes','PPH_1styes']}
    lab_vars = [('PT','凝血酶原时间','Prothrombin time'),('APTT','活化部分凝血活酶','Activated partial thromboplastin time'),('TT','凝血酶时间','Thrombin time'),('Fib','纤维蛋白原','Fibrinogen'),('ALT','丙氨酸氨基转移酶','Alanine aminotransferase'),('AST','天冬氨酸氨基转移酶','Aspartate aminotransferase'),('ALP','碱性磷酸酶','Alkaline phosphatase'),('TP','总蛋白','Total protein'),('ALB','白蛋白','Albumin'),('TB','总胆红素','Total bilirubin'),('DB','直接胆红素','Direct bilirubin'),('TBA','总胆汁酸','Total bile acid'),('Urea','快速尿素','Urea'),('Cr','肌酐','Creatinine'),('UA','尿酸','Uric acid'),('Ca','钙','Calcium'),('P','磷','Phosphorus'),('TSH','促甲状腺素','Thyroid-stimulating hormone'),('WBC','白细胞','White blood cell'),('LY','淋巴细胞绝对值','Lymphocyte absolute count'),('NE','中性粒细胞绝对值','Neutrophil absolute count'),('MO','单核细胞绝对值','Monocyte absolute count'),('BAS','嗜碱性粒细胞','Basophil'),('LY_pctn','淋巴细胞百分数','Lymphocyte percentage'),('NE_pctn','嗜中性粒细胞百分比','Neutrophil percentage'),('MO_pctn','单核细胞百分比','Monocyte percentage'),('BAS_pctn','嗜碱性粒细胞百分比','Basophil percentage'),('RBC','红细胞','Red blood cell'),('Hb','血红蛋白','Hemoglobin'),('MCV','平均红细胞体积','Mean corpuscular volume'),('RDW_CV','红细胞分布宽度CV','Red cell distribution width-CV'),('RDW_SD','红细胞分布宽度SD','Red cell distribution width-SD'),('PLT','血小板','Platelet'),('PCT','血小板压积','Plateletcrit'),('PTT','TT比率','TT ratio')]
    lab_data = {'Lab Test (CN / EN)':[f"{cn} / {en}" for _,cn,en in lab_vars],'Early P. / 早孕期':[0]*len(lab_vars),'Mid P. / 中孕期':[0]*len(lab_vars),'Late P. / 晚孕期':[0]*len(lab_vars),'variable_name_f':[f'{a}_1st_f' for a,_,_ in lab_vars],'variable_name_s':[f'{a}_1st_s' for a,_,_ in lab_vars],'variable_name_t':[f'{a}_1st_t' for a,_,_ in lab_vars]}

    df_baseline, df_lab = pd.DataFrame(baseline_data), pd.DataFrame(lab_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_baseline.to_excel(writer, sheet_name='数据上传表_基线特征', index=False)
        df_lab.to_excel(writer, sheet_name='数据上传表_实验室检查', index=False)
        for sheet_name, cols in [('数据上传表_基线特征', ['C']), ('数据上传表_实验室检查', ['E', 'F', 'G'])]:
            for col in cols:
                writer.sheets[sheet_name].column_dimensions[col].hidden = True
    output.seek(0)
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=Prediction_Template_Bilingual.xlsx"})


if __name__ == "__main__":
    import uvicorn
    # To run this app: `uvicorn backend:app --reload`
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
