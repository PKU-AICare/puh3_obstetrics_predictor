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