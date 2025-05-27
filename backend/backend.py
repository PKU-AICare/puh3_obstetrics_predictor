# backend.py

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from pydantic import BaseModel
from datetime import datetime, timedelta
import requests
import os
from typing import List, Optional
import io
import pandas as pd
from starlette.background import BackgroundTask

# Database settings
SQLALCHEMY_DATABASE_URL = "sqlite:///./pcos_database.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models ---
class PCOSCalculation(Base):
    __tablename__ = "pcos_calculations"
    
    id = Column(Integer, primary_key=True, index=True)
    amh = Column(Float, nullable=True) # Allow nullable for flexibility if some inputs are optional
    menstrual_start = Column(Integer, nullable=True)
    menstrual_end = Column(Integer, nullable=True)
    bmi = Column(Float, nullable=True)
    androstenedione = Column(Float, nullable=True)
    probability = Column(Float)
    risk_level = Column(String)
    ip_address = Column(String)
    location = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class VisitRecord(Base):
    __tablename__ = "visit_records"
    
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String)
    location = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# --- Pydantic Models ---
class PCOSInput(BaseModel):
    amh: Optional[float] = None
    menstrual_start: Optional[int] = None
    menstrual_end: Optional[int] = None
    bmi: Optional[float] = None
    androstenedione: Optional[float] = None

class PCOSResult(BaseModel):
    probability: float
    risk_level: str
    risk_percentage: float

class LocationStat(BaseModel):
    location: str
    count: int
    percentage: float

class WorldMapDataItem(BaseModel):
    name: str
    value: List[float] # [lng, lat, count]
    count: int
    last_visit: datetime # Keep if useful, otherwise can be simplified

# FastAPI application
app = FastAPI(title="PCOS Prediction Tool API")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
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
    if not ip:
        return "未知地区,Unknown"
    # Handle common local/private IPs
    if ip == "127.0.0.1" or ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172.16.") or ip == "::1":
        return "局域网/本地,Local Network"
    
    try:
        # Using a free tier IP geolocation service. Consider rate limits and terms for production.
        response = requests.get(f"http://ip-api.com/json/{ip}?fields=status,message,country,countryCode,regionName,city", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'success':
            country = data.get('country', 'Unknown Country')
            region = data.get('regionName', 'Unknown Region')
            # city = data.get('city', 'Unknown City') # Available if needed

            if country == 'China':
                loc_parts = ["中国"]
                if region and region != 'Unknown Region':
                    loc_parts.append(region)
                # if city and city != 'Unknown City': # Could add city for more granularity
                #     loc_parts.append(city)
                return "".join(loc_parts) + f",{region if region != 'Unknown Region' else 'China'},China"

            elif country != 'Unknown Country':
                loc_str = country
                if region and region != 'Unknown Region':
                    loc_str += f", {region}"
                return loc_str
            else: # Both unknown
                 return "未知地区,Unknown"

        return "未知地区 (查询失败),Unknown (Lookup Failed)"
    except requests.exceptions.Timeout:
        print(f"IP Geolocation request timed out for {ip}")
        return "未知地区 (超时),Unknown (Timeout)"
    except requests.exceptions.RequestException as e:
        print(f"IP Geolocation request failed for {ip}: {e}")
        return "未知地区 (查询失败),Unknown (Lookup Failed)"
    except Exception as e:
        print(f"Error getting location from IP {ip}: {e}")
        return "未知地区,Unknown"


def calculate_pcos_probability(amh: Optional[float], menstrual_start: Optional[int], menstrual_end: Optional[int], 
                              bmi: Optional[float], androstenedione: Optional[float]) -> tuple[float, str]:
    # Simplified scoring logic based on common PCOS indicators.
    # This should be replaced with a clinically validated model if available.
    score = 0
    
    # AMH (Anti-Müllerian Hormone) ng/mL
    if amh is not None:
        if amh > 7: score += 30       # Very high
        elif amh > 4.5: score += 20   # High
        elif amh > 2.5: score += 10   # Slightly elevated
    
    # Menstrual Cycle Length (days)
    if menstrual_start is not None and menstrual_end is not None and menstrual_start > 0 and menstrual_end >= menstrual_start:
        avg_cycle_length = (menstrual_start + menstrual_end) / 2.0
        if avg_cycle_length > 35 or avg_cycle_length < 21: score += 25 # Irregular (Oligomenorrhea/Amenorrhea or Polymenorrhea)
        elif avg_cycle_length > 31: score += 15 # Longer end of normal
    elif menstrual_start is not None and menstrual_start > 35 : # Simplified if only one value representing typical cycle
        score += 25

    # BMI (Body Mass Index) kg/m²
    if bmi is not None:
        if bmi >= 30: score += 20     # Obese
        elif bmi >= 25: score += 10   # Overweight
    
    # Androstenedione nmol/L (Example, normal range varies)
    if androstenedione is not None:
        if androstenedione > 12: score += 25 # High
        elif androstenedione > 9: score += 15  # Elevated
        elif androstenedione > 7: score += 5   # Upper normal / Slightly elevated

    # Normalize probability (0 to 1), cap at 95% to avoid overconfidence from simple model
    probability = min(max(score / 100.0, 0.01), 0.95) # Ensure at least 1% if any positive score, max 95%
    
    if probability >= 0.60: risk_level = "高危"
    elif probability >= 0.20: risk_level = "中危"
    else: risk_level = "低危"
    
    return probability, risk_level


def get_coordinates_for_location(location: str) -> Optional[List[float]]:
    # Simplified coordinate map. In a real app, use a geocoding service or a comprehensive DB.
    # Keys should match the format from get_location_from_ip as closely as possible.
    coord_map = {
        # China
        "中国北京,Beijing,China": [116.4074, 39.9042],
        "中国上海,Shanghai,China": [121.4737, 31.2304],
        "中国广东,Guangdong,China": [113.2644, 23.1291], # For "中国广东"
        "中国浙江,Zhejiang,China": [120.1551, 30.2741], # For "中国浙江"
        "中国天津,Tianjin,China": [117.2010, 39.1330],
        "中国山东,Shandong,China": [117.1582, 36.8701],
        "中国陕西,Shaanxi,China": [108.9480, 34.2632],
        "中国,China": [104.1954, 35.8617], # Fallback for "中国,China" if region is not specific

        # Other countries (examples, ip-api might be more specific)
        "United States": [-95.7129, 37.0902],
        "United States, California": [-119.4179, 36.7783],
        "局域网/本地,Local Network": [0,0], # Can be filtered out on frontend if 0,0
        # Add more as needed, or implement dynamic geocoding
    }
    # Direct match
    if location in coord_map:
        return coord_map[location]
    
    # Try matching country part if region is included
    parts = location.split(',')
    if len(parts) > 1:
        country_part = parts[0].strip()
        if country_part in coord_map: # e.g. "United States"
            return coord_map[country_part]

    # Default for unknown if not found and not explicitly mapped to None
    # print(f"Coordinates not found for location: {location}") # For debugging
    return None


async def cleanup_temp_file(filepath: str):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Cleaned up temp file: {filepath}")
    except OSError as e:
        print(f"Error cleaning up temp file {filepath}: {e}")

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)
    print("Database tables created (if they didn't exist).")

# --- Middleware for IP Logging ---
@app.middleware("http")
async def log_visits_middleware(request: Request, call_next):
    # Skip logging for OPTIONS requests or specific non-user paths
    excluded_paths = ["/docs", "/openapi.json", "/favicon.ico", "/api/download-template"] # Add other static/utility paths
    if request.method == "OPTIONS" or any(request.url.path.startswith(p) for p in excluded_paths) or ".js" in request.url.path or ".css" in request.url.path:
        return await call_next(request)

    client_ip = request.headers.get("X-Forwarded-For") or request.client.host if request.client else "unknown_ip"
    
    # For local dev, if X-Forwarded-For is not set, client.host might be 127.0.0.1
    # This is fine, get_location_from_ip handles local IPs.
    
    db_session_for_visit = SessionLocal()
    try:
        location = get_location_from_ip(client_ip)
        visit_record = VisitRecord(
            ip_address=client_ip, 
            location=location,
            created_at=datetime.utcnow()
        )
        db_session_for_visit.add(visit_record)
        db_session_for_visit.commit()
    except Exception as e:
        db_session_for_visit.rollback()
        print(f"Error in visit logging middleware for IP {client_ip}: {e}")
    finally:
        db_session_for_visit.close()
    
    response = await call_next(request)
    return response

# --- API Endpoints ---
@app.post("/api/calculate", response_model=PCOSResult)
async def calculate_pcos_endpoint(input_data: PCOSInput, request: Request, db: Session = Depends(get_db)):
    # Basic validation: ensure at least one input is provided
    if all(value is None for value in input_data.dict().values()):
         raise HTTPException(status_code=400, detail="At least one input field must be provided for calculation.")

    probability, risk_level = calculate_pcos_probability(
        input_data.amh, input_data.menstrual_start, input_data.menstrual_end,
        input_data.bmi, input_data.androstenedione
    )
    
    client_ip = request.headers.get("X-Forwarded-For") or request.client.host if request.client else "unknown_ip"
    location = get_location_from_ip(client_ip)
    
    try:
        calculation_record = PCOSCalculation(
            amh=input_data.amh,
            menstrual_start=input_data.menstrual_start,
            menstrual_end=input_data.menstrual_end,
            bmi=input_data.bmi,
            androstenedione=input_data.androstenedione,
            probability=probability,
            risk_level=risk_level,
            ip_address=client_ip,
            location=location,
            created_at=datetime.utcnow()
        )
        db.add(calculation_record)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error saving calculation for IP {client_ip}: {e}")
        raise HTTPException(status_code=500, detail="Error saving calculation result.")

    return PCOSResult(
        probability=probability,
        risk_level=risk_level,
        risk_percentage=round(probability * 100, 3)
    )

@app.get("/api/usage-stats", response_model=List[LocationStat])
async def get_usage_stats(db: Session = Depends(get_db)):
    query_result = (
        db.query(
            PCOSCalculation.location,
            func.count(PCOSCalculation.id).label("count")
        )
        .group_by(PCOSCalculation.location)
        .order_by(func.count(PCOSCalculation.id).desc())
        .limit(6) # Match original display of 6 items
        .all()
    )
    
    total_calculations = db.query(func.count(PCOSCalculation.id)).scalar() or 0
    
    stats = []
    for loc, count in query_result:
        location_name = loc if loc else "未知地区,Unknown"
        stats.append(LocationStat(
            location=location_name.split(',')[0], # Show primary name for brevity
            count=count,
            percentage=round((count / total_calculations * 100), 2) if total_calculations > 0 else 0
        ))
    return stats

@app.get("/api/visit-stats", response_model=List[LocationStat])
async def get_visit_stats(db: Session = Depends(get_db)):
    query_result = (
        db.query(
            VisitRecord.location,
            func.count(VisitRecord.id).label("count")
        )
        .group_by(VisitRecord.location)
        .order_by(func.count(VisitRecord.id).desc())
        .limit(6) # Match original display of 6 items
        .all()
    )
    
    total_visits = db.query(func.count(VisitRecord.id)).scalar() or 0

    stats = []
    for loc, count in query_result:
        location_name = loc if loc else "未知地区,Unknown"
        stats.append(LocationStat(
            location=location_name.split(',')[0], # Show primary name for brevity
            count=count,
            percentage=round((count / total_visits * 100), 2) if total_visits > 0 else 0
        ))
    return stats

@app.get("/api/world-map-data", response_model=List[WorldMapDataItem])
async def get_world_map_data(db: Session = Depends(get_db)):
    # Use recent visit records for a "live" map feel
    # For example, locations from visits in the last N days, or top N locations by visit
    # Here, we take top locations from VisitRecord
    recent_locations_query = (
        db.query(
            VisitRecord.location,
            func.count(VisitRecord.id).label("visit_count"),
            func.max(VisitRecord.created_at).label("last_activity")
        )
        # .filter(VisitRecord.created_at >= datetime.utcnow() - timedelta(days=7)) # Example: last 7 days
        .group_by(VisitRecord.location)
        .order_by(func.count(VisitRecord.id).desc())
        .limit(50) # Limit number of points on map for performance
        .all()
    )
            
    map_data = []
    for location_str, count, last_visit_dt in recent_locations_query:
        if not location_str or location_str.startswith("未知地区") or location_str.startswith("局域网"): 
            continue 
        
        coords = get_coordinates_for_location(location_str)
        if coords and (coords[0] != 0 or coords[1] != 0): # Ensure coords are valid and not (0,0) for local
            map_data.append(WorldMapDataItem(
                name=location_str.split(',')[0], # Display primary name
                value=coords + [float(count)], # ECharts value: [lng, lat, dataValue]
                count=count,
                last_visit=last_visit_dt
            ))
    return map_data

@app.post("/api/batch-calculate")
async def batch_calculate_pcos(file: UploadFile = File(...), request: Request = None, db: Session = Depends(get_db)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="只支持Excel文件 (.xlsx, .xls)")

    try:
        contents = await file.read()
        # Explicitly use openpyxl for .xlsx, xlrd for .xls if needed, or let pandas infer
        try:
            df = pd.read_excel(io.BytesIO(contents), engine='openpyxl' if file.filename.endswith('.xlsx') else None)
        except Exception as e_read:
            raise HTTPException(status_code=400, detail=f"无法读取Excel文件: {e_read}. 请确保文件格式正确。")


        required_columns_map = {
            'AMH': 'amh',
            '月经周期开始': 'menstrual_start',
            '月经周期结束': 'menstrual_end',
            'BMI': 'bmi',
            '雄烯二酮': 'androstenedione'
        }
        # Check if all required columns are present (using Chinese names from template)
        missing_cols = [col_zh for col_zh in required_columns_map.keys() if col_zh not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Excel文件缺少必需的列: {', '.join(missing_cols)}")

        results_output = []
        client_ip = request.headers.get("X-Forwarded-For") or request.client.host if request.client else "batch_process_unknown_ip"
        batch_location = get_location_from_ip(client_ip)

        for index, row in df.iterrows():
            try:
                # Convert to numeric, coercing errors to NaN, then handle NaN
                input_data_dict = {}
                valid_row = True
                for col_zh, col_en in required_columns_map.items():
                    val = pd.to_numeric(row.get(col_zh), errors='coerce')
                    if pd.isna(val):
                         # Allow missing values, as PCOSInput fields are Optional
                         input_data_dict[col_en] = None
                    else:
                        # Ensure integers for cycle days
                        if col_en in ['menstrual_start', 'menstrual_end']:
                            input_data_dict[col_en] = int(val)
                        else:
                            input_data_dict[col_en] = float(val)
                
                pcos_input = PCOSInput(**input_data_dict)
                
                probability, risk_level = calculate_pcos_probability(
                    pcos_input.amh, pcos_input.menstrual_start, pcos_input.menstrual_end,
                    pcos_input.bmi, pcos_input.androstenedione
                )
                
                # Save each calculation from batch
                calculation_record = PCOSCalculation(
                    amh=pcos_input.amh, menstrual_start=pcos_input.menstrual_start,
                    menstrual_end=pcos_input.menstrual_end, bmi=pcos_input.bmi,
                    androstenedione=pcos_input.androstenedione, probability=probability,
                    risk_level=risk_level, ip_address=f"{client_ip}_batch_row_{index+1}",
                    location=batch_location, created_at=datetime.utcnow()
                )
                db.add(calculation_record)

                results_output.append({
                    "original_index": index + 1,
                    "probability": probability,
                    "risk_level": risk_level,
                    "risk_percentage": round(probability * 100, 3),
                    "status": "成功"
                })

            except (ValueError, TypeError) as ve:
                results_output.append({ "original_index": index + 1, "error": f"行 {index+1} 数据格式错误: {ve}", "status": "失败" })
            except Exception as e_row:
                 results_output.append({ "original_index": index + 1, "error": f"行 {index+1} 处理错误: {str(e_row)}", "status": "失败" })
        
        db.commit()
        return {"results": results_output}

    except HTTPException:
        db.rollback() # Rollback if HTTPException was raised before commit
        raise
    except pd.errors.EmptyDataError:
        db.rollback()
        raise HTTPException(status_code=400, detail="上传的Excel文件为空或无法解析。")
    except Exception as e:
        db.rollback()
        print(f"Batch calculation general error: {e}") # Log detailed error
        raise HTTPException(status_code=500, detail=f"文件处理或批量计算过程中发生未知错误。")


@app.get("/api/download-template")
async def download_template_excel():
    # Define template structure based on Chinese column names expected by batch upload
    data = {
        'AMH': [5.0, None],  # Example values, None to show optionality
        '月经周期开始': [28, None], 
        '月经周期结束': [35, None],
        'BMI': [22.0, None], 
        '雄烯二酮': [8.0, None]
    }
    df = pd.DataFrame(data)
    
    temp_file_path = "PCOS_批量计算模板.xlsx" # Keep it simple for this context
    
    try:
        df.to_excel(temp_file_path, index=False, engine='openpyxl')
    except Exception as e:
        print(f"Error creating Excel template: {e}")
        raise HTTPException(status_code=500, detail="无法生成模板文件。")
    
    return FileResponse(
        temp_file_path,
        filename="PCOS_批量计算模板.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        background=BackgroundTask(cleanup_temp_file, temp_file_path)
    )

if __name__ == "__main__":
    import uvicorn
    # Ensure `reload_dirs` points to where `backend.py` is if you want reload on backend changes specifically.
    # uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["."])
    uvicorn.run(app, host="0.0.0.0", port=8000) # Simpler run for direct execution