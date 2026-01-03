import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Load artifacts
model = joblib.load('models/ridge_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
lot_frontage_median = joblib.load('models/lot_frontage_median.pkl')
electrical_mode = joblib.load('models/electrical_mode.pkl')

# Encoding maps (same as training)
quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
exposure_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
bsmt_fin_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
garage_fin_map = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
functional_map = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}

app = FastAPI(title="House Price Prediction API", version="1.0")

class HouseInput(BaseModel):
    MSSubClass: int = 60
    MSZoning: str = "RL"
    LotFrontage: Optional[float] = None
    LotArea: int = 10000
    Street: str = "Pave"
    LotShape: str = "Reg"
    LandContour: str = "Lvl"
    Utilities: str = "AllPub"
    LotConfig: str = "Inside"
    LandSlope: str = "Gtl"
    Neighborhood: str = "NAmes"
    Condition1: str = "Norm"
    Condition2: str = "Norm"
    BldgType: str = "1Fam"
    HouseStyle: str = "2Story"
    OverallQual: int = 5
    OverallCond: int = 5
    YearBuilt: int = 2000
    YearRemodAdd: int = 2005
    RoofStyle: str = "Gable"
    RoofMatl: str = "CompShg"
    Exterior1st: str = "VinylSd"
    Exterior2nd: str = "VinylSd"
    MasVnrType: Optional[str] = None
    MasVnrArea: Optional[float] = None
    ExterQual: str = "TA"
    ExterCond: str = "TA"
    Foundation: str = "PConc"
    BsmtQual: Optional[str] = None
    BsmtCond: Optional[str] = None
    BsmtExposure: Optional[str] = None
    BsmtFinType1: Optional[str] = None
    BsmtFinSF1: float = 0
    BsmtFinType2: Optional[str] = None
    BsmtFinSF2: float = 0
    BsmtUnfSF: float = 0
    TotalBsmtSF: float = 1000
    Heating: str = "GasA"
    HeatingQC: str = "TA"
    CentralAir: str = "Y"
    Electrical: Optional[str] = None
    FirstFlrSF: int = 1000  # 1stFlrSF
    SecondFlrSF: int = 500  # 2ndFlrSF
    LowQualFinSF: int = 0
    GrLivArea: int = 1500
    BsmtFullBath: int = 0
    BsmtHalfBath: int = 0
    FullBath: int = 2
    HalfBath: int = 0
    BedroomAbvGr: int = 3
    KitchenAbvGr: int = 1
    KitchenQual: str = "TA"
    TotRmsAbvGrd: int = 6
    Functional: str = "Typ"
    Fireplaces: int = 0
    GarageType: Optional[str] = None
    GarageYrBlt: Optional[float] = None
    GarageFinish: Optional[str] = None
    GarageCars: int = 2
    GarageArea: float = 500
    GarageQual: Optional[str] = None
    GarageCond: Optional[str] = None
    PavedDrive: str = "Y"
    WoodDeckSF: int = 0
    OpenPorchSF: int = 0
    EnclosedPorch: int = 0
    ThreeSsnPorch: int = 0  # 3SsnPorch
    ScreenPorch: int = 0
    PoolArea: int = 0
    MiscVal: int = 0
    MoSold: int = 6
    YrSold: int = 2010
    SaleType: str = "WD"
    SaleCondition: str = "Normal"

class HouseOutput(BaseModel):
    predicted_price: float
    price_range_low: float
    price_range_high: float

def preprocess_input(house: HouseInput) -> pd.DataFrame:
    """Replicate training preprocessing steps on input data."""

    data = house.model_dump()

    data['1stFlrSF'] = data.pop('FirstFlrSF')
    data['2ndFlrSF'] = data.pop('SecondFlrSF')
    data['3SsnPorch'] = data.pop('ThreeSsnPorch')

    df = pd.DataFrame([data])

    # Missing values
    cat_fill_none = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                     'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                     'BsmtFinType2', 'MasVnrType']
    for col in cat_fill_none:
        df[col] = df[col].fillna('None')
    
    # Numeric fill with 0
    num_fill_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea',
                     'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                     'BsmtFullBath', 'BsmtHalfBath']
    for col in num_fill_zero:
        df[col] = df[col].fillna(0)
    
    # LotFrontage with median
    df['LotFrontage'] = df['LotFrontage'].fillna(lot_frontage_median)
    
    # Electrical with mode
    df['Electrical'] = df['Electrical'].fillna(electrical_mode)

    # Feature engineering
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    
    # Drop columns (same as training)
    df = df.drop(columns=['YearBuilt', 'YearRemodAdd'])

    # Ordinal encoding
    quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                    'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']
    for col in quality_cols:
        df[col] = df[col].map(quality_map)
    
    df['BsmtExposure'] = df['BsmtExposure'].map(exposure_map)
    df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmt_fin_map)
    df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmt_fin_map)
    df['GarageFinish'] = df['GarageFinish'].map(garage_fin_map)
    df['Functional'] = df['Functional'].map(functional_map)
    
    # One-hot encoding
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Align with training features
    # Add missing columns with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Keep only training columns, in same order
    df = df[feature_columns]
    
    return df

@app.get("/")
async def root():
    return {"message": "Welcome to the House Price Prediction API!"}

@app.post("/predict", response_model=HouseOutput)
async def predict(house: HouseInput):
    # Preprocess input
    input_df = preprocess_input(house)

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict log price
    log_price_pred = model.predict(input_scaled)[0]

    # Convert to actual price
    price_pred = float(np.expm1(log_price_pred))

    return HouseOutput(
        predicted_price=round(price_pred, 2),
        price_range_low=round(price_pred * 0.85, 2),
        price_range_high=round(price_pred * 1.15, 2)
    )