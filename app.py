from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
from typing import List, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用实例
app = FastAPI(
    title="Housing Price Prediction API",
    description="API for predicting housing prices using a linear regression model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量，用于存储加载的模型和元数据
model = None
metadata = None


class HousingFeatures(BaseModel):
    """房屋特征数据模型，用于验证输入数据"""
    square_footage: float = Field(..., gt=0, description="房屋面积（平方英尺）")
    bedrooms: float = Field(..., gt=0, description="卧室数量")
    bathrooms: float = Field(..., gt=0, description="卫生间数量")
    year_built: float = Field(..., ge=1800, le=2025, description="建造年份")
    lot_size: float = Field(..., gt=0, description="地块面积")
    distance_to_city_center: float = Field(..., ge=0, description="到市中心的距离")
    school_rating: float = Field(..., ge=0, le=10, description="学校评分（0-10）")

    model_config = {
        "json_schema_extra": {
            "example": {
                "square_footage": 1500,
                "bedrooms": 3,
                "bathrooms": 2,
                "year_built": 2000,
                "lot_size": 6500,
                "distance_to_city_center": 5.0,
                "school_rating": 8.0
            }
        }
    }


class PredictionRequest(BaseModel):
    """单个预测请求数据模型"""
    features: HousingFeatures


class BatchPredictionRequest(BaseModel):
    """批量预测请求数据模型"""
    features_list: List[HousingFeatures]


class PredictionResponse(BaseModel):
    """单个预测响应数据模型"""
    input: dict
    predicted_price: float


class BatchPredictionResponse(BaseModel):
    """批量预测响应数据模型"""
    predictions: List[PredictionResponse]


def load_model():
    """加载训练好的模型和元数据"""
    global model, metadata
    try:
        # 加载模型文件
        model = joblib.load("model.pkl")
        # 加载元数据文件
        metadata = joblib.load("model_metadata.pkl")
        logger.info("Model loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        raise RuntimeError("Model files not found. Please run train.py first.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model/metadata: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """应用启动时自动加载模型"""
    load_model()


@app.get("/health", tags=["System"])
def health_check():
    """健康检查接口，返回服务状态"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/model-info", tags=["Model"])
def model_info():
    """获取模型信息接口，包括系数、截距和性能指标"""
    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return {
        "model_type": metadata.get("model_type", "Linear Regression"),
        "features": metadata["features"],
        "coefficients": [round(c, 6) for c in metadata["coefficients"]],
        "intercept": round(metadata["intercept"], 4),
        "performance_metrics": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in metadata["metrics"].items()
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """单个房屋价格预测接口"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # 将输入特征转换为numpy数组
        input_data = np.array([[
            request.features.square_footage,
            request.features.bedrooms,
            request.features.bathrooms,
            request.features.year_built,
            request.features.lot_size,
            request.features.distance_to_city_center,
            request.features.school_rating
        ]])

        # 使用模型进行预测
        price = model.predict(input_data)[0]
        logger.info(f"Prediction made: {price}")

        # 返回预测结果，保留两位小数
        return PredictionResponse(
            input=request.features.model_dump(),
            predicted_price=round(float(price), 2)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """批量房屋价格预测接口，最多支持100个样本"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    # 验证批量请求不为空
    if not request.features_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch cannot be empty"
        )

    # 验证批量大小不超过100
    if len(request.features_list) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size cannot exceed 100"
        )

    try:
        # 将所有输入特征转换为numpy数组
        input_data = np.array([
            [f.square_footage, f.bedrooms, f.bathrooms, f.year_built,
             f.lot_size, f.distance_to_city_center, f.school_rating]
            for f in request.features_list
        ])

        # 使用模型进行批量预测
        prices = model.predict(input_data)
        logger.info(f"Batch prediction made: {len(prices)} items")

        # 构建预测响应列表
        predictions = [
            PredictionResponse(
                input=f.model_dump(),
                predicted_price=round(float(prices[i]), 2)
            )
            for i, f in enumerate(request.features_list)
        ]

        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    # 使用uvicorn启动服务器
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)