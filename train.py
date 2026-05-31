import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 定义数据集所需的列
REQUIRED_COLUMNS = [
    "square_footage", "bedrooms", "bathrooms", "year_built",
    "lot_size", "distance_to_city_center", "school_rating", "price"
]


def validate_data(df: pd.DataFrame) -> bool:
    """验证数据集是否符合要求"""
    # 检查是否缺少必要的列
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False

    # 检查是否存在空值
    if df.isnull().any().any():
        logger.error("Dataset contains null values")
        return False

    # 检查特征列是否存在非正值
    if (df[REQUIRED_COLUMNS[:-1]] <= 0).any().any():
        logger.error("Dataset contains non-positive values in features")
        return False

    return True


def load_and_train(data_path: str = "Dataset.csv", test_size: float = 0.2, random_state: int = 42):
    """加载数据集并训练线性回归模型"""
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found: {data_path}")
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # 加载数据集
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    logger.info(f"Dataset loaded: {len(df)} rows")

    # 验证数据
    if not validate_data(df):
        raise ValueError("Dataset validation failed")

    # 分离特征和目标变量
    features = REQUIRED_COLUMNS[:-1]
    target = REQUIRED_COLUMNS[-1]

    X = df[features]
    y = df[target]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # 创建并训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Model training complete")

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算模型性能指标
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }

    # 输出性能指标
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"MSE: {mse:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"R2: {r2:.4f}")

    # 保存模型到文件
    joblib.dump(model, "model.pkl")
    logger.info("Model saved to model.pkl")

    # 保存模型元数据
    metadata = {
        "model_type": "Linear Regression",
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_,
        "metrics": metrics,
        "features": features
    }
    joblib.dump(metadata, "model_metadata.pkl")
    logger.info("Metadata saved to model_metadata.pkl")

    return model, metadata


if __name__ == "__main__":
    # 执行训练流程
    load_and_train()