# Housing Price Prediction API

房价预测模型 API，基于 FastAPI 和 Scikit-learn 构建的回归模型。

## 功能特性

- **单条预测** - 根据房屋特征返回房价预测
- **批量预测** - 一次最多支持 100 条预测
- **模型信息** - 查看模型系数和性能指标
- **健康检查** - 验证服务是否正常运行
- **容器化部署** - 支持 Docker 部署

## 技术栈

- Python 3.12+
- FastAPI
- Scikit-learn
- Pandas
- NumPy

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/predict` | POST | 单条房价预测 |
| `/predict/batch` | POST | 批量房价预测 |
| `/model-info` | GET | 模型系数和性能指标 |

## 本地运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train.py
```

### 3. 启动服务

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

### 4. 访问 API 文档

- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## Docker 部署

### 构建镜像

```bash
docker build -t house-price-api .
```

### 运行容器

```bash
docker run -p 8080:8080 house-price-api
```

## API 使用示例

### 单条预测

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "square_footage": 1500,
      "bedrooms": 3,
      "bathrooms": 2,
      "year_built": 2000,
      "lot_size": 6500,
      "distance_to_city_center": 5.0,
      "school_rating": 8.0
    }
  }'
```

### 批量预测

```bash
curl -X POST "http://localhost:8080/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "features_list": [
      {
        "square_footage": 1500,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 2000,
        "lot_size": 6500,
        "distance_to_city_center": 5.0,
        "school_rating": 8.0
      },
      {
        "square_footage": 2000,
        "bedrooms": 4,
        "bathrooms": 2.5,
        "year_built": 2010,
        "lot_size": 8500,
        "distance_to_city_center": 3.5,
        "school_rating": 9.0
      }
    ]
  }'
```

### 模型信息

```bash
curl http://localhost:8080/model-info
```

### 健康检查

```bash
curl http://localhost:8080/health
```

## 数据集字段

| 字段 | 类型 | 描述 |
|------|------|------|
| square_footage | float | 房屋面积（平方英尺） |
| bedrooms | float | 卧室数量 |
| bathrooms | float | 浴室数量 |
| year_built | float | 建造年份 |
| lot_size | float | 地块面积 |
| distance_to_city_center | float | 到市中心距离 |
| school_rating | float | 学校评分（0-10） |
| price | float | 房价（目标变量） |

## 模型性能

| 指标 | 值 |
|------|------|
| MAE | ~7916 |
| MSE | ~105617715 |
| RMSE | ~10277 |
| R² | ~0.98 |

## 项目结构

```
.
├── app.py                 # FastAPI 主应用
├── train.py               # 模型训练脚本
├── Dataset.csv            # 数据集
├── model.pkl             # 训练好的模型
├── model_metadata.pkl    # 模型元数据
├── requirements.txt      # 依赖包
└── Dockerfile            # 容器化配置
```

## 输入验证

API 对输入数据进行了严格的验证：

- `square_footage`: 必须大于 0
- `bedrooms`: 必须大于 0
- `bathrooms`: 必须大于 0
- `year_built`: 必须在 1800-2025 之间
- `lot_size`: 必须大于 0
- `distance_to_city_center`: 必须大于等于 0
- `school_rating`: 必须在 0-10 之间

## 许可证

MIT License