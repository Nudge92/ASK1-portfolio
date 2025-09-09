-----

# ASK1 저해제 pIC50 활성도 예측 AI 모델 개발

## 📜 프로젝트 개요

본 프로젝트는 신약 개발 초기 단계의 효율성을 높이기 위해, **ASK1 단백질 저해제의 pIC50 활성도를 예측하는 머신러닝 모델**을 개발하는 것을 목표로 합니다. 체계적인 데이터 처리, 피처 엔지니어링, 모델 최적화, 그리고 다양한 앙상블 기법을 적용하여 가장 성능이 뛰어난 예측 모델을 구축하고, SHAP 분석을 통해 모델의 예측 근거를 해석했습니다.

-----

## ✨ 주요 특징

  * **피처 엔지니어링**: 2D 피처(ECFP, 물리화학적 기술자)와 3D 피처(분자 도킹 점수)를 통합한 하이브리드 피처셋 사용
  * **모델 최적화**: `Optuna`를 이용한 1000회 이상의 하이퍼파라미터 튜닝 수행
  * **고급 앙상블**: 단순 앙상블을 넘어, **최적 가중치 앙상블** 및 **스태킹 앙상블** 등 고급 기법 적용
  * **모델 해석**: `SHAP` 분석을 통해 최종 모델의 예측 과정을 해석하고, 주요 피처의 영향력을 분석
  * **검증**: `Scaffold Split` 기반 5-Fold 교차 검증을 통해 모델의 일반화 성능을 신뢰성 있게 평가

-----

## 🚀 최종 결과

다양한 단일 모델과 앙상블 기법을 비교한 결과, **최적 가중치 앙상블(Optimal Weight Ensemble)** 모델이 \*\*가장 높은 R²(0.426)\*\*를 기록하며 종합적으로 가장 우수한 성능을 보였습니다.

| 모델 | 평균 RMSE | 평균 R² |
| :--- | :--- | :--- |
| **최적 가-중치 앙상블** | **0.887** | **0.426** |
| 스태킹 앙상블 | 0.886 | 0.391 |
| LightGBM (Optimized) | 0.885 | 0.340 |

### SHAP 모델 해석

최고 성능 단일 모델인 LightGBM에 대한 SHAP 분석 결과, **`docking_score`**, **`ECFP_980`**, **`TPSA`** 등이 pIC50 예측에 중요한 영향을 미치는 것을 확인했습니다. 이는 모델이 화학적으로 유의미한 패턴을 학습했음을 시사합니다.

-----

## 🛠️ 기술 스택

  * **언어**: Python
  * **주요 라이브러리**: Pandas, NumPy, RDKit, Scikit-learn, LightGBM, XGBoost, Optuna, SHAP, Matplotlib
  * **도구**: Jupyter Lab, Conda

-----

## 📁 프로젝트 구조

```
ASK1_portfolio/
├── data/
│   └── processed/
│       └── ask1_data_with_docking_scores.csv
├── models/
│   ├── best_lgbm_model.pkl
│   └── best_xgb_model.pkl
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Modeling_and_Tuning.ipynb
│   ├── 03_Train_and_Save_Models.ipynb
│   └── 04_Final_Analysis_and_Conclusion.ipynb
├── outputs/
│   └── figures/
│       ├── final_model_comparison.png
│       └── shap_summary_plot.png
├── requirements.txt
└── README.md
```

-----

## 📖 실행 방법

#### 1\. Conda로 핵심 라이브러리 설치

의존성이 복잡한 과학 컴퓨팅 라이브러리는 Conda를 통해 먼저 설치합니다.

```bash
# ask1_project_new 라는 이름의 새 환경을 만들고 활성화합니다.
conda create -n ask1_project_new python=3.9 -y
conda activate ask1_project_new

# Conda-forge 채널에서 핵심 라이브러리들을 설치합니다.
conda install -c conda-forge rdkit openbabel ipykernel jupyter pandas matplotlib seaborn plotly python-kaleido
```

#### 2\. Pip으로 나머지 라이브러리 설치

핵심 라이브러리 설치 후, `requirements.txt` 파일을 이용해 나머지 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

#### 3\. Jupyter Notebook 실행

`notebooks` 폴더로 이동하여 아래 순서대로 노트북을 실행합니다.

  * `01_Data_Preprocessing.ipynb`: 데이터 전처리
  * `02_Modeling_and_Tuning.ipynb`: 개별 모델 실험 및 튜닝
  * `03_Train_and_Save_Models.ipynb`: 최종 모델 훈련 및 저장
  * `04_Final_Analysis_and_Conclusion.ipynb`: 최종 분석 및 시각화