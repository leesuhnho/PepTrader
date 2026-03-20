## 프로젝트 펩펩

프로젝트 동기
주식 시장에서 데이터 기반의 예측 모델을 활용해 수익을 창출하고자 본 프로젝트를 시작하게 되었다. 단순한 감이나 뉴스에 의존하는 투자가 아닌, 머신러닝을 통해 보다 객관적이고 체계적인 매매 전략을 구축하는 것이 목표다.

LightGBM을 선택한 이유
초기에는 시계열 데이터 예측에 강점을 가진 LSTM(Long Short-Term Memory) 모델로 개발을 진행했다. 그러나 실제 개발 과정에서 몇 가지 한계에 부딪혔다.

과적합 문제: LSTM은 학습 데이터에 과적합되기 쉬운 구조적 특성이 있어, 실제 예측 성능이 기대에 미치지 못했다.
하이퍼파라미터 튜닝의 어려움: 레이어 수, 유닛 수, dropout 비율 등 튜닝해야 할 파라미터가 많고, 결과를 확인하기까지 시간이 오래 걸려 반복 실험이 비효율적이었다.
환경적 제약: Google Colab 환경에서 GPU 런타임 제한으로 인해 하루 실질적인 개발 시간이 1~2시간 남짓에 불과했다. GPU 없이는 LSTM 학습 속도가 현저히 떨어져 개발 사이클 자체가 느려지는 문제가 있었다.

이러한 이유로 CPU만으로도 빠르게 학습이 가능하고, 적은 데이터에서도 준수한 성능을 보여주며, 파라미터 튜닝이 비교적 직관적인 LightGBM(Light Gradient Boosting Machine) 으로 모델을 전환하게 되었다. LightGBM은 트리 기반 앙상블 모델로, 금융 데이터처럼 노이즈가 많은 환경에서도 과적합을 어느 정도 제어할 수 있다는 점에서 본 프로젝트에 적합한 선택이었다.

다운 받아야 하는 라이브러리는

```
!pip install pykrx pandas-ta tqdm requests-cache
!pip install aiolimiter
!pip install nest_asyncio
```
이렇습니다.
코렙 셀에다가 복붙하고 실행하시면 되요. 그리고 어떤 순서대로 실행을 시켜야 하나면
```
from google.colab import userdata
import os
my_secret_key = userdata.get('FRED_API_KEY')
os.environ['FRED_API_KEY'] = my_secret_key
# 병렬 worker 수
os.environ["RRE_TRADING_WORKERS"] = "6"
# 초당 최대 pykrx 호출 횟수
os.environ["RRE_TRADING_RPS"] = "6"
%run /content/drive/MyDrive/rre/datacl.py
```
```
import os

os.environ["RRE_DEBUG"] = "0"
# 1. 미래 데이터 누수 체크 (한 번 확인했으니 꺼도 되지만, 켜도 무방)
os.environ["RRE_LEAKCHECK"] = "1"

# 2. 피처 개수 엄격하게 관리 (데이터 품질 향상)
os.environ["STRICT_FEATURE_SCHEMA"] = "1"

# 실행
%run /content/drive/MyDrive/rre/run_preprocess.py

```
```
%run /content/drive/MyDrive/rre/rank.py
```
```
%env RRE_DEBUG=1

%run /content/drive/MyDrive/rre/run_train_lgbm.py
```
```
# 환경변수 설정
%env RRE_LOG_LEVEL=DEBUG

# 스크립트 실행
%run /content/drive/MyDrive/rre/backtest.py
```
### 폴더구조
```
drive/
└── MyDrive/
        └── rre/
            ├── _debug/
            ├── _diag/
            ├── btk/
            ├── data/
            ├── del/
            ├── model_lgbm/
            ├── backtest.py
            ├── datacl.py
            ├── ee.py
            ├── rank.py
            ├── run_preprocess.py
            ├── run_train_lgbm.py
            ├── rre_backtest_data/
            └── sample_data/

```




