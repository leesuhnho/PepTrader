## 프로젝트 펩펩
고등학생이 취미로 만든 lightgbm프로젝트인데 쓰고싶으신 분들 알아서 쓰세요. 제가 이 주식 ai로 부자 될려고 했는데 쉽지 않네요...ㅠㅠㅠ

나중에 취업할때 포트폴리오로 쓸려고 올립니다. 참고로 고등학생이 만든거에요


일다 사용 방법에 대해 알려드리자면 구글 코렙에서 실행시키여 합니다

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
대충 요런 느낌으로 실행하시면 됩니다. 상세히 설명하기 귀찮아서 이렇게 했는데 알아서 잘 이해하시길 바랄게요
그리고 이거 피쳐하고 데이터 수집할때 미래데이터 누수 문제 발생하지 않게 제가 진짜 진짜 진짜 엄청나게 노력했거든요 그리고 정규화, 정상성도 잘 했고 그래서 이거 잘만 활용하시면 그래도 괜찮은 결과가 나오지 않을까? 하는 생각이 듭니다.
하지만 저는 실패했습니다. 이걸로 거의 3달동안 피쳐엔지니어링도 파라미터 튜닝 여러가지 진짜 별에 별거 다 해봤는데 승률이 20%대를 못넘더라구요.
ㅠㅠㅠ 마이스터고인데 학교에서 전공공부도안하고 했는데 솔직히 시간이 아깝습니다. 
그래도 이정도로 노력했는데 나중에 이거 포트폴리오로 쓰면 좋겠죠.

혹시 제 코드보고 저한테 관심 있으시거나 코드에 대한 피드백, 기술적 토론을 하고 싶으시다면
이메일로 연락주세요.
ish92730175@dsm.hs.kr

