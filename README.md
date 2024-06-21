# 프로젝트 제목

Paad (Power plant acoustic anomaly detection)

## 프로젝트 설명

이 프로젝트는 MIMII, DCASE 데이터셋을 처리하여 기계 학습 모델 학습에 사용할 수 있는 형태로 준비하는 스크립트를 포함하고 있습니다.<br>
`prepare_dataset.py`는 특정 기계의 소리 데이터를 분류하고, 데이터셋의 디렉토리 경로와 레이블을 추출하는 기능을 제공합니다.

## 설치 방법

이 프로젝트를 사용하기 위해 필요한 사전 요구 사항은 Python 3.x입니다. 다음 단계에 따라 프로젝트를 설치하십시오:

```bash
git clone https://yourprojectlink.git
cd yourprojectdirectory
pip install -r requirements.txt
```

### 사용 방법
prepare_dataset.py 스크립트를 사용하여 데이터셋을 준비하려면 다음 명령어를 실행하십시오:
```bash
python prepare_dataset.py --machine <기계 종류> --decibel <데시벨 수준> --base_dir <데이터셋 디렉토리>
```

### 예시:
```bash
python prepare_dataset.py --machine fan --decibel -6_dB --base_dir ./data/
```

### 기능
MIMII 데이터셋에서 지정된 기계의 소리 데이터 분류
데이터셋의 디렉토리 경로와 레이블 추출

### 기여 방법
프로젝트에 기여하고 싶으신 분은 이슈를 생성하거나 풀 리퀘스트를 보내주세요.

### 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다. <br>
(아직 없음)자세한 내용은 LICENSE 파일을 참조하십시오.