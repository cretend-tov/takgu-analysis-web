# 탁구 3D 분석 웹앱

Streamlit으로 만든 탁구 영상 분석 웹 애플리케이션입니다.

- 공 궤적 검출
- Top-down 뷰
- 3D 궤적 시각화
- 스마트폰 브라우저 지원

## 로컬 실행 방법

```bash
pip install -r requirements.txt
streamlit run 탁구_3D_분석_웹앱.py
```

## 무료 클라우드 배포 방법 (Streamlit Community Cloud)

1. GitHub에 새 저장소를 만듭니다 (Public).
2. 아래 두 파일을 업로드합니다:
   - `탁구_3D_분석_웹앱.py`
   - `requirements.txt`
3. https://share.streamlit.io 에 접속하여 배포합니다.
4. 배포 완료 후 생성된 URL을 스마트폰에서 접속하세요.

## 사용 시 권장 사항

- 분석할 영상은 **10~15초 이내**가 가장 좋습니다.
- 스마트폰에서는 **가로 모드**로 사용하는 것을 추천합니다.
- 처음 배포 후 2~5분 정도 기다려주세요.

## 라이선스

개인 및 교육용으로 자유롭게 사용 가능합니다.