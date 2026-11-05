# app.py — 체형별 공정성을 고려한 사이즈 추천 시연
import os, json, joblib
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="체형별 사이즈 추천 시연", layout="wide")

# app.py가 있는 폴더를 기준으로 파일을 찾음 (Colab, Streamlit Cloud 어디서든 동작)
HERE = os.path.dirname(os.path.abspath(__file__))
def path(name): return os.path.join(HERE, name)

@st.cache_resource
def load():
    model = joblib.load(path("model_v2.pkl"))
    params = json.load(open(path("params.json"), encoding="utf-8"))
    items = pd.read_csv(path("item_stats.csv"))
    return model, params, items

model, P, items = load()
ORDER = ['hourglass', 'athletic', 'pear', 'petite', 'full bust', 'straight & narrow', 'apple']
KO = ['작음', '맞음', '큼']
CUP = {'aa': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'd+': 4.5, 'dd': 5, 'ddd/e': 6, 'f': 7, 'g': 8, 'h': 9, 'i': 10, 'j': 11}

# 옷 성향 3단계: 리뷰 50개 이상인 옷들의 실제 분포에서 대표값을 뽑음
pop = items[items['item_n'] >= 50]
ITEM_TYPE = {
    '작게 나오는 옷': dict(item_small=pop['item_small'].quantile(0.9), item_large=pop['item_large'].quantile(0.1)),
    '보통':          dict(item_small=pop['item_small'].median(),     item_large=pop['item_large'].median()),
    '크게 나오는 옷': dict(item_small=pop['item_small'].quantile(0.1), item_large=pop['item_large'].quantile(0.9)),
}
ITEM_N = float(pop['item_n'].median())

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)

def predict(df):
    # 모델 원점수 → 원래 비율 되돌리기 → temperature scaling → 보정 확률
    X = df[P['features']].copy()
    for c in ['body_type', 'category']:
        X[c] = pd.Categorical(X[c], categories=P['categories'][c])
    z = model.predict(X, raw_score=True)
    return softmax((z + np.array(P['log_prior'])) / P['T'])

def candidate_sets(q, bt):
    common = [KO[k] for k in range(3) if 1 - q[k] <= P['t_all']]
    fair = [KO[k] for k in range(3) if 1 - q[k] <= P['t_bk'][f"{bt}_{KO[k]}"]]
    return common, fair

tab1, tab2 = st.tabs(["사이즈 추천 체험", "연구 결과 대시보드"])

# ── 탭 1: 사이즈 추천 체험 ──
with tab1:
    st.header("내 체형으로 사이즈 추천 받아보기")
    st.caption("두 기준 모두 '정답이 후보 안에 90% 들어간다'를 목표로 한다.")

    c1, c2, c3 = st.columns(3)
    with c1:
        height = st.slider("키 (cm)", 145, 190, 165)
        weight = st.slider("몸무게 (kg)", 40, 110, 60)
        age = st.slider("나이", 18, 70, 30)
    with c2:
        body = st.selectbox("체형", ORDER, index=ORDER.index('pear'))
        band = st.selectbox("가슴 둘레", [30, 32, 34, 36, 38, 40], index=2)
        cup = st.selectbox("컵", list(CUP), index=list(CUP).index('c'))
    with c3:
        category = st.selectbox("옷 종류", [c for c in P['categories']['category'] if c != 'other'])
        size = st.slider("주문 사이즈", 0, 30, 8)
        item_type = st.radio("이 옷은?", list(ITEM_TYPE), index=1, horizontal=True)
        size_diff = st.slider("다른 사람들이 빌린 평균 사이즈보다", -8, 8, 0,
                              help="0이면 이 옷을 빌린 사람들의 평균 사이즈로 주문한 것")

    def row(bt):
        return dict(height_cm=height, weight_kg=weight, bust_band=band, cup_num=CUP[cup],
                    body_type=bt, category=category, size=size, age=age,
                    item_n=ITEM_N, size_diff=size_diff, **ITEM_TYPE[item_type])

    q = predict(pd.DataFrame([row(body)]))[0]
    common, fair = candidate_sets(q, body)

    left, right = st.columns(2)
    with left:
        st.subheader("AI가 예상한 확률")
        fig = go.Figure(go.Bar(x=KO, y=q, text=[f"{v*100:.0f}%" for v in q], textposition='outside'))
        fig.update_layout(yaxis=dict(range=[0, 1], title='확률'), height=320, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("추천 후보")
        st.metric("공통 기준", " 또는 ".join(common) if common else "-")
        st.metric("체형×결과별 기준", " 또는 ".join(fair) if fair else "-")
        st.caption("후보가 여러 개면 AI가 이 체형에 대해 확신하기 어렵다는 뜻이다.")

    st.subheader("같은 몸 크기에서 체형만 바꾸면?")
    qs = predict(pd.DataFrame([row(bt) for bt in ORDER]))
    table = []
    for bt, qq in zip(ORDER, qs):
        cm, fr = candidate_sets(qq, bt)
        table.append({'체형': bt, '작음': f"{qq[0]*100:.0f}%", '맞음': f"{qq[1]*100:.0f}%", '큼': f"{qq[2]*100:.0f}%",
                      '공통 기준 후보': " / ".join(cm), '체형×결과별 기준 후보': " / ".join(fr)})
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)

# ── 탭 2: 연구 결과 대시보드 ──
with tab2:
    st.header("체형별 연구 결과")
    pick = st.multiselect("비교할 체형", ORDER, default=ORDER)

    r = pd.read_csv(path("1_체형별_재현율.csv"), index_col=0).loc[pick]
    st.subheader("작은 옷·큰 옷 맞힌 비율 (95% 오차 범위)")
    fig = go.Figure()
    for col, lo, hi, name in [('작음_재현율', '작음_하한', '작음_상한', '작은 옷'),
                              ('큼_재현율', '큼_하한', '큼_상한', '큰 옷')]:
        fig.add_bar(x=r.index, y=r[col], name=name,
                    error_y=dict(type='data', symmetric=False, array=r[hi] - r[col], arrayminus=r[col] - r[lo]))
    fig.update_layout(barmode='group', yaxis=dict(range=[0, 1], title='재현율'), height=380)
    st.plotly_chart(fig, use_container_width=True)

    cv = pd.read_csv(path("6_conformal_포함률_후보수.csv"), index_col=0).loc[pick]
    kind = st.radio("결과", ['작음', '큼'], horizontal=True,
                    format_func=lambda k: "실제로 작았던 옷" if k == '작음' else "실제로 컸던 옷")
    a, b = st.columns(2)
    with a:
        st.subheader("정답 포함률")
        fig = go.Figure()
        fig.add_bar(x=cv.index, y=cv[f'공통_{kind}'], name='공통 기준')
        fig.add_bar(x=cv.index, y=cv[f'체형결과별_{kind}'], name='체형×결과별 기준')
        fig.add_hline(y=0.9, line_dash='dash', annotation_text='목표 90%')
        fig.update_layout(barmode='group', yaxis=dict(range=[0, 1]), height=360)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.subheader("평균 후보 수")
        fig = go.Figure()
        fig.add_bar(x=cv.index, y=cv['공통_후보수'], name='공통 기준')
        fig.add_bar(x=cv.index, y=cv['체형결과별_후보수'], name='체형×결과별 기준')
        fig.update_layout(barmode='group', yaxis=dict(range=[0, 3]), height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("작은 옷 리뷰에서 언급된 부위")
    parts = pd.read_csv(path("8_리뷰_부위분석.csv"), index_col=[0, 1]).loc['small'].loc[pick]
    fig = go.Figure(go.Heatmap(z=parts.values, x=parts.columns, y=parts.index, colorscale='Blues',
                               text=(parts.values * 100).round(0), texttemplate="%{text}%"))
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)
