# app.py
import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
from xgboost import XGBClassifier  # ensure import exists for unpickling

# =============== 全局页面设置（需尽早调用） ===============
st.set_page_config(layout="wide")
st.markdown("""
<style>
/* 列宽变化做平滑过渡动画 */
[data-testid="stHorizontalBlock"] [data-testid="column"]{
  transition: width .35s ease-in-out, margin .35s ease-in-out, padding .35s ease-in-out;
}
/* 图片自适应 */
.block-container img { max-width: 100%; height: auto; }
</style>
""", unsafe_allow_html=True)

# ===================== 0) Config =====================
# English feature lists for UI
disease_features_en = [
    "Prothrombin time (PT)", "Eosinophil percentage (Eos%)", "Total bilirubin (TBIL)",
    "Fibrinogen (Fbg)", "Monocyte count (Mono#)", "Monocyte percentage (Mono%)",
    "Uric acid (UA)", "Creatinine (Crea)", "BMI", "Systolic BP", "Gender",
    "Lymphocyte count (Lymph#)"
]
health_disease_features_en = [
    "Albumin (ALB)", "Basophil percentage (Baso%)", "NLR",
    "Neutrophil count (Neut#)", "Gender", "Urea",
    "Basophil count (Baso#)", "Eosinophil percentage (Eos%)"
]

# 计算 Step1 与 Step2 的重合特征，并用于“UI层去重”
overlap_en = sorted(set(disease_features_en) & set(health_disease_features_en))
disease_features_en_ui = [f for f in disease_features_en if f not in overlap_en]

# Chinese feature lists expected by the models (training-time names)
disease_features_cn = [
    "凝血酶原时间(PT)", "嗜酸性粒细胞百分比(Eos%)", "总胆红素(TBIL)",
    "纤维蛋白原(Fbg)", "单核细胞计数(Mono#)", "单核细胞百分比(Mono%)",
    "尿酸(UA)", "肌酐(Crea)", "bmi", "systolic", "gender", "淋巴细胞计数(Lymph#)"
]
health_disease_features_cn = [
    "白蛋白(ALB)", "嗜碱性粒细胞百分比(Baso%)", "nlr",
    "中性粒细胞计数(Neut#)", "gender", "尿素(Urea)",
    "嗜碱性粒细胞计数(Baso#)", "嗜酸性粒细胞百分比(Eos%)"
]

# English -> Chinese mapping (UI -> model)
feature_name_map_en2cn = {
    # health_disease
    "Albumin (ALB)": "白蛋白(ALB)",
    "Basophil percentage (Baso%)": "嗜碱性粒细胞百分比(Baso%)",
    "NLR": "nlr",
    "Neutrophil count (Neut#)": "中性粒细胞计数(Neut#)",
    "Gender": "gender",
    "Urea": "尿素(Urea)",
    "Basophil count (Baso#)": "嗜碱性粒细胞计数(Baso#)",
    "Eosinophil percentage (Eos%)": "嗜酸性粒细胞百分比(Eos%)",
    # disease
    "Prothrombin time (PT)": "凝血酶原时间(PT)",
    "Total bilirubin (TBIL)": "总胆红素(TBIL)",
    "Fibrinogen (Fbg)": "纤维蛋白原(Fbg)",
    "Monocyte count (Mono#)": "单核细胞计数(Mono#)",
    "Monocyte percentage (Mono%)": "单核细胞百分比(Mono%)",
    "Uric acid (UA)": "尿酸(UA)",
    "Creatinine (Crea)": "肌酐(Crea)",
    "BMI": "bmi",
    "Systolic BP": "systolic",
    "Lymphocyte count (Lymph#)": "淋巴细胞计数(Lymph#)"
}
# Chinese -> English (for SHAP labels & CSV中文范围映射)
feature_name_map_cn2en = {v: k for k, v in feature_name_map_en2cn.items()}

# Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
disease_model_path = os.path.join(BASE_DIR, "model", "1_xgb_disease_model.pkl")
health_disease_model_path = os.path.join(BASE_DIR, "model", "1_xgb_health_disease_model.pkl")
feature_range_path = os.path.join(BASE_DIR, "feature_range.csv")

# ===================== 1) Utils =====================
def translate_to_chinese(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename English feature names to Chinese names used during training."""
    rename_dict = {col: mapping[col] for col in df.columns if col in mapping}
    return df.rename(columns=rename_dict)

def get_model_feature_names(xgb_model, fallback_expected: list[str]) -> list[str]:
    """Read feature names stored in the booster; fallback to given expected list."""
    try:
        booster = xgb_model.get_booster()
        names = booster.feature_names
        if names:
            return list(names)
    except Exception:
        pass
    return list(fallback_expected)

def coerce_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def align_for_predict(
    df_cn: pd.DataFrame,
    model,
    expected_cn: list[str],
    fill_value=np.nan,
    ensure_float=True,
    categorical_cols=("gender",)
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Ensure df has exactly the model's expected columns (Chinese names) and order.
    Returns: (X_aligned, missing_cols, extra_cols)
    """
    expected = get_model_feature_names(model, expected_cn)
    missing = [c for c in expected if c not in df_cn.columns]
    extra   = [c for c in df_cn.columns if c not in expected]

    for c in missing:
        df_cn[c] = fill_value

    X = df_cn[expected].copy()

    num_cols = [c for c in expected if c not in categorical_cols]
    X = coerce_numeric(X, num_cols)
    if ensure_float:
        for c in num_cols:
            X[c] = X[c].astype(float)

    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].replace({"Male": 1, "Female": 0, "男": 1, "女": 0})
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, missing, extra

def render_shap_waterfall(model, X_row_cn: pd.Series, cn2en: dict, title: str):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pd.DataFrame([X_row_cn]))
    shap_values.feature_names = [cn2en.get(c, c) for c in shap_values.feature_names]
    st.write(title)
    fig, _ = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

# ---------- Random sample helpers ----------
def load_feature_ranges(csv_path: str) -> dict:
    """
    Load ranges from CSV (supports Chinese feature_name). CSV columns: feature_name, low, high
    Returns: {English UI name: (low, high)}
    """
    ranges = {}
    if not os.path.exists(csv_path):
        return ranges
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return ranges

    needed = {"feature_name", "low", "high"}
    if not needed.issubset(df.columns):
        return ranges

    valid_ui_names = set(health_disease_features_en + disease_features_en)
    for _, row in df.iterrows():
        raw_name = str(row["feature_name"]).strip()
        en_name = feature_name_map_cn2en.get(raw_name)
        if not en_name and raw_name in valid_ui_names:
            en_name = raw_name
        if not en_name:
            continue
        try:
            lo = float(row["low"]); hi = float(row["high"])
            if hi < lo:
                lo, hi = hi, lo
            ranges[en_name] = (lo, hi)
        except Exception:
            continue
    return ranges

def randomize_to_session(fields: list[str], ranges: dict, key_prefix: str):
    """
    写随机值到 session_state[key_prefix + feat]（小部件创建前）：
    - Gender: 随机 0/1
    - 数值: 若有范围则 uniform(low, high)
    """
    for feat in fields:
        key = f"{key_prefix}{feat}"
        if feat == "Gender":
            st.session_state[key] = int(np.random.choice([0, 1]))
        else:
            if feat in ranges:
                lo, hi = ranges[feat]
                val = np.random.uniform(lo, hi)
                st.session_state[key] = float(round(float(val), 3))

def ensure_defaults(fields: list[str], key_prefix: str):
    """为未初始化的控件填默认值（Gender=0，其它=0.0）"""
    for feat in fields:
        key = f"{key_prefix}{feat}"
        if key not in st.session_state:
            st.session_state[key] = 0 if feat == "Gender" else 0.0

# ===================== 2) Load models =====================
health_model = joblib.load(health_disease_model_path)
disease_model = joblib.load(disease_model_path)

# ===================== 3) UI =====================
st.title("Disease Prediction System")
st.markdown("**Step 1:** Health vs Disease → **Step 2:** If Disease, classify Aneurysm vs Dissection")

FEATURE_RANGES = load_feature_ranges(feature_range_path)

# --- 状态位 ---
ss = st.session_state
ss.setdefault("show_results", False)          # 是否采用左右两栏布局
ss.setdefault("_pending_predict", False)      # 下一轮是否需要立即预测并在右栏输出
ss.setdefault("_input_df_en", None)           # 缓存输入DataFrame（英文列）
ss.setdefault("_rand_step1_pending", False)
ss.setdefault("_rand_step2_pending", False)

# --- 布局（无侧边栏）---
if ss["show_results"]:
    left_col, right_col = st.columns([0.58, 0.42])
else:
    _sp1, left_col, _sp2 = st.columns([0.2, 0.6, 0.2])
    right_col = None

# ========== 左栏：仅保留“手动输入” ==========
with left_col:
    ensure_defaults(health_disease_features_en, "step1_")
    ensure_defaults(disease_features_en_ui, "step2_")

    if ss["_rand_step1_pending"]:
        randomize_to_session(health_disease_features_en, FEATURE_RANGES, "step1_")
        ss["_rand_step1_pending"] = False
    if ss["_rand_step2_pending"]:
        randomize_to_session(disease_features_en_ui, FEATURE_RANGES, "step2_")
        ss["_rand_step2_pending"] = False

    st.header("Step 1: Enter features for Health vs Disease")
    cols = st.columns(2)
    for i, feat in enumerate(health_disease_features_en):
        key = f"step1_{feat}"
        with cols[i % 2]:
            if feat == "Gender":
                st.selectbox(
                    "Gender",
                    [0, 1],
                    index=int(ss[key]) if ss[key] in [0, 1] else 0,
                    format_func=lambda x: "Female" if x == 0 else "Male",
                    key=key
                )
            else:
                st.number_input(feat, value=float(ss[key]), key=key)

    if st.button("🎲 Randomize Step 1 ", key="rand_step1_btn"):
        ss["_rand_step1_pending"] = True
        st.rerun()

    with st.expander("Step 2 optional inputs (used only if Step 1 predicts Disease)", expanded=False):
        cols2 = st.columns(2)
        for i, feat in enumerate(disease_features_en_ui):
            key = f"step2_{feat}"
            with cols2[i % 2]:
                if feat == "Gender":
                    st.selectbox(
                        f"{feat}", [0, 1],
                        index=int(ss[key]) if ss[key] in [0, 1] else 0,
                        format_func=lambda x: "Female" if x == 0 else "Male",
                        key=key
                    )
                else:
                    st.number_input(feat, value=float(ss[key]), key=key)

        if st.button("🎲 Randomize Step 2 ", key="rand_step2_btn"):
            ss["_rand_step2_pending"] = True
            st.rerun()

    # 组装英文输入（缓存到 session_state，便于 rerun 之后右栏读取）
    step1_values = {f: ss[f"step1_{f}"] for f in health_disease_features_en}
    step2_values = {f: ss[f"step2_{f}"] for f in disease_features_en_ui}  # 仅非重合特征
    # 合并后，重合特征来自 Step 1，非重合来自 Step 2
    input_df_en = pd.DataFrame([{**step1_values, **step2_values}])

    # ---- 运行按钮：缓存输入 -> 切换布局 -> 下一轮预测 ----
    run_now = st.button("Run Prediction", type="primary", use_container_width=True)
    if run_now:
        ss["_input_df_en"] = input_df_en
        ss["show_results"] = True
        ss["_pending_predict"] = True
        st.rerun()

# ========== 右栏：在下一轮渲染一开始就并排显示并预测 ==========
if ss["show_results"] and right_col is not None:
    with right_col:
        if ss["_pending_predict"]:
            input_df_en = ss.get("_input_df_en", None)
            if input_df_en is None:
                st.warning("No input found. Please enter features.")
            else:
                # ----- Step 1 -----
                X1_en = (
                    input_df_en[health_disease_features_en]
                    if set(health_disease_features_en).issubset(input_df_en.columns)
                    else input_df_en.copy()
                )
                X1_cn = translate_to_chinese(X1_en, feature_name_map_en2cn)
                if "gender" in X1_cn.columns:
                    X1_cn["gender"] = X1_cn["gender"].replace({"Male": 1, "Female": 0, "男": 1, "女": 0})
                X1, miss1, extra1 = align_for_predict(
                    X1_cn, health_model, health_disease_features_cn, categorical_cols=("gender",)
                )

                if miss1:
                    st.warning(f"Step 1: Missing features auto-filled: {miss1}")
                if extra1:
                    st.info(f"Step 1: Ignored extra columns: {extra1}")

                pred1_prob = health_model.predict_proba(X1)[0]
                pred1 = int(np.argmax(pred1_prob))
                pred1_label = "Healthy" if pred1 == 0 else "Disease"

                st.subheader(f"Step 1 Result: {pred1_label}")
                st.write(
                    f"Prediction probabilities → **Healthy: {pred1_prob[0]:.3f}**, "
                    f"**Disease: {pred1_prob[1]:.3f}**"
                )
                try:
                    render_shap_waterfall(
                        health_model, X1.iloc[0, :], feature_name_map_cn2en,
                        "Step 1 Feature Contributions (SHAP Waterfall)"
                    )
                except Exception as e:
                    st.error(f"SHAP rendering failed for Step 1: {e}")

                # ----- Step 2（仅 Step 1 为 Disease 时） -----
                if pred1 == 1:
                    need_en = disease_features_en  # 模型仍需完整特征；重合值来自 Step 1
                    X2_en = input_df_en[need_en] if set(need_en).issubset(input_df_en.columns) else input_df_en.copy()
                    X2_cn = translate_to_chinese(X2_en, feature_name_map_en2cn)
                    if "gender" in X2_cn.columns:
                        X2_cn["gender"] = X2_cn["gender"].replace({"Male": 1, "Female": 0, "男": 1, "女": 0})
                    X2, miss2, extra2 = align_for_predict(
                        X2_cn, disease_model, disease_features_cn, categorical_cols=("gender",)
                    )
                    if miss2:
                        st.warning(f"Step 2: Missing features auto-filled: {miss2}")
                    if extra2:
                        st.info(f"Step 2: Ignored extra columns: {extra2}")

                    pred2_prob = disease_model.predict_proba(X2)[0]
                    pred2 = int(np.argmax(pred2_prob))
                    pred2_label = "Aneurysm" if pred2 == 0 else "Dissection"

                    st.subheader(f"Step 2 Result: {pred2_label}")
                    st.write(
                        f"Prediction probabilities → **Aneurysm: {pred2_prob[0]:.3f}**, "
                        f"**Dissection: {pred2_prob[1]:.3f}**"
                    )
                    try:
                        render_shap_waterfall(
                            disease_model, X2.iloc[0, :], feature_name_map_cn2en,
                            "Step 2 Feature Contributions (SHAP Waterfall)"
                        )
                    except Exception as e:
                        st.error(f"SHAP rendering failed for Step 2: {e}")

            # 本轮预测完成，清理 pending 标志（保留 show_results 与缓存，便于二次查看）
            ss["_pending_predict"] = False
