# lfa_streamlit_ui.py
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import math
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="LFIA App", layout="wide")

# =========================================================
# Utility Functions
# =========================================================
def read_image_bytes(uploaded_file):
    data = uploaded_file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, flags=cv2.IMREAD_COLOR)

def auto_crop_strip(img, edge_pct=0.12):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    col_sum, row_sum = mag.sum(axis=0), mag.sum(axis=1)
    h, w = gray.shape
    col_thresh, row_thresh = max(1,int(col_sum.max()*edge_pct)), max(1,int(row_sum.max()*edge_pct))
    cols, rows = np.where(col_sum>col_thresh)[0], np.where(row_sum>row_thresh)[0]
    if cols.size==0 or rows.size==0:
        cw,ch=int(w*0.8), int(h*0.5)
        x0,y0=max(0,(w-cw)//2), max(0,(h-ch)//2)
        return img[y0:y0+ch, x0:x0+cw].copy()
    x0,x1, y0,y1=max(0,cols[0]-8), min(w-1,cols[-1]+8), max(0,rows[0]-8), min(h-1,rows[-1]+8)
    return img[y0:y1+1, x0:x1+1].copy()

def normalize_orientation(img):
    h,w=img.shape[:2]
    return img if w>=h else cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def compute_profile(img, channel='red', band=(0.25,0.75), smooth_k=5):
    h,w=img.shape[:2]
    top,bottom=int(h*band[0]), int(h*band[1])
    band_img = img[top:bottom,:,:]
    ch = band_img[:,:,2].astype(np.float32) if channel=='red' else cv2.cvtColor(band_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    prof = ch.mean(axis=0)
    if smooth_k>1: prof=np.convolve(prof,np.ones(smooth_k)/smooth_k,mode='same')
    return prof

def detect_lines(profile,min_distance=8):
    inv=-profile
    prom=max((inv.max()-inv.min())*0.05, np.std(inv)*0.5,1e-3)
    peaks,_=find_peaks(inv,distance=min_distance,prominence=prom)
    return np.sort(peaks)

def sample_mean_at(img,x,width=12,band=(0.25,0.75)):
    h,w=img.shape[:2]
    top,bottom=int(h*band[0]), int(h*band[1])
    half=max(1,width//2)
    lx,rx=max(0,int(x-half)), min(w,int(x+half))
    roi=img[top:bottom,lx:rx]
    if roi.size==0: return 0.0
    return float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32).mean())

def estimate_background(profile):
    n=len(profile)
    return float((profile[:max(1,n//8)].mean()+profile[-max(1,n//8):].mean())/2.0)

def log_linear(points):
    pts=np.array(points,dtype=float)
    conc,signal=pts[:,0],pts[:,1]
    y=np.log(conc)
    X=np.vstack([signal,np.ones_like(signal)]).T
    slope,intercept=np.linalg.lstsq(X,y,rcond=None)[0]
    return float(intercept), float(slope)

def predict_log_linear(a,b,signal):
    return float(math.exp(a+b*signal))

def fourpl(x,A,B,C,D):
    return A+(D-A)/(1.0+(x/C)**B)

def fit_4pl(points):
    pts=np.array(points,dtype=float)
    conc,signal=pts[:,0],pts[:,1]
    p0=[min(conc)*0.8,1.0,np.median(signal),max(conc)*1.2]
    popt,_ = curve_fit(fourpl, signal, conc, p0=p0, maxfev=20000)
    return popt

def predict_4pl(popt,signal):
    return float(fourpl(signal,*popt))

# =========================================================
# Dummy Competitive Calibration Data (automatic)
# =========================================================
dummy_calib = {
    "Tetrodotoxin (TTX)": [
        (0.1, 0.92), (0.5, 0.85), (1, 0.78), (5, 0.61), (10, 0.49), (25, 0.32)
    ],
    "Domoic Acid (DA)": [
        (5, 0.93), (10, 0.86), (25, 0.74), (50, 0.61), (100, 0.50), (250, 0.33)
    ],
    "Okadaic Acid (OA)": [
        (1, 0.91), (2, 0.84), (5, 0.73), (10, 0.62), (20, 0.51), (50, 0.30)
    ]
}

EU_LIMITS = {
    "Tetrodotoxin (TTX)": 44,
    "Domoic Acid (DA)": 20000,
    "Okadaic Acid (OA)": 160
}

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    mode = st.selectbox("Mode", ["Single analyte"])
    toxin = st.selectbox("Select Toxin", ["Tetrodotoxin (TTX)", "Domoic Acid (DA)", "Okadaic Acid (OA)"])
    calib_method = st.selectbox("Calibration method", ["log-linear", "4PL"])
    channel = st.selectbox("Profile channel", ["red","gray"])
    smooth_k = st.slider("Profile smoothing", 1, 21, 5, step=2)

# =========================================================
# Upload only ONCE
# =========================================================
uploaded_img = st.file_uploader("Upload strip image", type=["jpg","jpeg","png"])

if uploaded_img:
    img = read_image_bytes(uploaded_img)
    norm = normalize_orientation(auto_crop_strip(img))
    profile = compute_profile(norm, channel, band=(0.25,0.75), smooth_k=smooth_k)
    peaks = detect_lines(profile)
    bg = estimate_background(profile)

    st.image(cv2.cvtColor(norm, cv2.COLOR_BGR2RGB), caption="Processed LFIA Strip")

    # ------------------ Profile Graph ------------------
    fig1, ax1 = plt.subplots()
    ax1.plot(profile)
    ax1.set_title("Intensity Profile")
    ax1.set_xlabel("Pixel position")
    ax1.set_ylabel("Intensity")
    st.pyplot(fig1)

    st.write("**Detected lines:**", peaks.tolist())
    st.write("**Background noise:**", bg)

    # ------------------ Extract Control & Test ------------------
    if len(peaks) >= 2:
        ctrl, test = int(peaks[0]), int(peaks[1])
        ctrl_corr = max(0, sample_mean_at(norm, ctrl) - bg)
        test_corr = max(0, sample_mean_at(norm, test) - bg)

        T_C = test_corr / (ctrl_corr if ctrl_corr != 0 else 1e-6)

        st.subheader("Line Intensities")
        dfres = pd.DataFrame({
            "Line": ["Control", "Test"],
            "Raw": [sample_mean_at(norm, ctrl), sample_mean_at(norm, test)],
            "Corrected": [ctrl_corr, test_corr]
        })
        st.dataframe(dfres)

        st.write("### **T/C Ratio:**", T_C)

        # ------------------ Calibration Curve ------------------
        cal = dummy_calib[toxin]
        concs = [c for c,_ in cal]
        sigs = [s for _,s in cal]

        fig2, ax2 = plt.subplots()
        ax2.scatter(concs, sigs)
        ax2.set_xscale("log")
        ax2.set_xlabel("Concentration (ng/mL)")
        ax2.set_ylabel("T/C Ratio")
        ax2.set_title("Calibration Curve")
        st.pyplot(fig2)

        # ------------------ Predict Concentration ------------------
        if calib_method == "log-linear":
            a,b = log_linear(cal)
            conc_pred = predict_log_linear(a,b,T_C)
        else:
            popt = fit_4pl(cal)
            conc_pred = predict_4pl(popt,T_C)

        st.subheader("Predicted Concentration")
        st.write(f"### **{conc_pred:.2f} ng/mL**")

        # ------------------ Positive / Negative / Invalid ------------------
        EU = EU_LIMITS[toxin]
        if ctrl_corr < 5:
            st.markdown("<h3 style='color:red;'>INVALID: Control line too weak</h3>", unsafe_allow_html=True)
        elif conc_pred > EU:
            st.markdown("<h3 style='color:red;'>POSITIVE (Above EU limit)</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:red;'>NEGATIVE (Below EU limit)</h3>", unsafe_allow_html=True)

        # ------------------ Export Results ------------------
        result_str = f"""
Toxin: {toxin}
T/C Ratio: {T_C}
Predicted Concentration: {conc_pred} ng/mL
EU Limit: {EU} ng/mL
"""

        st.download_button(
            "Download Results",
            data=result_str,
            file_name="LFIA_results.txt"
        )
