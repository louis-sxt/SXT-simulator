import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.fftpack import fft2, fftshift, ifftshift
from skimage.transform import radon, iradon, resize
from skimage.data import shepp_logan_phantom
from skimage.draw import disk
from PIL import Image

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SXT Zone Plate Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for scientific look and responsiveness
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    .stSlider > div > div > div > div { background-color: #007acc; }
    h1, h2, h3 { color: #007acc; }
    /* Tighten margins for plots */
    .element-container { margin-bottom: 0.5rem; }
    div[data-testid="stBlock"] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. PHYSICS HELPER FUNCTIONS ---
def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    # Avoid divide by zero
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile

# --- 3. PHYSICS ENGINE (Cached) ---
@st.cache_data
def run_simulation(phantom_type, custom_file, zone_width_nm, defocus, astig, coma):
    size = 256
    
    # --- A. PHANTOM GENERATION ---
    fov_um = 10.0
    px_size_nm = 39.06 

    if phantom_type == "Custom Upload":
        if custom_file is not None:
            try:
                image = Image.open(custom_file).convert('L')
                img_array = np.array(image)
                # Resize to simulation grid
                img_resized = resize(img_array, (size, size), anti_aliasing=True)
                # Normalize 0-1
                phantom = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min())
            except:
                phantom = np.zeros((size, size))
        else:
            phantom = np.zeros((size, size))
            
    elif "Membrane" in phantom_type:
        # High Mag Mode for Membrane
        fov_um = 2.56 
        px_size_nm = 10.0 
        phantom = np.zeros((size, size))
        
        cy, cx = size // 2, size // 2
        y, x = np.ogrid[:size, :size]
        radius_nm = 800.0
        thick_nm = 20.0
        gap_nm = 100.0
        
        r_px = radius_nm / px_size_nm
        thick_px = thick_nm / px_size_nm 
        gap_px = gap_nm / px_size_nm
        
        dist = np.sqrt((y - cy)**2 + (x - cx)**2)
        
        # Layer 1 & 2
        mask1 = (dist >= r_px) & (dist <= r_px + thick_px)
        mask2 = (dist >= r_px + thick_px + gap_px) & (dist <= r_px + 2*thick_px + gap_px)
        phantom[mask1] = 0.8 
        phantom[mask2] = 0.8
        
        if "30 Beads" in phantom_type:
            bead_diam_nm = 100.0
            bead_r_px = (bead_diam_nm / 2) / px_size_nm
            
            # Distribute 30 beads randomly
            # Seed is handled by session state outside function
            for _ in range(30):
                bx = random.uniform(20, size - 20)
                by = random.uniform(20, size - 20)
                if 0 <= bx < size and 0 <= by < size:
                    rr, cc = disk((int(by), int(bx)), bead_r_px, shape=phantom.shape)
                    phantom[rr, cc] = 1.0

    elif "Bead" in phantom_type:
        phantom = np.zeros((size, size))
        def add_beads(count, diam_nm, intensity):
            radius_px = (diam_nm / 2) / px_size_nm
            radius_px = max(0.5, radius_px) 
            for _ in range(count):
                r = int(random.uniform(20, size-20))
                c = int(random.uniform(20, size-20))
                rr, cc = disk((r, c), radius_px, shape=phantom.shape)
                phantom[rr, cc] = intensity
        add_beads(10, 100, 1.0)
        add_beads(15, 250, 0.8)
        add_beads(10, 500, 0.6)
        
    else: 
        # Standard Shepp-Logan
        phantom = shepp_logan_phantom()
        phantom = resize(phantom, (size, size))

    # --- B. OPTICS (PSF Calculation) ---
    lam = 2.4e-9      
    D = 60e-6         
    dr = zone_width_nm * 1e-9 
    stop_D = 25e-6    
    
    grid_sz = 128
    x = np.linspace(-1, 1, grid_sz)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    
    # Annular Aperture
    stop_ratio = stop_D / D
    pupil_mask = (R <= 1.0) & (R >= stop_ratio)
    
    # Aberrations (Zernike-like)
    defocus_mag = defocus * 8.0 
    astig_mag   = astig * 5.0
    coma_mag    = coma * 5.0
    
    phase = (defocus_mag * R**2) + (astig_mag * R**2 * np.cos(2*Theta)) + (coma_mag * R**3 * np.cos(Theta))
    pupil = pupil_mask * np.exp(1j * phase)
    
    # Compute Raw PSF
    psf_raw = np.abs(fftshift(fft2(ifftshift(pupil))))**2
    
    # Crop central part
    center = grid_sz // 2
    crop = 32
    psf_kernel_raw = psf_raw[center-crop:center+crop, center-crop:center+crop]
    
    # --- SCALE PSF PHYSICS ---
    # Adjust kernel size based on physical resolution vs pixel size
    # Baseline: 25nm ZP at 39nm/px -> scale ~0.6
    scale_physics = (zone_width_nm / 25.0) * (39.06 / px_size_nm)
    
    if scale_physics < 0.5: scale_physics = 0.5
    
    new_dim = int(psf_kernel_raw.shape[0] * scale_physics)
    if new_dim % 2 == 0: new_dim += 1 # Ensure odd dimension for center alignment
    
    psf_kernel = resize(psf_kernel_raw, (new_dim, new_dim), anti_aliasing=True)
    psf_kernel /= np.sum(psf_kernel)
    
    # --- C. IMAGING (Convolution) ---
    blurred_img = fftconvolve(phantom, psf_kernel, mode='same')
    
    # --- D. RECONSTRUCTION (Tomography) ---
    theta = np.linspace(0., 180., 180, endpoint=False)
    sinogram = radon(blurred_img, theta=theta)
    recon = iradon(sinogram, theta=theta, filter_name='ramp')
    
    # --- E. MTF ANALYSIS ---
    otf_2d = np.abs(fftshift(fft2(psf_kernel, shape=(size, size))))
    otf_2d /= otf_2d.max() 
    mtf_curve = radial_profile(otf_2d, (size//2, size//2))
    freq_axis = np.linspace(0, 0.5, len(mtf_curve))
    
    # 10% Cutoff Calculation
    cutoff_indices = np.where(mtf_curve < 0.1)[0]
    if len(cutoff_indices) > 0:
        cutoff_freq_px = freq_axis[cutoff_indices[0]]
        res_nm = (1 / (2 * cutoff_freq_px)) * px_size_nm if cutoff_freq_px > 0 else 999
    else:
        res_nm = px_size_nm
        cutoff_freq_px = 0.5

    # Theoretical limit (Rayleigh approx: 1.22 * dr)
    theoretical_limit = 1.22 * zone_width_nm 
    # The system cannot resolve better than the pixel size (Nyquist)
    base_limit = max(theoretical_limit, px_size_nm)
    
    pct_worse = ((res_nm - base_limit) / base_limit) * 100 if res_nm < 999 else 9999
    if pct_worse < 0: pct_worse = 0

    return {
        'phantom': phantom, 'psf': psf_kernel, 'recon': recon,
        'mtf_y': mtf_curve, 'mtf_x': freq_axis,
        'res_nm': res_nm, 'pct_worse': pct_worse, 'cutoff_freq': cutoff_freq_px
    }

# --- 4. APP LAYOUT ---
st.title("🔬 SXT Zone Plate Simulator")
st.markdown("Use the sidebar to adjust optical parameters. **Plots resize automatically.**")

# Session State for Random Seed Stability
if 'seed' not in st.session_state:
    st.session_state.seed = 42

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("1. Sample Selection")
    phantom_type = st.selectbox(
        "Phantom Type",
        ["Standard (Shepp-Logan)", "Bead Mixture (100/250/500nm)", "Nuclear Membrane (Double Layer)", "Nuclear Membrane + 30 Beads", "Custom Upload"]
    )
    
    custom_file = None
    if phantom_type == "Custom Upload":
        custom_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'tif'])

    if st.button("🎲 Regenerate Random Sample"):
        st.session_state.seed = random.randint(0, 10000)

    st.header("2. Zone Plate Optics")
    zp_text = st.selectbox(
        "Outer Zone Width (Δr)",
        ["25 nm (High Res)", "40 nm (Standard)", "60 nm (Low Res)"],
        index=1
    )
    zone_width_nm = int(zp_text.split(" ")[0])

    st.header("3. Optical Aberrations")
    defocus = st.slider("Defocus (Z-axis)", 0.0, 10.0, 0.0, 0.1)
    astig = st.slider("Astigmatism", 0.0, 10.0, 0.0, 0.1)
    coma = st.slider("Coma", 0.0, 10.0, 0.0, 0.1)

    st.header("4. Display Contrast")
    cont_in = st.slider("Input Contrast", 0.5, 5.0, 1.0, 0.1)
    cont_out = st.slider("Recon Contrast", 0.5, 5.0, 1.0, 0.1)

# --- EXECUTION ---
# Set seed before generation
random.seed(st.session_state.seed)
np.random.seed(st.session_state.seed)

res = run_simulation(
    phantom_type, custom_file, zone_width_nm, 
    defocus, astig, coma
)

# --- PLOTTING ---
def apply_contrast(img, gain):
    return np.clip(img * gain, 0, 1.0)

img_in = apply_contrast(res['phantom'], cont_in)
img_out = apply_contrast(res['recon'], cont_out)

# Create 2x2 Grid
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Figure settings for responsiveness
# Setting figsize determines aspect ratio (Square 5x5)
# Streamlit will stretch this to fit the column width
DEFAULT_FIGSIZE = (5, 5)

# 1. PSF
with col1:
    st.subheader("1. Beam Profile (PSF)")
    fig1, ax1 = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax1.imshow(res['psf'], cmap='inferno')
    ax1.axis('off')
    fig1.patch.set_facecolor('#0e1117') 
    st.pyplot(fig1, use_container_width=True)

# 2. Input
with col2:
    st.subheader("2. Input Sample")
    fig2, ax2 = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax2.imshow(img_in, cmap='gray', vmin=0, vmax=1)
    ax2.axis('off')
    fig2.patch.set_facecolor('#0e1117')
    st.pyplot(fig2, use_container_width=True)

# 3. Reconstruction
with col3:
    st.subheader("3. Final Reconstruction")
    fig3, ax3 = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax3.imshow(img_out, cmap='gray', vmin=0, vmax=1)
    ax3.axis('off')
    fig3.patch.set_facecolor('#0e1117')
    st.pyplot(fig3, use_container_width=True)

# 4. MTF Analysis
with col4:
    st.subheader("4. MTF Analysis")
    fig4, ax4 = plt.subplots(figsize=DEFAULT_FIGSIZE)
    
    # Dark Mode Styles
    fig4.patch.set_facecolor('#0e1117')
    ax4.set_facecolor('#262730')
    ax4.tick_params(colors='white')
    ax4.xaxis.label.set_color('white')
    ax4.yaxis.label.set_color('white')
    for spine in ax4.spines.values():
        spine.set_color('white')

    # Data
    ax4.plot(res['mtf_x'], res['mtf_y'], color='#ffff55', linewidth=2, label='MTF')
    ax4.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Nyquist')
    
    # Resolution Text
    res_nm = res['res_nm']
    pct = res['pct_worse']
    deg_str = f"({pct:.0f}% worse)" if pct >= 1.0 else "(Optimal)"
    col_text = '#55ff55' if res_nm < 50 else '#ffff55' if res_nm < 100 else '#ff5555'
    
    # Fixed Text Position (transAxes keeps it in top-left relative to frame)
    ax4.text(0.05, 0.9, f"Res: {res_nm:.1f} nm\n{deg_str}", 
             transform=ax4.transAxes,
             color=col_text, fontsize=12, weight='bold', 
             bbox=dict(facecolor='black', alpha=0.7, edgecolor=col_text))
             
    ax4.plot(res['cutoff_freq'], 0.1, 'wo')
    
    ax4.set_xlabel("Frequency (cycles/px)")
    ax4.set_ylabel("Contrast")
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, color='#444', linestyle='--')
    
    st.pyplot(fig4, use_container_width=True)
