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

# Custom CSS for scientific look
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    .stSlider > div > div > div > div { background-color: #007acc; }
    h1, h2, h3 { color: #007acc; }
    .plot-container { border: 1px solid #333; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 2. PHYSICS HELPER FUNCTIONS ---
def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile

# --- 3. PHANTOM GENERATOR (Cached & Stable) ---
# We separate this so moving the defocus slider doesn't regenerate the random beads
def generate_phantom(phantom_type, custom_file, seed):
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    size = 256
    phantom = np.zeros((size, size))
    fov_um = 10.0
    px_size_nm = 39.06 

    if phantom_type == "Custom Upload":
        if custom_file is not None:
            try:
                image = Image.open(custom_file).convert('L')
                img_array = np.array(image)
                img_resized = resize(img_array, (size, size), anti_aliasing=True)
                phantom = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min())
            except:
                phantom = np.zeros((size, size))
            
    elif "Membrane" in phantom_type:
        fov_um = 2.56 
        px_size_nm = 10.0 
        
        cy, cx = size // 2, size // 2
        y, x = np.ogrid[:size, :size]
        radius_nm = 800.0
        thick_nm = 20.0
        gap_nm = 100.0
        
        r_px = radius_nm / px_size_nm
        thick_px = thick_nm / px_size_nm 
        gap_px = gap_nm / px_size_nm
        
        dist = np.sqrt((y - cy)**2 + (x - cx)**2)
        mask1 = (dist >= r_px) & (dist <= r_px + thick_px)
        mask2 = (dist >= r_px + thick_px + gap_px) & (dist <= r_px + 2*thick_px + gap_px)
        phantom[mask1] = 0.8 
        phantom[mask2] = 0.8
        
        if "30 Beads" in phantom_type:
            bead_diam_nm = 100.0
            bead_r_px = (bead_diam_nm / 2) / px_size_nm
            for _ in range(30):
                bx = random.uniform(20, size - 20)
                by = random.uniform(20, size - 20)
                if 0 <= bx < size and 0 <= by < size:
                    rr, cc = disk((int(by), int(bx)), bead_r_px, shape=phantom.shape)
                    phantom[rr, cc] = 1.0

    elif "Bead" in phantom_type:
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
        phantom = shepp_logan_phantom()
        phantom = resize(phantom, (size, size))
        
    return phantom, px_size_nm

# --- 4. OPTICS ENGINE ---
def run_optics(phantom, px_size_nm, zone_width_nm, defocus, astig, coma):
    size = 256
    
    # --- B. OPTICS ---
    lam = 2.4e-9      
    D = 60e-6         
    dr = zone_width_nm * 1e-9 
    stop_D = 25e-6    
    
    grid_sz = 128
    x = np.linspace(-1, 1, grid_sz)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    stop_ratio = stop_D / D
    pupil_mask = (R <= 1.0) & (R >= stop_ratio)
    
    # Scale inputs (Fixed the 10x bug here)
    # The slider gives 0-10 directly.
    defocus_mag = defocus * 8.0 
    astig_mag   = astig * 5.0
    coma_mag    = coma * 5.0
    
    phase = (defocus_mag * R**2) + (astig_mag * R**2 * np.cos(2*Theta)) + (coma_mag * R**3 * np.cos(Theta))
    pupil = pupil_mask * np.exp(1j * phase)
    psf_raw = np.abs(fftshift(fft2(ifftshift(pupil))))**2
    
    center = grid_sz // 2
    crop = 32
    psf_kernel = psf_raw[center-crop:center+crop, center-crop:center+crop]
    
    # Scale PSF for high mag
    base_px_size = 39.06
    scale_factor = base_px_size / px_size_nm
    if scale_factor > 1.1: 
        new_dim = int(psf_kernel.shape[0] * scale_factor)
        psf_kernel = resize(psf_kernel, (new_dim, new_dim), anti_aliasing=True)
    
    psf_kernel /= np.sum(psf_kernel)
    
    # --- C. IMAGING ---
    blurred_img = fftconvolve(phantom, psf_kernel, mode='same')
    theta = np.linspace(0., 180., 180, endpoint=False)
    sinogram = radon(blurred_img, theta=theta)
    recon = iradon(sinogram, theta=theta, filter_name='ramp')
    
    # --- D. MTF ---
    otf_2d = np.abs(fftshift(fft2(psf_kernel, shape=(size, size))))
    otf_2d /= otf_2d.max() 
    
    # Radial Profile
    y, x = np.indices((size, size))
    r = np.sqrt((x - size//2)**2 + (y - size//2)**2).astype(int)
    tbin = np.bincount(r.ravel(), otf_2d.ravel())
    nr = np.bincount(r.ravel())
    mtf_curve = tbin / np.maximum(nr, 1)
    
    freq_axis = np.linspace(0, 0.5, len(mtf_curve))
    
    cutoff_indices = np.where(mtf_curve < 0.1)[0]
    if len(cutoff_indices) > 0:
        cutoff_freq_px = freq_axis[cutoff_indices[0]]
        res_nm = (1 / (2 * cutoff_freq_px)) * px_size_nm if cutoff_freq_px > 0 else 999
    else:
        res_nm = px_size_nm
        cutoff_freq_px = 0.5

    # Theoretical limit
    theoretical_limit = 1.22 * zone_width_nm 
    base_limit = max(theoretical_limit, px_size_nm)
    
    pct_worse = ((res_nm - base_limit) / base_limit) * 100 if res_nm < 999 else 9999
    if pct_worse < 0: pct_worse = 0

    return {
        'psf': psf_kernel, 'recon': recon,
        'mtf_y': mtf_curve, 'mtf_x': freq_axis,
        'res_nm': res_nm, 'pct_worse': pct_worse, 'cutoff_freq': cutoff_freq_px
    }

# --- 5. APP LAYOUT ---
st.title("🔬 SXT Zone Plate Simulator")
st.markdown("Use the sidebar to adjust optical parameters and visualize the effect on Soft X-ray Tomography resolution.")

# Initialize Session State for Phantom Stability
if 'seed' not in st.session_state:
    st.session_state.seed = 42

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Sample Selection")
    phantom_type = st.selectbox(
        "Phantom Type",
        ["Standard (Shepp-Logan)", "Bead Mixture (100/250/500nm)", "Nuclear Membrane (Double Layer)", "Nuclear Membrane + 30 Beads", "Custom Upload"]
    )
    
    custom_file = None
    if phantom_type == "Custom Upload":
        custom_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'tif'])

    if st.button("🎲 Regenerate Sample"):
        st.session_state.seed = random.randint(0, 10000)

    st.header("2. Zone Plate Optics")
    zp_text = st.selectbox(
        "Outer Zone Width (Δr)",
        ["25 nm (High Res)", "40 nm (Standard)", "60 nm (Low Res)"],
        index=1
    )
    zone_width_nm = int(zp_text.split(" ")[0])

    st.header("3. Optical Aberrations")
    # Removed / 10.0 in the call, values are 0-10
    defocus = st.slider("Defocus (Z-axis)", 0.0, 10.0, 0.0, 0.1)
    astig = st.slider("Astigmatism", 0.0, 10.0, 0.0, 0.1)
    coma = st.slider("Coma", 0.0, 10.0, 0.0, 0.1)

    st.header("4. Display Contrast")
    cont_in = st.slider("Input Contrast", 0.5, 5.0, 1.0, 0.1)
    cont_out = st.slider("Recon Contrast", 0.5, 5.0, 1.0, 0.1)

# --- EXECUTION ---
# 1. Generate or Retrieve Phantom (Only runs if type/seed/file changes)
phantom, px_size_nm = generate_phantom(phantom_type, custom_file, st.session_state.seed)

# 2. Run Physics
# We pass the slider values DIRECTLY (defocus, astig, coma) without /10.0
# The scaling is handled inside run_optics to match the desktop version.
res = run_optics(phantom, px_size_nm, zone_width_nm, defocus, astig, coma)

# --- PLOTTING ---
def apply_contrast(img, gain):
    return np.clip(img * gain, 0, 1.0)

img_in = apply_contrast(phantom, cont_in)
img_out = apply_contrast(res['recon'], cont_out)

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Row 1
with col1:
    st.subheader("1. Beam Profile (PSF)")
    fig1, ax1 = plt.subplots()
    ax1.imshow(res['psf'], cmap='inferno')
    ax1.axis('off')
    fig1.patch.set_facecolor('#0e1117') 
    st.pyplot(fig1)

with col2:
    st.subheader("2. Input Sample")
    fig2, ax2 = plt.subplots()
    ax2.imshow(img_in, cmap='gray', vmin=0, vmax=1)
    ax2.axis('off')
    fig2.patch.set_facecolor('#0e1117')
    st.pyplot(fig2)

# Row 2
with col3:
    st.subheader("3. Final Reconstruction")
    fig3, ax3 = plt.subplots()
    ax3.imshow(img_out, cmap='gray', vmin=0, vmax=1)
    ax3.axis('off')
    fig3.patch.set_facecolor('#0e1117')
    st.pyplot(fig3)

with col4:
    st.subheader("4. MTF Analysis")
    fig4, ax4 = plt.subplots()
    
    fig4.patch.set_facecolor('#0e1117')
    ax4.set_facecolor('#262730')
    ax4.tick_params(colors='white')
    ax4.xaxis.label.set_color('white')
    ax4.yaxis.label.set_color('white')
    ax4.spines['bottom'].set_color('white')
    ax4.spines['top'].set_color('white') 
    ax4.spines['left'].set_color('white')
    ax4.spines['right'].set_color('white')

    ax4.plot(res['mtf_x'], res['mtf_y'], color='#ffff55', linewidth=2, label='MTF')
    ax4.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Nyquist')
    
    res_nm = res['res_nm']
    pct = res['pct_worse']
    deg_str = f"({pct:.0f}% worse)" if pct >= 1.0 else "(Optimal)"
    col_text = '#55ff55' if res_nm < 50 else '#ffff55' if res_nm < 100 else '#ff5555'
    
    ax4.text(0.25, 0.7, f"Res: {res_nm:.1f} nm\n{deg_str}", 
             color=col_text, fontsize=12, weight='bold', 
             bbox=dict(facecolor='black', alpha=0.7, edgecolor=col_text))
             
    ax4.plot(res['cutoff_freq'], 0.1, 'wo')
    
    ax4.set_xlabel("Frequency (cycles/px)")
    ax4.set_ylabel("Contrast")
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, color='#444', linestyle='--')
    
    st.pyplot(fig4)