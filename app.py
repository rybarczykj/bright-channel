import io
import cv2
import numpy as np
from flask import Flask, render_template_string, request, send_file, jsonify
from pathlib import Path
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()
from bright_channel import (
    bright_channel, dark_channel, normalize_bright_channel, erode_bright_channel,
    compute_illumination_invariants, dehaze, to_u8, shadow_segmentation,
    colorize_segments
)

app = Flask(__name__)

CACHE = {}


def load_image(path, max_dim=1200):
    key = f"img:{path}:{max_dim}"
    if key in CACHE:
        return CACHE[key]

    img = cv2.imread(str(path))
    if img is None:
        return None, None

    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    img_float = img.astype(np.float64) / 255.0

    norm_rgb, c1c2c3, log_chrom = compute_illumination_invariants(img_float)
    guides = [
        np.mean(inv, axis=2).astype(np.float32)
        for inv in [norm_rgb, c1c2c3, log_chrom]
    ]

    CACHE[key] = (img, img_float, guides)
    return CACHE[key]


def compute(image_name, img_float, guides, kappa, beta, gf_radius, gf_eps, mode='shadow'):
    cache_key = f"compute:{image_name}:{kappa}:{beta}:{gf_radius}:{gf_eps}:{mode}"
    if cache_key in CACHE:
        return CACHE[cache_key]

    if mode == 'haze':
        bc = dark_channel(img_float, kappa)
    else:
        bc = bright_channel(img_float, kappa)
    bc_norm = normalize_bright_channel(bc, beta)
    bc_ref = erode_bright_channel(bc_norm, kappa)

    if gf_radius < 1:
        mrf = bc_ref
    else:
        bc_f32 = bc_ref.astype(np.float32)
        results = []
        for guide in guides:
            filtered = cv2.ximgproc.guidedFilter(guide, bc_f32, radius=gf_radius, eps=gf_eps)
            results.append(filtered)
        mrf = np.mean(results, axis=0).astype(np.float64)

    CACHE[cache_key] = (bc_ref, mrf)
    return bc_ref, mrf


def get_seg_confidence(image_name, img_float, kappa, beta, mode):
    seg_key = f"seg:{image_name}:{kappa}:{beta}:{mode}"
    if seg_key in CACHE:
        return CACHE[seg_key][0]
    if mode == 'haze':
        dc = dark_channel(img_float, kappa)
        dc_norm = normalize_bright_channel(dc, beta)
        dc_ref = erode_bright_channel(dc_norm, kappa)
        bc_ref = 1.0 - dc_ref
    else:
        bc = bright_channel(img_float, kappa)
        bc_norm = normalize_bright_channel(bc, beta)
        bc_ref = erode_bright_channel(bc_norm, kappa)
    result = shadow_segmentation(img_float, bc_ref, felz_scale=max(kappa * 15, 50))
    CACHE[seg_key] = result
    return result[0]


def encode_png(arr):
    if arr.dtype != np.uint8:
        arr = to_u8(arr)
    _, buf = cv2.imencode('.png', arr)
    return io.BytesIO(buf.tobytes())


COLORMAPS = {
    'inferno': cv2.COLORMAP_INFERNO,
    'viridis': cv2.COLORMAP_VIRIDIS,
    'magma': cv2.COLORMAP_MAGMA,
    'plasma': cv2.COLORMAP_PLASMA,
    'hot': cv2.COLORMAP_HOT,
    'bone': cv2.COLORMAP_BONE,
}

GRAY_RANGES = {
    'grayscale': (0, 255),
    'gray_light': (100, 255),
    'gray_lighter': (150, 255),
    'gray_lightest': (190, 255),
}


def apply_colormap(gray, name='inferno'):
    u8 = to_u8(gray)
    if name in GRAY_RANGES:
        lo, hi = GRAY_RANGES[name]
        return np.clip(lo + (u8.astype(np.float32) / 255.0) * (hi - lo), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, COLORMAPS.get(name, cv2.COLORMAP_INFERNO))


DATA_DIR = Path(__file__).parent / "data"


def list_images():
    exts = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    images = []
    for ext in exts:
        images.extend(DATA_DIR.glob(ext))
    return sorted(set(images), key=lambda p: p.name)


HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Bright Channel Explorer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1a1a1a; color: #e0e0e0; font-family: system-ui, sans-serif; }
  .controls {
    position: fixed; top: 0; left: 0; width: 280px; height: 100vh;
    background: #252525; padding: 16px; overflow-y: auto; z-index: 10;
    border-right: 1px solid #333;
  }
  .controls h2 { font-size: 14px; margin-bottom: 12px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
  .slider-group { margin-bottom: 14px; }
  .slider-group label { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
  .slider-group label span { color: #88f; font-variant-numeric: tabular-nums; }
  input[type=range] { width: 100%; accent-color: #88f; }
  .view-toggle { margin-bottom: 16px; }
  .view-toggle button {
    padding: 6px 10px; margin: 2px; font-size: 12px; cursor: pointer;
    background: #333; color: #ccc; border: 1px solid #444; border-radius: 4px;
  }
  .view-toggle button.active { background: #446; border-color: #88f; color: #fff; }
  #image-select { display: none; }
  .image-list { margin-bottom: 12px; max-height: 180px; overflow-y: auto; }
  .image-list .thumb {
    display: flex; align-items: center; gap: 8px; padding: 4px 6px;
    cursor: pointer; border-radius: 4px; margin-bottom: 2px;
  }
  .image-list .thumb:hover { background: #333; }
  .image-list .thumb.active { background: #446; outline: 1px solid #88f; }
  .image-list .thumb img { width: 48px; height: 32px; object-fit: cover; border-radius: 3px; }
  .image-list .thumb span { font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .canvas-wrap {
    margin-left: 280px; height: 100vh; display: flex; align-items: center;
    justify-content: center; overflow: hidden;
  }
  #output { max-width: 100%; max-height: 100vh; image-rendering: auto; }
  .timing { font-size: 11px; color: #666; margin-top: 8px; }
  .drop-overlay {
    display: none; position: fixed; inset: 0; z-index: 100;
    background: rgba(68, 68, 255, 0.15); border: 3px dashed #88f;
    align-items: center; justify-content: center;
    font-size: 24px; color: #88f; pointer-events: none;
  }
  .drop-overlay.active { display: flex; }
  .action-btn {
    width: 100%; padding: 8px; margin-top: 6px; font-size: 13px; cursor: pointer;
    border-radius: 4px;
  }
  #save-btn { background: #363; color: #cfc; border: 1px solid #4a4; }
  #save-btn:hover { background: #484; }
  #save-preset-btn { background: #336; color: #ccf; border: 1px solid #44a; }
  #save-preset-btn:hover { background: #448; }
  #delete-preset-btn { background: #433; color: #fcc; border: 1px solid #a44; font-size: 11px; padding: 4px; }
  #delete-preset-btn:hover { background: #544; }
  .status-msg { font-size: 11px; color: #6c6; margin-top: 4px; }
  .preset-select {
    width: 100%; padding: 6px; background: #333; color: #ccc;
    border: 1px solid #444; border-radius: 4px; margin-top: 6px; font-size: 12px;
  }
</style>
</head>
<body>

<div class="controls">
  <h2>Image</h2>
  <select id="image-select">
    {% for img in images %}
    <option value="{{ img }}">{{ img }}</option>
    {% endfor %}
  </select>
  <div class="image-list" id="image-list">
    {% for img in images %}
    <div class="thumb{% if loop.first %} active{% endif %}" data-name="{{ img }}">
      <img src="/thumb/{{ img }}">
      <span>{{ img }}</span>
    </div>
    {% endfor %}
  </div>

  <h2>Mode</h2>
  <div class="view-toggle">
    <button class="active" data-mode="shadow">Shadow (bright ch.)</button>
    <button data-mode="haze">Haze (dark ch.)</button>
  </div>

  <h2>Channel</h2>
  <div class="slider-group">
    <label>Patch size (kappa) <span id="v-kappa">15</span></label>
    <input type="range" id="kappa" min="1" max="81" step="2" value="15">
  </div>
  <div class="slider-group">
    <label>Beta (normalize %) <span id="v-beta">0.05</span></label>
    <input type="range" id="beta" min="0.01" max="0.5" step="0.01" value="0.05">
  </div>

  <div id="seg-weight-section" style="margin-bottom: 14px;">
    <label style="font-size: 13px; cursor: pointer;">
      <input type="checkbox" id="seg-weight" style="accent-color: #88f; margin-right: 6px;">
      Weight by seg. confidence
    </label>
  </div>

  <div id="gf-section">
  <h2>MRF / Guided Filter</h2>
  <div class="slider-group">
    <label>Radius <span id="v-gf_radius">40</span></label>
    <input type="range" id="gf_radius" min="0" max="80" step="1" value="40">
  </div>
  <div class="slider-group">
    <label>Epsilon (log) <span id="v-gf_eps_log">0.0010</span></label>
    <input type="range" id="gf_eps_log" min="-4" max="1" step="0.1" value="-3">
  </div>
  <div style="margin-top: 6px;">
    <label style="font-size: 13px; cursor: pointer;">
      <input type="checkbox" id="color-guide" checked style="accent-color: #88f; margin-right: 6px;">
      Color guide (He et al.)
    </label>
  </div>
  </div>

  <h2>Output</h2>
  <div class="slider-group">
    <label>Gamma <span id="v-gamma">1.0</span></label>
    <input type="range" id="gamma" min="0.1" max="5.0" step="0.1" value="1.0">
  </div>

  <div class="view-toggle" id="view-toggle">
    <h2>View</h2>
    <div id="shadow-views">
      <button class="active" data-view="mrf">MRF</button>
      <button data-view="refined">Refined</button>
      <button data-view="shadow_depth">Shadow Depth</button>
      <button data-view="shadow_depth_gray">Depth (gray)</button>
      <button data-view="albedo">Albedo</button>
      <button data-view="seg_confidence">Seg. Confidence</button>
      <button data-view="seg_vis">Segmentation</button>
      <button data-view="seg_shadow">Shadow Mask</button>
      <button data-view="seg_qcand">Good Candidates</button>
      <button data-view="original">Original</button>
    </div>
    <div id="haze-views" style="display:none">
      <button data-view="dehazed">Dehazed</button>
      <button data-view="transmission">Transmission</button>
      <button data-view="depth">Depth</button>
      <button data-view="depth_gray">Depth (gray)</button>
      <button data-view="dark_channel">Dark Channel</button>
      <button data-view="seg_confidence">Seg. Confidence</button>
      <button data-view="seg_vis">Segmentation</button>
      <button data-view="seg_shadow">Haze Mask</button>
      <button data-view="seg_qcand">Good Candidates</button>
      <button data-view="original">Original</button>
    </div>
  </div>

  <div id="colormap-section" style="display:none; margin-top: 8px;">
    <label style="font-size: 12px; color: #aaa;">Colormap</label>
    <select class="preset-select" id="colormap-select" style="margin-top: 4px;">
      <option value="gray_light">Gray (light)</option>
      <option value="viridis">Viridis</option>
      <option value="magma">Magma</option>
      <option value="plasma">Plasma</option>
      <option value="hot">Hot</option>
      <option value="bone">Bone</option>
      <option value="inferno">Inferno</option>
      <option value="grayscale">Grayscale</option>
      <option value="gray_lighter">Gray (lighter)</option>
      <option value="gray_lightest">Gray (lightest)</option>
    </select>
  </div>
  <div id="segstyle-section" style="display:none; margin-top: 8px;">
    <label style="font-size: 12px; color: #aaa;">Segment coloring</label>
    <select class="preset-select" id="segstyle-select" style="margin-top: 4px;">
      <option value="random_tinted">Random + confidence tint</option>
      <option value="mean_color">Mean image color</option>
      <option value="random_plain">Random (no tint)</option>
      <option value="gray_random">Grayscale random</option>
      <option value="gray_weighted">Grayscale weighted</option>
    </select>
  </div>

  <div class="timing" id="timing"></div>

  <h2>Presets</h2>
  <select class="preset-select" id="preset-select">
    <option value="">— select preset —</option>
  </select>
  <button class="action-btn" id="save-preset-btn">Save Preset</button>
  <button id="delete-preset-btn">Delete Selected Preset</button>

  <h2 style="margin-top:16px">Export</h2>
  <button class="action-btn" id="save-btn">Save Output (full res)</button>
  <div class="status-msg" id="save-status"></div>
</div>

<div class="drop-overlay" id="drop-overlay">Drop image here</div>
<div class="canvas-wrap">
  <img id="output">
</div>

<script>
  const sliders = ['kappa', 'beta', 'gf_radius', 'gf_eps_log', 'gamma'];
  const formatters = {
    kappa: v => v,
    beta: v => parseFloat(v).toFixed(2),
    gf_radius: v => v,
    gf_eps_log: v => Math.pow(10, parseFloat(v)).toFixed(4),
    gamma: v => parseFloat(v).toFixed(1),
  };

  let currentView = 'mrf';
  let currentMode = 'shadow';
  let debounceTimer = null;

  const colormapViews = new Set(['seg_confidence', 'seg_qcand', 'shadow_depth', 'depth']);

  function getParams() {
    const p = { view: currentView, mode: currentMode, image: document.getElementById('image-select').value };
    sliders.forEach(s => p[s] = document.getElementById(s).value);
    p['gf_eps'] = Math.pow(10, parseFloat(p['gf_eps_log']));
    if (colormapViews.has(currentView)) p['colormap'] = document.getElementById('colormap-select').value;
    if (currentView === 'seg_vis') p['segstyle'] = document.getElementById('segstyle-select').value;
    if (document.getElementById('seg-weight').checked) p['seg_weight'] = '1';
    p['color_guide'] = document.getElementById('color-guide').checked ? '1' : '0';
    return p;
  }

  function update() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      const params = getParams();
      sliders.forEach(s => {
        document.getElementById('v-' + s).textContent = formatters[s](params[s]);
      });
      const qs = new URLSearchParams(params).toString();
      const t0 = performance.now();
      const img = document.getElementById('output');
      const newImg = new Image();
      newImg.onload = () => {
        img.src = newImg.src;
        document.getElementById('timing').textContent =
          `${(performance.now() - t0).toFixed(0)}ms round-trip`;
      };
      newImg.src = '/render?' + qs + '&t=' + Date.now();
    }, 50);
  }

  sliders.forEach(s => document.getElementById(s).addEventListener('input', update));
  document.getElementById('colormap-select').addEventListener('change', update);
  document.getElementById('segstyle-select').addEventListener('change', update);
  document.getElementById('seg-weight').addEventListener('change', update);
  document.getElementById('color-guide').addEventListener('change', update);
  function bindThumbs() {
    document.querySelectorAll('.thumb').forEach(t => {
      t.addEventListener('click', () => {
        document.querySelectorAll('.thumb').forEach(x => x.classList.remove('active'));
        t.classList.add('active');
        document.getElementById('image-select').value = t.dataset.name;
        update();
      });
    });
  }
  bindThumbs();

  function updateControlVisibility() {
    const isSeg = currentView.startsWith('seg_');
    document.getElementById('gf-section').style.display = isSeg ? 'none' : '';
    document.getElementById('seg-weight-section').style.display = isSeg ? 'none' : '';
    document.getElementById('colormap-section').style.display = colormapViews.has(currentView) ? '' : 'none';
    document.getElementById('segstyle-section').style.display = currentView === 'seg_vis' ? '' : 'none';
  }

  document.querySelectorAll('#view-toggle button[data-view]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#view-toggle button[data-view]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentView = btn.dataset.view;
      updateControlVisibility();
      update();
    });
  });

  document.querySelectorAll('.view-toggle button[data-mode]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.view-toggle button[data-mode]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentMode = btn.dataset.mode;
      const showId = currentMode === 'haze' ? 'haze-views' : 'shadow-views';
      const hideId = currentMode === 'haze' ? 'shadow-views' : 'haze-views';
      document.getElementById(hideId).style.display = 'none';
      document.getElementById(showId).style.display = '';
      const hasView = document.querySelector(`#${showId} button[data-view="${currentView}"]`);
      if (!hasView) currentView = currentMode === 'haze' ? 'dehazed' : 'mrf';
      document.querySelectorAll('#view-toggle button[data-view]').forEach(b => b.classList.remove('active'));
      const activeBtn = document.querySelector(`#${showId} button[data-view="${currentView}"]`);
      if (activeBtn) activeBtn.classList.add('active');
      updateControlVisibility();
      update();
    });
  });

  // Presets (server-backed, saved to presets.json)
  function applyPreset(preset) {
    if (preset.mode) {
      currentMode = preset.mode;
      document.querySelectorAll('button[data-mode]').forEach(b => b.classList.remove('active'));
      const modeBtn = document.querySelector(`button[data-mode="${currentMode}"]`);
      if (modeBtn) modeBtn.classList.add('active');
      document.getElementById('shadow-views').style.display = currentMode === 'shadow' ? '' : 'none';
      document.getElementById('haze-views').style.display = currentMode === 'haze' ? '' : 'none';
    }
    if (preset.view) {
      currentView = preset.view;
      document.querySelectorAll('#view-toggle button[data-view]').forEach(b => b.classList.remove('active'));
      const viewBtn = document.querySelector(`button[data-view="${currentView}"]`);
      if (viewBtn) viewBtn.classList.add('active');
    }
    sliders.forEach(s => {
      if (preset[s] !== undefined) document.getElementById(s).value = preset[s];
    });
    document.getElementById('color-guide').checked = preset.color_guide === '1';
    updateControlVisibility();
    update();
  }

  async function refreshPresetSelect() {
    const res = await fetch('/presets');
    const presets = await res.json();
    const sel = document.getElementById('preset-select');
    const prev = sel.value;
    sel.innerHTML = '<option value="">— select preset —</option>';
    Object.keys(presets).sort().forEach(name => {
      const opt = document.createElement('option');
      opt.value = name; opt.textContent = name;
      sel.appendChild(opt);
    });
    if (prev && presets[prev]) sel.value = prev;
    return presets;
  }

  document.getElementById('preset-select').addEventListener('change', async (e) => {
    if (!e.target.value) return;
    const res = await fetch('/presets');
    const presets = await res.json();
    if (presets[e.target.value]) applyPreset(presets[e.target.value]);
  });

  document.getElementById('save-preset-btn').addEventListener('click', async () => {
    const name = prompt('Preset name:');
    if (!name) return;
    const params = {};
    sliders.forEach(s => params[s] = document.getElementById(s).value);
    params.mode = currentMode;
    params.view = currentView;
    params.color_guide = document.getElementById('color-guide').checked ? '1' : '0';
    await fetch('/presets/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name, params})
    });
    await refreshPresetSelect();
    document.getElementById('preset-select').value = name;
  });

  document.getElementById('delete-preset-btn').addEventListener('click', async () => {
    const sel = document.getElementById('preset-select');
    if (!sel.value) return;
    await fetch('/presets/delete', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: sel.value})
    });
    await refreshPresetSelect();
  });

  refreshPresetSelect();

  // Save output
  document.getElementById('save-btn').addEventListener('click', async () => {
    const params = getParams();
    const qs = new URLSearchParams(params).toString();
    const status = document.getElementById('save-status');
    status.textContent = 'Saving...';
    const res = await fetch('/save?' + qs);
    const data = await res.json();
    status.textContent = data.filename;
    setTimeout(() => status.textContent = '', 3000);
  });

  // Drag and drop
  let dragCounter = 0;
  const overlay = document.getElementById('drop-overlay');

  document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragCounter++;
    overlay.classList.add('active');
  });
  document.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter <= 0) { dragCounter = 0; overlay.classList.remove('active'); }
  });
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', async (e) => {
    e.preventDefault();
    dragCounter = 0;
    overlay.classList.remove('active');
    const files = [...e.dataTransfer.files].filter(f => f.type.startsWith('image/'));
    for (const file of files) {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/upload', { method: 'POST', body: form });
      const data = await res.json();
      if (data.name) {
        const sel = document.getElementById('image-select');
        const opt = document.createElement('option');
        opt.value = data.name; opt.textContent = data.name;
        sel.appendChild(opt);
        sel.value = data.name;

        // Add thumbnail immediately
        const list = document.getElementById('image-list');
        document.querySelectorAll('.thumb').forEach(x => x.classList.remove('active'));
        const div = document.createElement('div');
        div.className = 'thumb active';
        div.dataset.name = data.name;
        div.innerHTML = `<img src="/thumb/${data.name}"><span>${data.name}</span>`;
        list.appendChild(div);
        bindThumbs();
        list.scrollTop = list.scrollHeight;

        update();
      }
    }
  });

  update();

  setInterval(async () => {
    const res = await fetch('/images');
    const imgs = await res.json();
    const sel = document.getElementById('image-select');
    const current = sel.value;
    const existing = new Set([...sel.options].map(o => o.value));
    const incoming = new Set(imgs);
    if (imgs.length !== existing.size || imgs.some(i => !existing.has(i))) {
      sel.innerHTML = '';
      const list = document.getElementById('image-list');
      list.innerHTML = '';
      imgs.forEach(name => {
        const opt = document.createElement('option');
        opt.value = name; opt.textContent = name;
        sel.appendChild(opt);
        const div = document.createElement('div');
        div.className = 'thumb' + (name === current ? ' active' : '');
        div.dataset.name = name;
        div.innerHTML = `<img src="/thumb/${name}"><span>${name}</span>`;
        list.appendChild(div);
      });
      bindThumbs();
      if (incoming.has(current)) sel.value = current;
      else { sel.selectedIndex = sel.options.length - 1; update(); }
    }
  }, 3000);
</script>
</body>
</html>
"""


@app.route('/')
def index():
    images = [p.name for p in list_images()]
    return render_template_string(HTML, images=images)


@app.route('/thumb/<name>')
def thumb(name):
    path = DATA_DIR / name
    img = cv2.imread(str(path))
    if img is None:
        return "Not found", 404
    h, w = img.shape[:2]
    scale = 96 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return send_file(encode_png(small), mimetype='image/png')


PRESETS_FILE = Path(__file__).parent / "presets.json"


def load_presets():
    if PRESETS_FILE.exists():
        import json
        return json.loads(PRESETS_FILE.read_text())
    return {}


def save_presets_file(presets):
    import json
    PRESETS_FILE.write_text(json.dumps(presets, indent=2))


@app.route('/presets')
def get_presets():
    return jsonify(load_presets())


@app.route('/presets/save', methods=['POST'])
def save_preset():
    data = request.get_json()
    name = data.get('name', '')
    params = data.get('params', {})
    if not name:
        return jsonify({'error': 'No name'}), 400
    presets = load_presets()
    presets[name] = params
    save_presets_file(presets)
    return jsonify({'ok': True})


@app.route('/presets/delete', methods=['POST'])
def delete_preset():
    data = request.get_json()
    name = data.get('name', '')
    presets = load_presets()
    if name in presets:
        del presets[name]
        save_presets_file(presets)
    return jsonify({'ok': True})


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f:
        return jsonify({'error': 'No file'}), 400
    name = f.filename
    ext = Path(name).suffix.lower()

    if ext in ('.heic', '.heif'):
        pil_img = Image.open(f.stream)
        pil_img = pil_img.convert('RGB')
        name = Path(name).stem + '.png'
        dest = DATA_DIR / name
        if not dest.exists():
            pil_img.save(str(dest))
    else:
        dest = DATA_DIR / name
        if not dest.exists():
            f.save(str(dest))

    return jsonify({'name': name})


@app.route('/images')
def images_list():
    return jsonify([p.name for p in list_images()])


@app.route('/render')
def render():
    imgs = list_images()
    image_name = request.args.get('image', imgs[0].name if imgs else '')
    image_path = DATA_DIR / image_name
    view = request.args.get('view', 'mrf')

    kappa = int(request.args.get('kappa', 15))
    beta = float(request.args.get('beta', 0.1))
    gamma = float(request.args.get('gamma', 1.0))
    gf_radius = int(request.args.get('gf_radius', 8))
    gf_eps = float(request.args.get('gf_eps', 0.01))
    mode = request.args.get('mode', 'shadow')
    cmap = request.args.get('colormap', 'inferno')
    color_guide = request.args.get('color_guide', '1') == '1'

    data = load_image(image_path)
    if data is None:
        return "Image not found", 404
    img, img_float, guides = data

    if view == 'original':
        buf = encode_png(img)
    elif view in ('dehazed', 'transmission', 'depth', 'depth_gray', 'dark_channel'):
        omega = 1.0 - beta
        gf_r = max(gf_radius, 1)
        dehaze_key = f"dehaze:{image_name}:{kappa}:{omega}:{gf_r}:{gf_eps}:{color_guide}"
        if dehaze_key in CACHE:
            J, t_raw, t_ref, depth, A, dc = CACHE[dehaze_key]
        else:
            J, t_raw, t_ref, depth, A, dc = dehaze(
                img_float, kappa=kappa, omega=omega, t0=0.1,
                gf_radius=gf_r, gf_eps=gf_eps, color_guide=color_guide
            )
            CACHE[dehaze_key] = (J, t_raw, t_ref, depth, A, dc)

        if request.args.get('seg_weight') == '1':
            conf = get_seg_confidence(image_name, img_float, kappa, beta, mode)
            t_ref = t_ref * conf
            depth = depth * conf

        if view == 'dehazed':
            if gamma != 1.0:
                J = np.power(np.clip(J, 0, 1), gamma)
            buf = encode_png(to_u8(J))
        elif view == 'transmission':
            t_vis = t_ref
            if gamma != 1.0:
                t_vis = np.power(np.clip(t_vis, 0, 1), gamma)
            buf = encode_png(to_u8(t_vis))
        elif view == 'depth':
            d = depth
            if gamma != 1.0:
                d = np.power(np.clip(d, 0, 1), gamma)
            buf = encode_png(apply_colormap(d, cmap))
        elif view == 'depth_gray':
            d = depth
            if gamma != 1.0:
                d = np.power(np.clip(d, 0, 1), gamma)
            buf = encode_png(to_u8(d))
        else:
            buf = encode_png(to_u8(dc))
    elif view.startswith('seg_'):
        seg_key = f"seg:{image_name}:{kappa}:{beta}:{mode}"
        if seg_key in CACHE:
            confidence_map, seg_labels, shadow_intensity, q_cand_map = CACHE[seg_key]
        else:
            if mode == 'haze':
                dc = dark_channel(img_float, kappa)
                dc_norm = normalize_bright_channel(dc, beta)
                dc_ref = erode_bright_channel(dc_norm, kappa)
                bc_ref = 1.0 - dc_ref
            else:
                bc = bright_channel(img_float, kappa)
                bc_norm = normalize_bright_channel(bc, beta)
                bc_ref = erode_bright_channel(bc_norm, kappa)
            confidence_map, seg_labels, shadow_intensity, q_cand_map = shadow_segmentation(
                img_float, bc_ref, felz_scale=max(kappa * 15, 50))
            CACHE[seg_key] = (confidence_map, seg_labels, shadow_intensity, q_cand_map)

        if view == 'seg_confidence':
            if gamma != 1.0:
                confidence_map = np.power(np.clip(confidence_map, 0, 1), gamma)
            buf = encode_png(apply_colormap(confidence_map, cmap))
        elif view == 'seg_vis':
            segstyle = request.args.get('segstyle', 'random_tinted')
            buf = encode_png(colorize_segments(img_float, seg_labels, confidence_map, segstyle))
        elif view == 'seg_shadow':
            if gamma != 1.0:
                shadow_intensity = np.power(np.clip(shadow_intensity, 0, 1), gamma)
            buf = encode_png(to_u8(shadow_intensity))
        elif view == 'seg_qcand':
            if gamma != 1.0:
                q_cand_map = np.power(np.clip(q_cand_map, 0, 1), gamma)
            buf = encode_png(apply_colormap(q_cand_map, cmap))
        else:
            buf = encode_png(to_u8(confidence_map))
    else:
        bc_ref, mrf = compute(image_name, img_float, guides, kappa, beta, gf_radius, gf_eps, mode)

        if request.args.get('seg_weight') == '1':
            conf = get_seg_confidence(image_name, img_float, kappa, beta, mode)
            bc_ref = bc_ref * conf
            mrf = mrf * conf

        if view == 'refined':
            out = np.power(np.clip(bc_ref, 0, 1), gamma) if gamma != 1.0 else bc_ref
            buf = encode_png(out)
        elif view in ('shadow_depth', 'shadow_depth_gray'):
            from bright_channel import transmission_to_depth
            d = transmission_to_depth(mrf)
            if gamma != 1.0:
                d = np.power(np.clip(d, 0, 1), gamma)
            if view == 'shadow_depth':
                buf = encode_png(apply_colormap(d, cmap))
            else:
                buf = encode_png(to_u8(d))
        elif view == 'albedo':
            illum = np.maximum(mrf, 0.05)
            albedo = np.clip(img_float / illum[:, :, None], 0, 1)
            if gamma != 1.0:
                albedo = np.power(albedo, gamma)
            buf = encode_png(to_u8(albedo))
        else:
            out = np.power(np.clip(mrf, 0, 1), gamma) if gamma != 1.0 else mrf
            buf = encode_png(out)

    return send_file(buf, mimetype='image/png')


def load_image_full(path):
    key = f"img_full:{path}"
    if key in CACHE:
        return CACHE[key]
    img = cv2.imread(str(path))
    if img is None:
        return None
    img_float = img.astype(np.float64) / 255.0
    norm_rgb, c1c2c3, log_chrom = compute_illumination_invariants(img_float)
    guides = [np.mean(inv, axis=2).astype(np.float32) for inv in [norm_rgb, c1c2c3, log_chrom]]
    CACHE[key] = (img, img_float, guides)
    return CACHE[key]


@app.route('/save')
def save():
    imgs = list_images()
    image_name = request.args.get('image', imgs[0].name if imgs else '')
    image_path = DATA_DIR / image_name
    view = request.args.get('view', 'mrf')
    kappa = int(request.args.get('kappa', 15))
    beta = float(request.args.get('beta', 0.1))
    gamma = float(request.args.get('gamma', 1.0))
    gf_radius = int(request.args.get('gf_radius', 8))
    gf_eps = float(request.args.get('gf_eps', 0.01))
    mode = request.args.get('mode', 'shadow')
    cmap = request.args.get('colormap', 'inferno')

    data = load_image_full(image_path)
    if data is None:
        return jsonify({'error': 'Image not found'}), 404
    img, img_float, guides = data

    stem = Path(image_name).stem
    suffix = f"_{mode}_k{kappa}_b{beta}_g{gamma}_r{gf_radius}_e{gf_eps:.4f}_{view}"

    if view == 'original':
        result = img
    elif view in ('dehazed', 'transmission', 'depth', 'depth_gray', 'dark_channel'):
        omega = 1.0 - beta
        gf_r = max(gf_radius, 1)
        J, t_raw, t_ref, depth, A, dc = dehaze(
            img_float, kappa=kappa, omega=omega, t0=0.1,
            gf_radius=gf_r, gf_eps=gf_eps
        )
        if view == 'dehazed':
            out = J if gamma == 1.0 else np.power(np.clip(J, 0, 1), gamma)
            result = to_u8(out)
        elif view == 'transmission':
            t_vis = t_ref if gamma == 1.0 else np.power(np.clip(t_ref, 0, 1), gamma)
            result = to_u8(t_vis)
        elif view == 'depth':
            d = depth if gamma == 1.0 else np.power(np.clip(depth, 0, 1), gamma)
            result = apply_colormap(d, cmap)
        elif view == 'depth_gray':
            d = depth if gamma == 1.0 else np.power(np.clip(depth, 0, 1), gamma)
            result = to_u8(d)
        else:
            result = to_u8(dc)
    elif view.startswith('seg_'):
        if mode == 'haze':
            dc = dark_channel(img_float, kappa)
            dc_norm = normalize_bright_channel(dc, beta)
            dc_ref = erode_bright_channel(dc_norm, kappa)
            bc_ref_seg = 1.0 - dc_ref
        else:
            bc = bright_channel(img_float, kappa)
            bc_norm = normalize_bright_channel(bc, beta)
            bc_ref_seg = erode_bright_channel(bc_norm, kappa)
        confidence_map, seg_labels, shadow_intensity, q_cand_map = shadow_segmentation(
            img_float, bc_ref_seg, felz_scale=max(kappa * 15, 50))
        if view == 'seg_confidence':
            v = confidence_map if gamma == 1.0 else np.power(np.clip(confidence_map, 0, 1), gamma)
            result = apply_colormap(v, cmap)
        elif view == 'seg_vis':
            segstyle = request.args.get('segstyle', 'random_tinted')
            result = colorize_segments(img_float, seg_labels, confidence_map, segstyle)
        elif view == 'seg_shadow':
            v = shadow_intensity if gamma == 1.0 else np.power(np.clip(shadow_intensity, 0, 1), gamma)
            result = to_u8(v)
        elif view == 'seg_qcand':
            v = q_cand_map if gamma == 1.0 else np.power(np.clip(q_cand_map, 0, 1), gamma)
            result = apply_colormap(v, cmap)
        else:
            result = to_u8(confidence_map)
    else:
        bc_ref, mrf = compute(image_name, img_float, guides, kappa, beta, gf_radius, gf_eps, mode)
        if view == 'refined':
            if gamma != 1.0:
                bc_ref = np.power(np.clip(bc_ref, 0, 1), gamma)
            result = to_u8(bc_ref)
        elif view in ('shadow_depth', 'shadow_depth_gray'):
            from bright_channel import transmission_to_depth
            d = transmission_to_depth(mrf)
            if gamma != 1.0:
                d = np.power(np.clip(d, 0, 1), gamma)
            if view == 'shadow_depth':
                result = apply_colormap(d, cmap)
            else:
                result = to_u8(d)
        elif view == 'albedo':
            illum = np.maximum(mrf, 0.05)
            albedo = img_float / illum[:, :, None]
            albedo = np.clip(albedo, 0, 1)
            if gamma != 1.0:
                albedo = np.power(albedo, gamma)
            result = to_u8(albedo)
        else:
            result = to_u8(mrf)

    out_dir = DATA_DIR / "saved"
    out_dir.mkdir(exist_ok=True)
    filename = f"{stem}{suffix}.png"
    cv2.imwrite(str(out_dir / filename), result)

    return jsonify({'filename': filename, 'path': str(out_dir / filename)})


if __name__ == '__main__':
    print(f"Found {len(list_images())} images in {DATA_DIR}")
    app.run(debug=False, port=5555)
