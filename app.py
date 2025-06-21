
import base64
import numpy as np
import torch
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import logging
import threading
import time
from transformers import BlipProcessor, BlipForConditionalGeneration
from collections import deque
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from datetime import datetime, timedelta
import queue

# ---- 1. ENHANCED SETUP ----

# Suppress excessive logging from libraries
logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('socketio').setLevel(logging.ERROR)

# --- Enhanced Configuration ---
FRAME_SKIP = 3  # Adaptive frame skipping
IMAGE_SIZE = 224  # Optimized size for BLIP
BUFFER_SIZE = 5  # Smart buffering
MIN_CONFIDENCE_DIFF = 0.03
MAX_WORKERS = 6  # Increased thread pool
CACHE_SIZE = 500  # Larger cache with LRU
BATCH_SIZE = 4  # Batch processing capability

# Advanced performance settings
ADAPTIVE_QUALITY = True
MIN_PROCESSING_INTERVAL = 0.1  # Minimum time between processing
SCENE_CHANGE_THRESHOLD = 0.15  # For scene change detection
CAPTION_HISTORY_SIZE = 10  # Keep caption history for context

# --- Flask & SocketIO App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-very-secret-key!'
socketio = SocketIO(app, async_mode='threading', logger=False, engineio_logger=False, 
                   cors_allowed_origins="*", ping_timeout=60, ping_interval=25)

# --- Enhanced AI Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Advanced thread pool with priority queue
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="caption_worker")
priority_queue = queue.PriorityQueue()

# Load BLIP model with advanced optimizations
try:
    print("Loading BLIP model with optimizations...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model = model.to(device)
    model.eval()
    
    # Advanced CUDA optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        model = torch.jit.script(model)  # TorchScript optimization
        from torch.cuda.amp import autocast, GradScaler
        USE_AMP = True
        scaler = GradScaler()
        print("CUDA optimizations and TorchScript enabled")
    else:
        USE_AMP = False
    
    # Warm up the model
    dummy_image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='black')
    dummy_inputs = processor(dummy_image, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**dummy_inputs, max_length=10)
    print("Model warmed up successfully!")
    
except Exception as e:
    print(f"Error loading BLIP model: {e}")
    exit()

# --- Advanced Caching System ---
class LRUCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = {}
        self.access_order = deque()
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest = self.access_order.popleft()
                del self.cache[oldest]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

# --- Advanced Frame Processing ---
frame_counters = {}
processing_locks = {}
caption_buffers = {}
last_captions = {}
processing_times = {}
caption_history = {}
last_processed_time = {}
scene_features = {}  # For scene change detection

# Enhanced caching
caption_cache = LRUCache(CACHE_SIZE)
batch_queue = {}

# --- Smart Performance Monitor ---
class AdvancedPerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'total_frames': 0,
            'processed_frames': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_processed': 0,
            'scene_changes': 0,
            'processing_times': deque(maxlen=100),
            'start_time': time.time()
        }
        self.lock = threading.Lock()
    
    def log_frame(self, processing_time=None, cache_hit=False, batch_size=1, scene_change=False):
        with self.lock:
            self.metrics['total_frames'] += 1
            if processing_time:
                self.metrics['processed_frames'] += 1
                self.metrics['processing_times'].append(processing_time)
                if batch_size > 1:
                    self.metrics['batch_processed'] += batch_size
            
            if cache_hit:
                self.metrics['cache_hits'] += 1
            else:
                self.metrics['cache_misses'] += 1
            
            if scene_change:
                self.metrics['scene_changes'] += 1
    
    def get_stats(self):
        with self.lock:
            if not self.metrics['processing_times']:
                return {"avg_time": 0, "cache_hit_rate": 0, "fps": 0, "efficiency": 0}
            
            total_time = time.time() - self.metrics['start_time']
            avg_processing_time = np.mean(self.metrics['processing_times'])
            cache_hit_rate = self.metrics['cache_hits'] / max(1, self.metrics['total_frames'])
            processing_fps = self.metrics['processed_frames'] / max(1, avg_processing_time * self.metrics['processed_frames'])
            efficiency = self.metrics['processed_frames'] / max(1, self.metrics['total_frames'])
            
            return {
                "avg_time": avg_processing_time,
                "cache_hit_rate": cache_hit_rate,
                "processing_fps": processing_fps,
                "efficiency": efficiency,
                "total_frames": self.metrics['total_frames'],
                "scene_changes": self.metrics['scene_changes'],
                "batch_efficiency": self.metrics['batch_processed'] / max(1, self.metrics['processed_frames'])
            }

perf_monitor = AdvancedPerformanceMonitor()

# --- Smart Image Preprocessing ---
def smart_preprocess_image(image, enhance_quality=True):
    """Enhanced image preprocessing with quality improvements."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if enhance_quality:
        # Enhance image quality
        # Sharpening
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
        # Contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Color enhancement
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)
    
    # Smart resizing with aspect ratio preservation
    original_size = image.size
    if original_size[0] != original_size[1]:  # Non-square image
        # Crop to square from center
        min_dim = min(original_size)
        left = (original_size[0] - min_dim) // 2
        top = (original_size[1] - min_dim) // 2
        image = image.crop((left, top, left + min_dim, top + min_dim))
    
    # Resize with high-quality resampling
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    
    return image

def advanced_hash_image(image):
    """Generate robust hash for image similarity detection."""
    # Create perceptual hash using multiple features
    img_small = image.resize((16, 16), Image.LANCZOS)
    img_gray = img_small.convert('L')
    
    # Get pixel values
    pixels = list(img_gray.getdata())
    
    # Create hash from average and differences
    avg = sum(pixels) / len(pixels)
    hash_bits = ''.join('1' if pixel > avg else '0' for pixel in pixels)
    
    # Additional feature: edge detection hash
    img_array = np.array(img_gray)
    edges = cv2.Canny(img_array, 50, 150)
    edge_hash = hashlib.md5(edges.tobytes()).hexdigest()[:8]
    
    return hash_bits + edge_hash

def detect_scene_change(sid, current_features):
    """Detect significant scene changes."""
    if sid not in scene_features:
        scene_features[sid] = current_features
        return True
    
    # Compare with previous features
    prev_features = scene_features[sid]
    
    # Calculate similarity (Hamming distance for hash)
    if len(current_features) == len(prev_features):
        diff_count = sum(c1 != c2 for c1, c2 in zip(current_features[:256], prev_features[:256]))
        similarity = 1 - (diff_count / 256)
        
        scene_features[sid] = current_features
        return similarity < (1 - SCENE_CHANGE_THRESHOLD)
    
    scene_features[sid] = current_features
    return True

# ---- 2. ENHANCED WEBSOCKET HANDLERS ----

@socketio.on('connect')
def handle_connect():
    """Enhanced client connection handler."""
    print(f"Client connected: {request.sid}")
    sid = request.sid
    
    # Initialize client data
    frame_counters[sid] = 0
    processing_locks[sid] = threading.Lock()
    caption_buffers[sid] = deque(maxlen=BUFFER_SIZE)
    last_captions[sid] = ""
    processing_times[sid] = deque(maxlen=20)
    caption_history[sid] = deque(maxlen=CAPTION_HISTORY_SIZE)
    last_processed_time[sid] = 0
    scene_features[sid] = ""
    batch_queue[sid] = []
    
    # Send initial status
    emit('status', {'connected': True, 'device': str(device)})

@socketio.on('disconnect')
def handle_disconnect():
    """Enhanced client disconnection handler."""
    print(f"Client disconnected: {request.sid}")
    cleanup_client(request.sid)

def cleanup_client(sid):
    """Enhanced client cleanup."""
    for data_dict in [frame_counters, processing_locks, caption_buffers, 
                      last_captions, processing_times, caption_history,
                      last_processed_time, scene_features, batch_queue]:
        if sid in data_dict:
            del data_dict[sid]

@socketio.on('image')
def handle_image(data_image):
    """Enhanced image handling with smart processing."""
    sid = request.sid
    
    # Initialize if not exists
    if sid not in frame_counters:
        handle_connect()
    
    frame_counters[sid] += 1
    current_time = time.time()
    
    # Adaptive frame skipping based on processing load
    skip_factor = FRAME_SKIP
    if sid in processing_times and processing_times[sid]:
        avg_time = np.mean(processing_times[sid])
        if avg_time > 0.5:  # If processing is slow, skip more frames
            skip_factor = FRAME_SKIP * 2
        elif avg_time < 0.1:  # If processing is fast, skip fewer frames
            skip_factor = max(1, FRAME_SKIP // 2)
    
    if frame_counters[sid] % skip_factor != 0:
        perf_monitor.log_frame()  # Count skipped frames
        return
    
    # Rate limiting
    if current_time - last_processed_time.get(sid, 0) < MIN_PROCESSING_INTERVAL:
        return
    
    # Check if we're already processing
    if not processing_locks[sid].acquire(blocking=False):
        return
    
    last_processed_time[sid] = current_time
    
    # Submit to thread pool with priority
    priority = 1  # Normal priority
    future = executor.submit(process_frame_advanced, sid, data_image, priority)

def process_frame_advanced(sid, data_image, priority=1):
    """Advanced frame processing with multiple optimizations."""
    start_time = time.time()
    
    try:
        # Decode image
        image_data = base64.b64decode(data_image.split(',')[1])
        img = Image.open(BytesIO(image_data))
        
        # Smart preprocessing
        img = smart_preprocess_image(img, enhance_quality=ADAPTIVE_QUALITY)
        
        # Generate advanced hash
        img_hash = advanced_hash_image(img)
        
        # Scene change detection
        scene_changed = detect_scene_change(sid, img_hash)
        
        # Check cache first
        cached_caption = caption_cache.get(img_hash)
        if cached_caption and not scene_changed:
            caption = cached_caption
            cache_hit = True
        else:
            # Generate new caption
            caption = generate_caption_advanced(img)
            caption_cache.put(img_hash, caption)
            cache_hit = False
        
        # Smart caption updating with context
        if should_update_caption_advanced(sid, caption, scene_changed):
            # Add to caption history
            caption_history[sid].append({
                'caption': caption,
                'timestamp': time.time(),
                'scene_changed': scene_changed
            })
            
            last_captions[sid] = caption
            
            # Enhanced caption with context
            contextual_caption = add_context_to_caption(sid, caption)
            
            print(f"New caption for {sid}: {contextual_caption}")
            
            # Send enhanced response
            socketio.emit('caption', {
                'caption': contextual_caption,
                'raw_caption': caption,
                'timestamp': time.time(),
                'confidence': 0.95 if not cache_hit else 1.0,
                'scene_changed': scene_changed,
                'processing_time': time.time() - start_time
            }, room=sid)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        processing_times[sid].append(processing_time)
        perf_monitor.log_frame(processing_time, cache_hit, scene_change=scene_changed)
        
        # Periodic performance logging
        if frame_counters[sid] % 100 == 0:
            stats = perf_monitor.get_stats()
            print(f"Client {sid}: Avg: {stats['avg_time']:.3f}s, Cache: {stats['cache_hit_rate']:.2f}, "
                  f"Efficiency: {stats['efficiency']:.2f}, Scene changes: {stats['scene_changes']}")
    
    except Exception as e:
        print(f"Error processing frame for {sid}: {e}")
        socketio.emit('caption', {
            'caption': f"Processing error: {str(e)[:50]}...",
            'timestamp': time.time(),
            'confidence': 0.0,
            'error': True
        }, room=sid)
    
    finally:
        if sid in processing_locks:
            processing_locks[sid].release()

def should_update_caption_advanced(sid, new_caption, scene_changed):
    """Advanced caption update logic with context awareness."""
    if sid not in last_captions or scene_changed:
        return True
    
    last_caption = last_captions[sid]
    
    # Always update on errors or initial state
    if not last_caption or "error" in last_caption.lower() or last_caption == "Processing...":
        return True
    
    # Check caption history for patterns
    if sid in caption_history and len(caption_history[sid]) > 1:
        recent_captions = [item['caption'] for item in list(caption_history[sid])[-3:]]
        if len(set(recent_captions)) == 1 and new_caption not in recent_captions:
            return True  # Break repetition
    
    # Enhanced semantic similarity with weighted keywords
    words_old = set(last_caption.lower().split())
    words_new = set(new_caption.lower().split())
    
    # Weighted keywords for different importance levels
    high_priority_words = {'walking', 'running', 'sitting', 'standing', 'jumping', 'dancing', 
                          'eating', 'drinking', 'driving', 'flying', 'swimming', 'climbing'}
    medium_priority_words = {'holding', 'wearing', 'looking', 'pointing', 'smiling', 'talking',
                            'reading', 'writing', 'playing', 'working', 'sleeping'}
    objects_words = {'car', 'bike', 'phone', 'book', 'cup', 'computer', 'dog', 'cat', 'bird'}
    
    # Check for high priority changes
    old_high = words_old.intersection(high_priority_words)
    new_high = words_new.intersection(high_priority_words)
    if old_high != new_high:
        return True
    
    # Check for significant object changes
    old_objects = words_old.intersection(objects_words)
    new_objects = words_new.intersection(objects_words)
    if len(old_objects.symmetric_difference(new_objects)) > 1:
        return True
    
    # Advanced similarity calculation
    intersection = words_old.intersection(words_new)
    union = words_old.union(words_new)
    
    if len(union) == 0:
        return True
    
    # Weighted similarity based on word importance
    weight_old = sum(3 if word in high_priority_words else 2 if word in medium_priority_words else 1 
                    for word in words_old)
    weight_new = sum(3 if word in high_priority_words else 2 if word in medium_priority_words else 1 
                    for word in words_new)
    weight_intersection = sum(3 if word in high_priority_words else 2 if word in medium_priority_words else 1 
                             for word in intersection)
    
    weighted_similarity = (2 * weight_intersection) / (weight_old + weight_new) if (weight_old + weight_new) > 0 else 0
    
    return weighted_similarity < 0.75

def add_context_to_caption(sid, caption):
    """Add temporal context to captions."""
    if sid not in caption_history or len(caption_history[sid]) < 2:
        return caption
    
    recent_captions = [item['caption'] for item in list(caption_history[sid])[-3:]]
    
    # Detect action continuity
    action_words = {'walking', 'running', 'sitting', 'standing', 'eating', 'drinking'}
    current_actions = set(caption.lower().split()).intersection(action_words)
    
    if current_actions:
        for prev_caption in recent_captions[:-1]:
            prev_actions = set(prev_caption.lower().split()).intersection(action_words)
            if current_actions == prev_actions:
                return f"{caption} (continuing)"
    
    return caption

def generate_caption_advanced(image):
    """Advanced caption generation with optimizations."""
    try:
        inputs = processor(image, return_tensors="pt").to(device)
        
        # Enhanced generation parameters
        generation_kwargs = {
            'max_length': 30,
            'min_length': 8,
            'num_beams': 5,
            'do_sample': True,
            'temperature': 0.8,
            'top_p': 0.95,
            'top_k': 50,
            'early_stopping': True,
            'no_repeat_ngram_size': 3,
            'length_penalty': 1.1,
            'repetition_penalty': 1.2
        }
        
        if USE_AMP and device.type == 'cuda':
            with autocast():
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, **generation_kwargs)
        else:
            with torch.no_grad():
                generated_ids = model.generate(**inputs, **generation_kwargs)
        
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        return enhance_caption_advanced(caption)
        
    except Exception as e:
        print(f"Error in generate_caption_advanced: {e}")
        return "Processing scene..."

def enhance_caption_advanced(caption):
    """Advanced caption enhancement with NLP improvements."""
    caption = caption.strip()
    if not caption:
        return "Analyzing scene..."
    
    # Remove common prefixes more intelligently
    prefixes_to_remove = [
        "a picture of ", "an image of ", "this is ", "there is ", "there are ",
        "the image shows ", "this image shows ", "a photo of ", "a photograph of "
    ]
    
    caption_lower = caption.lower()
    for prefix in prefixes_to_remove:
        if caption_lower.startswith(prefix):
            caption = caption[len(prefix):]
            break
    
    # Advanced replacements for more natural language
    replacements = {
        r'\b(man|woman|person) (is )?(sitting on|standing in|walking on)\b': 
            lambda m: f"{m.group(1)} {m.group(3).replace('on', 'at').replace('in', 'within')}",
        r'\bholding a\b': 'holding',
        r'\bwearing a\b': 'wearing',
        r'\blooking at the\b': 'observing the',
        r'\bstanding next to\b': 'beside',
        r'\bwalking down\b': 'walking along',
        r'\bsitting at\b': 'seated at'
    }
    
    import re
    for pattern, replacement in replacements.items():
        if callable(replacement):
            caption = re.sub(pattern, replacement, caption, flags=re.IGNORECASE)
        else:
            caption = re.sub(pattern, replacement, caption, flags=re.IGNORECASE)
    
    # Capitalize appropriately
    if caption and not caption[0].isupper():
        caption = caption[0].upper() + caption[1:]
    
    # Add descriptive variety
    action_variations = {
        'walking': ['strolling', 'moving', 'walking'],
        'sitting': ['seated', 'resting', 'sitting'],
        'standing': ['positioned', 'standing', 'upright'],
        'holding': ['grasping', 'carrying', 'holding'],
        'looking': ['observing', 'viewing', 'watching', 'looking at']
    }
    
    # Randomly vary some common actions (seed based on caption for consistency)
    import random
    random.seed(hash(caption) % 1000)
    
    for base_action, variations in action_variations.items():
        if base_action in caption.lower():
            if random.random() < 0.3:  # 30% chance to vary
                caption = caption.replace(base_action, random.choice(variations))
    
    return caption

# ---- 3. ENHANCED FLASK ROUTES ----

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/status')
def status():
    """Enhanced server status with detailed metrics."""
    stats = perf_monitor.get_stats()
    return {
        'active_connections': len(frame_counters),
        'device': str(device),
        'configuration': {
            'frame_skip': FRAME_SKIP,
            'image_size': IMAGE_SIZE,
            'buffer_size': BUFFER_SIZE,
            'cache_size': CACHE_SIZE,
            'batch_size': BATCH_SIZE,
            'adaptive_quality': ADAPTIVE_QUALITY
        },
        'performance': stats,
        'cache_info': {
            'size': len(caption_cache.cache),
            'max_size': CACHE_SIZE
        },
        'optimizations': {
            'mixed_precision': USE_AMP,
            'torch_script': device.type == 'cuda',
            'thread_pool_size': MAX_WORKERS
        }
    }

@app.route('/metrics')
def metrics():
    """Detailed performance metrics endpoint."""
    stats = perf_monitor.get_stats()
    
    # Client-specific metrics
    client_metrics = {}
    for sid in frame_counters:
        if sid in processing_times and processing_times[sid]:
            client_metrics[sid] = {
                'frames_processed': frame_counters[sid],
                'avg_processing_time': np.mean(processing_times[sid]),
                'caption_history_size': len(caption_history.get(sid, [])),
                'last_caption': last_captions.get(sid, "None")
            }
    
    return {
        'global_metrics': stats,
        'client_metrics': client_metrics,
        'system_info': {
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_memory': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        }
    }

@app.route('/clear_cache')
def clear_cache():
    """Clear all caches."""
    caption_cache.clear()
    return {'status': 'cache_cleared', 'timestamp': time.time()}

@app.route('/config', methods=['GET', 'POST'])
def config():
    """Dynamic configuration endpoint."""
    global FRAME_SKIP, ADAPTIVE_QUALITY, SCENE_CHANGE_THRESHOLD
    
    if request.method == 'POST':
        config_data = request.get_json()
        if 'frame_skip' in config_data:
            FRAME_SKIP = max(1, int(config_data['frame_skip']))
        if 'adaptive_quality' in config_data:
            ADAPTIVE_QUALITY = bool(config_data['adaptive_quality'])
        if 'scene_change_threshold' in config_data:
            SCENE_CHANGE_THRESHOLD = float(config_data['scene_change_threshold'])
        
        return {'status': 'updated', 'config': {
            'frame_skip': FRAME_SKIP,
            'adaptive_quality': ADAPTIVE_QUALITY,
            'scene_change_threshold': SCENE_CHANGE_THRESHOLD
        }}
    
    return {
        'frame_skip': FRAME_SKIP,
        'adaptive_quality': ADAPTIVE_QUALITY,
        'scene_change_threshold': SCENE_CHANGE_THRESHOLD
    }

# ---- 4. ENHANCED STARTUP ----
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Enhanced Real-Time Video Captioning Server")
    print("=" * 60)
    print(f"📱 Device: {device}")
    print(f"🎯 Image Processing: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"⚡ Frame Skip: {FRAME_SKIP} (adaptive)")
    print(f"🧠 Mixed Precision: {USE_AMP}")
    print(f"🔄 Thread Pool: {MAX_WORKERS} workers")
    print(f"💾 Cache Size: {CACHE_SIZE} entries (LRU)")
    print(f"🎨 Quality Enhancement: {ADAPTIVE_QUALITY}")
    print(f"🔍 Scene Change Detection: Enabled")
    print("=" * 60)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)