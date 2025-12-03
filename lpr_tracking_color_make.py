#!/usr/bin/env python3

import gradio as gr
import sys
import os
import json
import tempfile
from pathlib import Path
from datetime import timedelta
import cv2
import numpy as np
import pandas as pd
import re
import math
import shutil

# GStreamer imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

# Attempt to import pyds (DeepStream)
DS_AVAILABLE = False
try:
    # Standard path for pyds bindings often included in DeepStream containers
    sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
    import pyds
    DS_AVAILABLE = True
except ImportError:
    # Fallback to checking local path or python dist-packages
    try:
        import pyds
        DS_AVAILABLE = True
    except ImportError:
        print("⚠️  WARNING: 'pyds' library not found. DeepStream features will not work.")

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "input_videos")
CONFIG_FOLDER = os.path.join(BASE_DIR, "config")

# Ensure folders exist
Path(INPUT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(CONFIG_FOLDER).mkdir(parents=True, exist_ok=True)

SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
# UK Format: 2 letters, 2 numbers, space, 3 letters (e.g., AB19 XYZ)
UK_PLATE_PATTERN = re.compile(r'^[A-Z]{2}[0-9]{2}\s?[A-Z]{3}$')

def get_config_path(filename):
    return os.path.join(CONFIG_FOLDER, filename)

class UKPlateValidator:
    @staticmethod
    def is_valid_uk_plate(plate_text):
        # Basic regex check
        clean = plate_text.strip().upper()
        # Allow slight variations for detection (e.g., missing space)
        clean_nospace = clean.replace(' ', '')
        if len(clean_nospace) != 7:
            return False
        # Create a temp pattern for no-space check
        pattern_nospace = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$')
        return pattern_nospace.match(clean_nospace) is not None
    
    @staticmethod
    def format_uk_plate(plate_text):
        clean = plate_text.replace(' ', '').upper()
        if len(clean) == 7:
            return f"{clean[0:4]} {clean[4:7]}"
        return clean

class VehicleDatabase:
    def __init__(self, excel_path=None):
        self.excel_path = excel_path
        self.database = {}
        if excel_path: 
            self.load_database()
    
    def load_database(self):
        if not self.excel_path or not os.path.exists(self.excel_path): 
            return False
        try:
            df = pd.read_excel(self.excel_path)
            # Normalize column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Look for license plate column
            plate_col = next((c for c in df.columns if 'plate' in c), None)
            
            if not plate_col: 
                return False
                
            for _, row in df.iterrows():
                plate = str(row[plate_col]).strip().upper().replace(' ', '')
                if len(plate) < 2: 
                    continue
                
                # Try to find owner and make columns loosely
                owner_col = next((c for c in df.columns if 'owner' in c or 'name' in c), 'owner_name')
                make_col = next((c for c in df.columns if 'make' in c or 'model' in c), 'car_make')
                
                plate_fmt = UKPlateValidator.format_uk_plate(plate)
                vehicle_info = {
                    'owner': str(row.get(owner_col, 'Unknown')).strip(),
                    'db_make': str(row.get(make_col, 'Unknown')).strip(),
                    'plate_display': plate_fmt
                }
                self.database[plate] = vehicle_info
                # Also store formatted version
                self.database[plate_fmt.replace(' ', '')] = vehicle_info
            
            print(f"✅ Loaded {len(self.database)} entries from database")
            return True
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return False
    
    def get_vehicle_info(self, plate):
        plate_clean = plate.strip().upper().replace(' ', '')
        if plate_clean in self.database: 
            return self.database[plate_clean]
        return None

class LPRDetector:
    def __init__(self, video_path, target_plates=None, output_json="lpr_results.json", 
                 output_folder="detected_vehicles", validate_uk_format=True, 
                 frame_skip=1, debug_mode=False):
        self.video_path = video_path
        self.output_json = output_json
        self.output_folder = output_folder
        self.target_plates = set(target_plates) if target_plates else set()
        self.frame_count = 0
        self.plate_detections = {}
        self.saved_plates = set()
        self.frames_to_save = {}
        self.validate_uk_format = validate_uk_format
        self.frame_skip = frame_skip
        self.debug_mode = debug_mode
        
        # Create output directory
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        
        # Get actual video FPS
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
        else:
            self.video_fps = 30.0
        
        print(f"🎬 Video FPS: {self.video_fps}")

    def frame_to_timestamp(self, frame_number):
        seconds = frame_number / self.video_fps
        return str(timedelta(seconds=int(seconds)))

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            sys.stdout.write("\n✅ End of stream\n")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write(f"⚠️  Warning: {err}: {debug}\n")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write(f"❌ Error: {err}: {debug}\n")
            loop.quit()
        return True

    def extract_vehicle_attributes(self, obj_meta):
        """
        Extracts vehicle TYPE and MAKE from parent object.
        Matched to Config: gie-unique-id=4 (Vehicle Type)
        """
        veh_make = "Unknown"
        veh_type = "Unknown"
        
        # Check if this object has a parent (the vehicle detection)
        # If obj_meta is already the vehicle (PGIE), use it directly.
        # If obj_meta is the plate (SGIE), look at the parent.
        target_obj = obj_meta.parent if obj_meta.parent else obj_meta
        
        # Iterate through the classifiers attached to the Vehicle object
        l_classifier = target_obj.classifier_meta_list
        
        while l_classifier is not None:
            try:
                cls_meta = pyds.NvDsClassifierMeta.cast(l_classifier.data)
                
                # --- MATCHING YOUR CONFIG FILE ---
                # Your config has: gie-unique-id=4
                if cls_meta.unique_component_id == 4:
                    l_label = cls_meta.label_info_list
                    if l_label:
                        p_lbl_info = pyds.NvDsLabelInfo.cast(l_label.data)
                        if p_lbl_info.result_label:
                            # The model returns labels like "Coupe", "Sedan", "SUV"
                            veh_type = p_lbl_info.result_label.strip()

                # --- OPTIONAL: Vehicle Make (Brand) ---
                # If you add a VehicleMakeNet later, usually people set that to ID 5
                elif cls_meta.unique_component_id == 5:
                    l_label = cls_meta.label_info_list
                    if l_label:
                        p_lbl_info = pyds.NvDsLabelInfo.cast(l_label.data)
                        if p_lbl_info.result_label:
                            veh_make = p_lbl_info.result_label.strip()
                            
            except StopIteration:
                break
            
            try:
                l_classifier = l_classifier.next
            except StopIteration:
                break
        
        return veh_make, veh_type

    def save_frame_info(self, frame_number, obj_meta, plate_text, vehicle_info):
        if plate_text in self.frames_to_save: 
            return False
        
        # Use parent (Car) bbox for the image if available, else Plate bbox
        target_meta = obj_meta.parent if obj_meta.parent else obj_meta
        rect = target_meta.rect_params
        
        bbox = {
            'left': int(rect.left),
            'top': int(rect.top),
            'width': int(rect.width),
            'height': int(rect.height)
        }
        
        self.frames_to_save[plate_text] = {
            'frame_number': frame_number,
            'bbox': bbox,
            'detected_make': vehicle_info['make'],
            'detected_type': vehicle_info['type']
        }
        return True

    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer: 
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            self.frame_count += 1
            
            # Optimization: Skip frames if configured
            if self.frame_count % self.frame_skip != 0:
                try: l_frame = l_frame.next
                except StopIteration: break
                continue
            
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Check for LPD Component ID (Usually 2 in TrafficCamNet LPD samples)
                # Adjust '2' if your PGIE/SGIE setup uses different class/component IDs
                if obj_meta.unique_component_id == 2:
                    
                    plate_text = ""
                    
                    # Extract text from LPR classifier (Usually attached to the Plate Object)
                    l_classifier = obj_meta.classifier_meta_list
                    while l_classifier is not None:
                        try:
                            cls_meta = pyds.NvDsClassifierMeta.cast(l_classifier.data)
                            l_label = cls_meta.label_info_list
                            if l_label is not None:
                                label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                                if label_info.result_label:
                                    plate_text = label_info.result_label
                                    break
                        except StopIteration:
                            break
                        try: l_classifier = l_classifier.next
                        except StopIteration: break
                    
                    if plate_text and len(plate_text) > 2:
                        
                        is_valid_uk = UKPlateValidator.is_valid_uk_plate(plate_text)
                        
                        # Filter by UK format
                        if self.validate_uk_format and not is_valid_uk:
                            if self.debug_mode:
                                print(f"DEBUG: Skipping invalid UK plate: {plate_text}")
                            try: l_obj = l_obj.next
                            except StopIteration: break
                            continue
                        
                        # Extract Vehicle Attributes (TYPE and MAKE) from Parent
                        veh_make, veh_type = self.extract_vehicle_attributes(obj_meta)
                        
                        plate_clean = plate_text.strip().upper().replace(' ', '')
                        is_target = (not self.target_plates) or (plate_clean in self.target_plates)
                        timestamp = self.frame_to_timestamp(self.frame_count)
                        
                        if plate_clean not in self.plate_detections:
                            self.plate_detections[plate_clean] = {
                                "first_seen": timestamp,
                                "last_seen": timestamp,
                                "count": 1,
                                "make": veh_make,
                                "type": veh_type,
                                "is_target": is_target
                            }
                            print(f"🎯 NEW DETECTION: {plate_text} | Type: {veh_type} | Make: {veh_make}")
                        else:
                            d = self.plate_detections[plate_clean]
                            d["count"] += 1
                            d["last_seen"] = timestamp
                            if d["make"] == "Unknown" and veh_make != "Unknown": d["make"] = veh_make
                            if d["type"] == "Unknown" and veh_type != "Unknown": d["type"] = veh_type

                        # Save high-confidence frames
                        # Only save if we haven't saved this plate yet
                        if plate_clean not in self.saved_plates:
                            v_info = {'make': veh_make, 'type': veh_type}
                            if self.save_frame_info(self.frame_count, obj_meta, plate_clean, v_info):
                                self.saved_plates.add(plate_clean)

                try: l_obj = l_obj.next
                except StopIteration: break
            try: l_frame = l_frame.next
            except StopIteration: break
        
        return Gst.PadProbeReturn.OK

    def extract_frames(self):
        if not self.frames_to_save: 
            return
        
        print(f"\n📸 Extracting {len(self.frames_to_save)} vehicle images...")
        cap = cv2.VideoCapture(self.video_path)
        sorted_frames = sorted(self.frames_to_save.items(), key=lambda x: x[1]['frame_number'])
        
        for idx, (plate, info) in enumerate(sorted_frames, 1):
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, info['frame_number'] - 1)
            ret, frame = cap.read()
            if ret and frame is not None:
                bbox = info['bbox']
                if bbox:
                    h, w, _ = frame.shape
                    # Add padding
                    pad = 50
                    l = max(0, bbox['left'] - pad)
                    t = max(0, bbox['top'] - pad)
                    r = min(w, bbox['left'] + bbox['width'] + pad)
                    b = min(h, bbox['top'] + bbox['height'] + pad)
                    
                    # Ensure valid crop
                    if l < r and t < b: 
                        crop = frame[t:b, l:r]
                        fname = f"{plate.replace(' ', '')}_{info['frame_number']}.jpg"
                        fpath = os.path.join(self.output_folder, fname)
                        cv2.imwrite(fpath, crop)
                        self.frames_to_save[plate]['image_path'] = fpath
                        print(f"  ✓ Saved {idx}/{len(sorted_frames)}: {fname}")
        
        cap.release()
        print("✅ Frame extraction complete")

    def run(self):
        if not DS_AVAILABLE: 
            print("❌ DeepStream not available!")
            return
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting DeepStream LPR Analysis")
        print(f"{'='*60}")
        
        # Check Config Files
        required_configs = [
            "pgie_trafficcamnet_config.txt",
            "sgie_lpd_DetectNet2_us.txt", # Ensure this path matches your file
            "sgie_lpr_us_config.txt",     # Ensure this path matches your file
            "sgie_vehicle_type_config.txt"
        ]
        
        missing = []
        for c in required_configs:
            if not os.path.exists(get_config_path(c)):
                missing.append(c)
        
        if missing:
            print(f"❌ MISSING CONFIG FILES: {missing}")
            print(f"Please place them in: {CONFIG_FOLDER}")
            return

        Gst.init(None)
        pipeline = Gst.Pipeline()
        
        # Create elements
        source = Gst.ElementFactory.make("filesrc", "file-source")
        h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        
        # Inference Engines
        pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
        sgie1 = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine1") # LPD
        sgie2 = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine2") # LPR
        sgie3 = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine3") # Vehicle Type/Make
        
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        # IMPORTANT: Caps filter to force RGBA for OSD
        caps = Gst.ElementFactory.make("capsfilter", "caps-filter")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        sink = Gst.ElementFactory.make("fakesink", "fake-sink")

        if not all([source, h264parser, decoder, streammux, pgie, sgie1, sgie2, sgie3, nvvidconv, caps, nvosd, sink]):
            print("❌ Failed to create pipeline elements")
            return

        # Add to pipeline
        for elem in [source, h264parser, decoder, streammux, pgie, sgie1, sgie2, sgie3, nvvidconv, caps, nvosd, sink]:
            pipeline.add(elem)

        # Configure Source
        source.set_property('location', self.video_path)
        
        # Configure StreamMux
        streammux.set_property('width', 1920)
        streammux.set_property('height', 1080)
        streammux.set_property('batch-size', 1)
        # 4000000 nanoseconds = 4ms
        streammux.set_property('batched-push-timeout', 4000000)
        
        # Configure Inference
        pgie.set_property('config-file-path', get_config_path("pgie_trafficcamnet_config.txt"))
        sgie1.set_property('config-file-path', get_config_path("sgie_lpd_DetectNet2_us.txt"))
        sgie2.set_property('config-file-path', get_config_path("sgie_lpr_us_config.txt"))
        sgie3.set_property('config-file-path', get_config_path("sgie_vehicle_type_config.txt"))
        
        # Configure Converter
        # 0=Default, 3=Unified(dGPU), 4=NVMap(Jetson). 0 is usually safest.
        nvvidconv.set_property("nvbuf-memory-type", 0) 
        
        # Configure Caps
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        
        # Link
        source.link(h264parser)
        h264parser.link(decoder)
        
        sinkpad = streammux.get_request_pad("sink_0")
        srcpad = decoder.get_static_pad("src")
        srcpad.link(sinkpad)
        
        streammux.link(pgie)
        pgie.link(sgie1)
        sgie1.link(sgie2)
        sgie2.link(sgie3)
        sgie3.link(nvvidconv)
        nvvidconv.link(caps)
        caps.link(nvosd)
        nvosd.link(sink)

        # Bus and Loops
        loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, loop)

        # Add Probe
        osd_sink = nvosd.get_static_pad("sink")
        osd_sink.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

        # Run
        print("▶️  Starting pipeline...")
        pipeline.set_state(Gst.State.PLAYING)
        
        try:
            loop.run()
        except Exception as e:
            print(f"❌ Processing error: {e}")
        finally:
            print("\n🛑 Stopping pipeline...")
            pipeline.set_state(Gst.State.NULL)
        
        # Post-process
        self.extract_frames()
        return

# --- GRADIO UI ---
vehicle_db = None

def load_db(file):
    global vehicle_db
    if file is None: return "❌ No file"
    path = file.name if hasattr(file, 'name') else file
    vehicle_db = VehicleDatabase(path)
    if vehicle_db.database: return f"✅ Loaded {len(vehicle_db.database)} vehicles"
    return "❌ Failed"

def analyze(src_type, upload, folder, targets, debug):
    if not DS_AVAILABLE: 
        return [], "❌ DeepStream library not available."
    
    vid = upload if src_type == "Upload" else os.path.join(INPUT_FOLDER, folder)
    if not vid or not os.path.exists(vid): return [], "❌ Video not found"
    
    t_list = [x.strip().upper().replace(' ', '') for x in targets.split(',') if x.strip()]
    output_dir = tempfile.mkdtemp(prefix="lpr_out_")
    
    det = LPRDetector(vid, t_list, output_folder=output_dir, debug_mode=debug)
    det.run()
    
    res = "# 🚗 Recognition Results\n\n"
    res += f"**Processed Frames:** {det.frame_count} | **Unique Plates:** {len(det.plate_detections)}\n\n---\n\n"
    
    imgs = []
    for plate, info in det.plate_detections.items():
        plate_fmt = UKPlateValidator.format_uk_plate(plate)
        db_str = "No Match"
        if vehicle_db:
            v = vehicle_db.get_vehicle_info(plate)
            if v: db_str = f"**{v['owner']}** ({v['db_make']})"
            
        res += f"### 📋 {plate_fmt}\n"
        res += f"- **Detected:** {info['make']} {info['type']}\n"
        res += f"- **DB Record:** {db_str}\n"
        res += f"- **Seen:** {info['first_seen']} to {info['last_seen']}\n\n"
        
        if plate in det.frames_to_save and 'image_path' in det.frames_to_save[plate]:
            imgs.append((det.frames_to_save[plate]['image_path'], plate_fmt))
            
    return imgs, res

def create_ui():
    # Detect files immediately on launch for the initial state
    initial_files = []
    if os.path.exists(INPUT_FOLDER):
        initial_files = [f for f in os.listdir(INPUT_FOLDER) 
                         if any(f.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS)]
        initial_files.sort()
    
    with gr.Blocks() as app:
        gr.Markdown("""
        # 🚗 DeepStream UK License Plate Recognition System
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                # --- Database Section ---
                with gr.Accordion("📊 Vehicle Database", open=False):
                    db = gr.File(label="Upload Excel Database")
                    btn_db = gr.Button("Load Database", variant="secondary")
                    db_status = gr.Textbox(label="Database Status", interactive=False)
                
                # --- Video Source Section ---
                with gr.Accordion("🎥 Video Source", open=True):
                    src = gr.Radio(
                        ["Upload", "Folder"], 
                        value="Upload", 
                        label="Select Video Source"
                    )
                    
                    # Upload Component
                    u_vid = gr.Video(label="Upload Video File", visible=True)
                    
                    # Folder Components (Grouped)
                    with gr.Column(visible=False) as folder_col:
                        gr.Markdown(f"📂 *Reading from: {INPUT_FOLDER}*")
                        with gr.Row():
                            f_vid = gr.Dropdown(
                                choices=initial_files,
                                value=initial_files[0] if initial_files else None,
                                label="Select Video File",
                                interactive=True, # FORCE INTERACTIVE
                                scale=3
                            )
                            # Add a dedicated refresh button
                            refresh_btn = gr.Button("🔄 Refresh", scale=1)

                # --- Settings Section ---
                with gr.Accordion("🎯 Detection Settings", open=True):
                    tgt = gr.Textbox(
                        label="Target Plates (comma-separated)",
                        placeholder="e.g., AB12CDE, XY34FGH"
                    )
                    debug = gr.Checkbox(label="Enable Debug Mode", value=False)
                
                run = gr.Button("🚀 Run Analysis", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                with gr.Tab("🖼️ Detected Vehicles"):
                    gal = gr.Gallery(label="Vehicle Images")
                with gr.Tab("📋 Detailed Report"):
                    out = gr.Markdown()
        
        # --- LOGIC HANDLERS ---
        
        # 1. Handle Switching between Upload and Folder (Visibility only)
        def toggle_source(source_type):
            if source_type == "Folder":
                return gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=False)

        # 2. Handle Refreshing the File List (Content only)
        def refresh_file_list():
            print(f"🔄 Scanning folder: {INPUT_FOLDER}")
            if not os.path.exists(INPUT_FOLDER):
                print("❌ Folder does not exist")
                return gr.update(choices=[], value=None, label="❌ Folder not found")
            
            files = [f for f in os.listdir(INPUT_FOLDER) 
                     if any(f.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS)]
            files.sort()
            
            print(f"✅ Found {len(files)} videos: {files}")
            
            if not files:
                return gr.update(choices=[], value=None, label="⚠️ No videos found in folder")
            
            # Return new list and select the first one automatically
            return gr.update(choices=files, value=files[0], label="Select Video File", interactive=True)

        # --- EVENT BINDINGS ---
        
        # When radio button changes, toggle visibility
        src.change(toggle_source, inputs=[src], outputs=[u_vid, folder_col])
        
        # When refresh button clicked, update dropdown
        refresh_btn.click(refresh_file_list, inputs=None, outputs=[f_vid])
        
        # Database load
        btn_db.click(load_db, inputs=[db], outputs=[db_status])
        
        # Run analysis
        run.click(
            analyze, 
            inputs=[src, u_vid, f_vid, tgt, debug], 
            outputs=[gal, out]
        )
    
    return app

if __name__ == "__main__":
    if DS_AVAILABLE:
        print("✅ DeepStream Ready")
        app = create_ui()
        app.launch(server_name="0.0.0.0", server_port=7555)
    else:
        print("❌ DeepStream Not Found. Install pyds.")