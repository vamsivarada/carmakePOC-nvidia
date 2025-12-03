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

# GStreamer imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

# Attempt to import pyds (DeepStream)
try:
    sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
    import pyds
    DS_AVAILABLE = True
except ImportError:
    print("⚠️  WARNING: 'pyds' library not found. DeepStream features will not work.")
    DS_AVAILABLE = False

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "input_videos")
EXCEL_DATABASE = os.path.join(BASE_DIR, "vehicle_database.xlsx")
CONFIG_FOLDER = os.path.join(BASE_DIR, "config")

SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
UK_PLATE_PATTERN = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$')

def get_config_path(filename):
    """Helper to find config files in local folder or absolute path."""
    local_path = os.path.join(CONFIG_FOLDER, filename)
    if os.path.exists(local_path):
        return local_path
    # Fallback to hardcoded workspace path if local not found
    fallback = f"/workspaces/cartheft/python_lpr/config/{filename}"
    return fallback if os.path.exists(fallback) else None

class UKPlateValidator:
    @staticmethod
    def is_valid_uk_plate(plate_text):
        clean = plate_text.replace(' ', '').upper().strip()
        if len(clean) != 7:
            return False
        return UK_PLATE_PATTERN.match(clean) is not None
    
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
        self.load_database()
    
    def load_database(self):
        if not self.excel_path or not os.path.exists(self.excel_path):
            return False
        
        try:
            df = pd.read_excel(self.excel_path)
            # Normalize headers
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Check for essential columns (relaxed matching)
            if 'license_plate' not in df.columns:
                print("❌ Database missing 'license_plate' column")
                return False
            
            valid_count = 0
            for _, row in df.iterrows():
                plate = str(row['license_plate']).strip().upper().replace(' ', '')
                
                if not UKPlateValidator.is_valid_uk_plate(plate):
                    continue
                
                valid_count += 1
                plate_with_space = UKPlateValidator.format_uk_plate(plate)
                
                vehicle_info = {
                    'make': str(row.get('car_make', 'Unknown')).strip(),
                    'type': str(row.get('car_type', 'Unknown')).strip(),
                    'location': str(row.get('location', 'Unknown')).strip(),
                    'plate_display': plate_with_space
                }
                
                self.database[plate] = vehicle_info
                self.database[plate_with_space] = vehicle_info
            
            print(f"✅ Loaded {valid_count} valid UK plates")
            return True
            
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return False
    
    def get_vehicle_info(self, plate):
        plate_clean = plate.strip().upper().replace(' ', '')
        if plate_clean in self.database:
            return self.database[plate_clean]
        
        plate_with_space = UKPlateValidator.format_uk_plate(plate_clean)
        if plate_with_space in self.database:
            return self.database[plate_with_space]
        
        return None

class LicensePlateCorrector:
    def __init__(self, region='UK'):
        self.region = region
        
    def correct_uk_format(self, plate_text):
        clean = plate_text.replace(' ', '').upper()
        if len(clean) != 7:
            return clean
        
        corrected = list(clean)
        
        # UK Format: 2 letters, 2 numbers, 3 letters (e.g., AB12 CDE)
        # Positions 0,1: Letters
        for i in [0, 1]:
            if corrected[i].isdigit():
                if corrected[i] == '0': corrected[i] = 'O'
                elif corrected[i] == '1': corrected[i] = 'I'
                elif corrected[i] == '4': corrected[i] = 'A'
                elif corrected[i] == '8': corrected[i] = 'B'
        
        # Positions 2,3: Numbers
        for i in [2, 3]:
            if corrected[i].isalpha():
                if corrected[i] == 'O': corrected[i] = '0'
                elif corrected[i] == 'I': corrected[i] = '1'
                elif corrected[i] == 'S': corrected[i] = '5'
                elif corrected[i] == 'Z': corrected[i] = '2'
                elif corrected[i] == 'B': corrected[i] = '8'
                elif corrected[i] == 'G': corrected[i] = '6'
        
        # Positions 4,5,6: Letters
        for i in [4, 5, 6]:
            if corrected[i].isdigit():
                if corrected[i] == '0': corrected[i] = 'O'
                elif corrected[i] == '1': corrected[i] = 'I'
                elif corrected[i] == '8': corrected[i] = 'B'
        
        return ''.join(corrected)
    
    def process(self, plate_text):
        original = plate_text.strip()
        plate_text = self.correct_uk_format(plate_text)
        plate_text = UKPlateValidator.format_uk_plate(plate_text)
        return plate_text, plate_text != original.replace(' ', '')

class LPRDetector:
    def __init__(self, video_path, target_plates=None, output_json="lpr_results.json", 
                 output_folder="detected_vehicles", enable_corrections=True, 
                 validate_uk_format=True, frame_skip=2):
        self.video_path = video_path
        self.output_json = output_json
        self.output_folder = output_folder
        self.target_plates = set(target_plates) if target_plates else set()
        self.results = {"license_plates": []}
        self.frame_count = 0
        self.processed_frames = 0
        self.plate_detections = {}
        self.saved_plates = set()
        self.frames_to_save = {}
        self.enable_corrections = enable_corrections
        self.validate_uk_format = validate_uk_format
        self.frame_skip = frame_skip
        self.corrector = LicensePlateCorrector(region='UK') if enable_corrections else None
        self.validation_stats = {"total": 0, "valid": 0, "invalid": 0}
        self.video_fps = self.get_video_fps()
        
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
    
    def get_video_fps(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps if fps > 0 else 30.0
        except:
            return 30.0
    
    def frame_to_timestamp(self, frame_number):
        if self.video_fps <= 0: return "00:00:00"
        seconds = frame_number / self.video_fps
        td = timedelta(seconds=seconds)
        # Format explicitly to handle milliseconds better
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
        
    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("\n✅ End of Stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"❌ Error: {err}")
            loop.quit()
        return True

    def decodebin_child_added(self, child_proxy, Object, name, user_data):
        if name.find("decodebin") != -1:
            Object.connect("child-added", self.decodebin_child_added, user_data)

    def create_source_bin(self, index, uri):
        bin_name = f"source-bin-{index:02d}"
        nbin = Gst.Bin.new(bin_name)
        if not nbin: return None
        
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        if not uri_decode_bin: return None
        
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", self.cb_newpad, nbin)
        uri_decode_bin.connect("child-added", self.decodebin_child_added, nbin)
        
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad: return None
        return nbin

    def cb_newpad(self, decodebin, decoder_src_pad, data):
        caps = decoder_src_pad.get_current_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        source_bin = data
        if gstname.find("video") != -1:
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                print("Failed to link decoder src pad")

    def save_frame_info(self, frame_number, obj_meta, plate_text, vehicle_type):
        try:
            # Only save the best shot (first valid detection usually)
            if plate_text not in self.frames_to_save:
                bbox = None
                if obj_meta.parent:
                    vehicle_meta = obj_meta.parent
                    left = int(vehicle_meta.rect_params.left)
                    top = int(vehicle_meta.rect_params.top)
                    width = int(vehicle_meta.rect_params.width)
                    height = int(vehicle_meta.rect_params.height)
                    
                    if width > 0 and height > 0:
                        bbox = {'left': left, 'top': top, 'width': width, 'height': height}
                
                self.frames_to_save[plate_text] = {
                    'frame_number': frame_number,
                    'vehicle_type': vehicle_type,
                    'bbox': bbox
                }
                return True
            return False
        except Exception:
            return False

    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer: return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            self.frame_count += 1
            
            # Optimization: Skip frames logic
            if self.frame_count % self.frame_skip != 0:
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
                continue
            
            self.processed_frames += 1
            
            # Map Vehicle Object IDs to Types (Car, Truck, etc.)
            vehicles = {}
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                # PGIE class_id 1 is usually vehicle in standard configs, check your config!
                # Assuming TrafficCamNet: 0=Car, 2=TwoWheeler. Adjust based on your model.
                if obj_meta.unique_component_id == 1:
                    vehicles[obj_meta.object_id] = {
                        "type": obj_meta.obj_label,
                        "id": obj_meta.object_id
                    }
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            # Process LPR Results
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # SGIE (LPR) unique ID is usually 2 or 3 depending on pipeline
                if obj_meta.unique_component_id == 2:
                    parent_id = obj_meta.parent.object_id if obj_meta.parent else None
                    vehicle_type = vehicles.get(parent_id, {}).get("type", "Unknown") if parent_id else "Unknown"
                    
                    l_classifier = obj_meta.classifier_meta_list
                    while l_classifier is not None:
                        try:
                            classifier_meta = pyds.NvDsClassifierMeta.cast(l_classifier.data)
                        except StopIteration:
                            break
                        
                        l_label = classifier_meta.label_info_list
                        while l_label is not None:
                            try:
                                label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                                try:
                                    plate_text = label_info.result_label
                                except UnicodeDecodeError:
                                    plate_text = ""
                                
                                if plate_text and plate_text.strip():
                                    plate_raw = plate_text.strip()
                                    plate = plate_raw
                                    was_corrected = False
                                    
                                    # 1. OCR Correction
                                    if self.enable_corrections and self.corrector:
                                        plate, was_corrected = self.corrector.process(plate_raw)
                                    
                                    # 2. Validation
                                    self.validation_stats["total"] += 1
                                    is_valid_uk = UKPlateValidator.is_valid_uk_plate(plate)
                                    
                                    if is_valid_uk:
                                        self.validation_stats["valid"] += 1
                                    else:
                                        self.validation_stats["invalid"] += 1
                                        if self.validate_uk_format:
                                            # Skip invalid plates if validation is enforced
                                            l_label = l_label.next
                                            continue
                                    
                                    # 3. Target Filtering
                                    is_target = (not self.target_plates) or (plate in self.target_plates)
                                    
                                    # 4. Timeline Tracking
                                    timestamp = self.frame_to_timestamp(self.frame_count)
                                    
                                    if plate not in self.plate_detections:
                                        self.plate_detections[plate] = {
                                            "first_seen_frame": self.frame_count,
                                            "first_seen_timestamp": timestamp,
                                            "last_seen_frame": self.frame_count,
                                            "last_seen_timestamp": timestamp,
                                            "vehicle_type": vehicle_type,
                                            "confidence": label_info.result_prob,
                                            "count": 1,
                                            "is_target": is_target,
                                            "is_valid_uk": is_valid_uk,
                                            "all_timestamps": [timestamp]
                                        }
                                        if is_target:
                                            print(f"🎯 FOUND: {plate} | {vehicle_type} @ {timestamp}")
                                    else:
                                        # Update existing entry
                                        d = self.plate_detections[plate]
                                        d["count"] += 1
                                        d["last_seen_frame"] = self.frame_count
                                        d["last_seen_timestamp"] = timestamp
                                        d["all_timestamps"].append(timestamp)
                                        # Update confidence if higher
                                        if label_info.result_prob > d["confidence"]:
                                            d["confidence"] = label_info.result_prob
                                    
                                    # 5. Save Frame Request
                                    if is_valid_uk and is_target and plate not in self.saved_plates:
                                        saved = self.save_frame_info(self.frame_count, obj_meta, plate, vehicle_type)
                                        if saved:
                                            self.saved_plates.add(plate)
                                    
                            except StopIteration:
                                break
                            try:
                                l_label = l_label.next
                            except StopIteration:
                                break
                        try:
                            l_classifier = l_classifier.next
                        except StopIteration:
                            break
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def extract_frames(self):
        if not self.frames_to_save:
            return
        
        print(f"\n📸 Extracting {len(self.frames_to_save)} images from source...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("❌ Failed to open video for extraction")
            return
        
        # Sort by frame number to minimize seeking
        frames_sorted = sorted(self.frames_to_save.items(), key=lambda x: x[1]['frame_number'])
        
        extracted = 0
        for plate, info in frames_sorted:
            frame_num = info['frame_number']
            bbox = info['bbox']
            
            # Note: CAP_PROP_POS_FRAMES is 0-indexed
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            vehicle_img = frame
            if bbox:
                # Add padding
                h, w, _ = frame.shape
                pad = 30
                left = max(0, bbox['left'] - pad)
                top = max(0, bbox['top'] - pad)
                right = min(w, bbox['left'] + bbox['width'] + pad)
                bottom = min(h, bbox['top'] + bbox['height'] + pad)
                
                if left < right and top < bottom:
                    vehicle_img = frame[top:bottom, left:right]
            
            safe_plate = "".join(c if c.isalnum() else "_" for c in plate)
            filename = f"{safe_plate}_{frame_num}.jpg"
            filepath = os.path.join(self.output_folder, filename)
            
            if cv2.imwrite(filepath, vehicle_img):
                extracted += 1
                self.frames_to_save[plate]['image_path'] = filepath
        
        cap.release()
        print(f"✅ Extracted {extracted} images")

    def save_results(self):
        # 1. Extract images using OpenCV
        self.extract_frames()
        
        # 2. Filter valid plates for reporting
        valid_detections = {k: v for k, v in self.plate_detections.items() if v.get('is_valid_uk', True)}
        
        summary = {
            "total_frames": self.frame_count,
            "processed_frames": self.processed_frames,
            "video_fps": self.video_fps,
            "plates": []
        }
        
        for plate, info in valid_detections.items():
            duration_frames = info["last_seen_frame"] - info["first_seen_frame"]
            duration_seconds = duration_frames / self.video_fps if self.video_fps else 0
            
            plate_entry = {
                "plate": plate,
                "vehicle_type": info["vehicle_type"],
                "first_seen": info["first_seen_timestamp"],
                "last_seen": info["last_seen_timestamp"],
                "detection_count": info["count"],
                "duration_sec": round(duration_seconds, 2),
                "timestamps": info["all_timestamps"],
                "is_target": info.get("is_target", False)
            }
            
            if plate in self.frames_to_save and 'image_path' in self.frames_to_save[plate]:
                plate_entry["image_path"] = self.frames_to_save[plate]['image_path']
            
            summary["plates"].append(plate_entry)
        
        # Save JSON
        with open(self.output_json, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.final_summary = summary
        print(f"✅ Saved results to {self.output_json}")

    def run(self):
        if not DS_AVAILABLE:
            print("❌ Cannot run pipeline: DeepStream not installed.")
            return

        Gst.init(None)
        pipeline = Gst.Pipeline()
        
        # Create elements
        uri = f"file://{os.path.abspath(self.video_path)}"
        source_bin = self.create_source_bin(0, uri)
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
        sgie1 = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine1")
        sgie2 = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine2")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        sink = Gst.ElementFactory.make("fakesink", "fake-sink")

        if not all([source_bin, streammux, pgie, sgie1, sgie2, nvvidconv, nvosd, sink]):
            print("❌ Failed to create GStreamer elements")
            return

        # Add to pipeline
        pipeline.add(source_bin)
        pipeline.add(streammux)
        pipeline.add(pgie)
        pipeline.add(sgie1)
        pipeline.add(sgie2)
        pipeline.add(nvvidconv)
        pipeline.add(nvosd)
        pipeline.add(sink)

        # Configure Elements
        streammux.set_property('width', 1920)
        streammux.set_property('height', 1080)
        streammux.set_property('batch-size', 1)
        nvvidconv.set_property("nvbuf-memory-type", 3)
        
        # Check config files
        pgie_config = get_config_path("pgie_trafficcamnet_config.txt")
        sgie1_config = get_config_path("sgie_lpd_DetectNet2_us.txt")
        sgie2_config = get_config_path("sgie_lpr_us_config.txt")
        
        if not all([pgie_config, sgie1_config, sgie2_config]):
            print("❌ Missing configuration files in ./config/ or /workspaces/...")
            return

        pgie.set_property('config-file-path', pgie_config)
        sgie1.set_property('config-file-path', sgie1_config)
        sgie2.set_property('config-file-path', sgie2_config)
        
        sink.set_property('sync', 0) # Faster processing, disable for real-time visualization

        # Linking
        sinkpad = streammux.get_request_pad("sink_0")
        srcpad = source_bin.get_static_pad("src")
        srcpad.link(sinkpad)
        
        streammux.link(pgie)
        pgie.link(sgie1)
        sgie1.link(sgie2)
        sgie2.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(sink)

        # Signal handling
        loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, loop)

        # Add Probe
        osdsinkpad = nvosd.get_static_pad("sink")
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

        print(f"🚀 Running Pipeline on: {self.video_path}")
        pipeline.set_state(Gst.State.PLAYING)
        
        try:
            loop.run()
        except Exception as e:
            print(f"❌ Pipeline Error: {e}")
        finally:
            pipeline.set_state(Gst.State.NULL)
            self.save_results()

# --- GLOBAL VARS ---
vehicle_db = None

def get_videos_from_folder():
    if not os.path.exists(INPUT_FOLDER): return []
    return sorted([f for f in os.listdir(INPUT_FOLDER) 
                   if any(f.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS)])

def load_vehicle_database(excel_file=None):
    global vehicle_db
    db_path = excel_file.name if excel_file else EXCEL_DATABASE
    vehicle_db = VehicleDatabase(db_path)
    
    if vehicle_db.database:
        valid_plates = len(set([k.replace(' ', '') for k in vehicle_db.database.keys()])) // 2
        return f"✅ Loaded {valid_plates} plates from DB"
    return "⚠️ Failed to load database"

def process_video_ui(video_source, uploaded_video, folder_video, target_plates_text, 
                     enable_corrections, validate_uk, frame_skip, progress=gr.Progress()):
    
    if not DS_AVAILABLE:
        return [], "## ❌ Error\nDeepStream libraries (`pyds`) are missing. This app requires the NVIDIA DeepStream Docker container."

    # Determine Input Path
    video_path = None
    if video_source == "Upload Video":
        if uploaded_video is None: return [], "❌ Upload a video first"
        video_path = uploaded_video
    else:
        if not folder_video: return [], "❌ Select a video first"
        video_path = os.path.join(INPUT_FOLDER, folder_video)
        if not os.path.exists(video_path): return [], f"❌ Video not found at {video_path}"

    progress(0, desc="Initializing...")
    
    # Create temp workspace
    temp_dir = tempfile.mkdtemp(prefix="lpr_session_")
    output_dir = os.path.join(temp_dir, "images")
    output_json = os.path.join(temp_dir, "results.json")
    
    # Parse Targets
    target_plates = []
    if target_plates_text.strip():
        raw_targets = [p.strip() for p in target_plates_text.replace('\n', ',').split(',') if p.strip()]
        if enable_corrections:
            corrector = LicensePlateCorrector()
            target_plates = [corrector.process(p)[0] for p in raw_targets]
        else:
            target_plates = raw_targets

    # Initialize Detector
    detector = LPRDetector(
        video_path=video_path,
        target_plates=target_plates,
        output_json=output_json,
        output_folder=output_dir,
        enable_corrections=enable_corrections,
        validate_uk_format=validate_uk,
        frame_skip=frame_skip
    )
    
    progress(0.2, desc="Running Analysis (this may take time)...")
    
    # Run Pipeline (Redirect stdout to capture logs if needed, omitted here for simplicity)
    detector.run()
    
    progress(0.9, desc="Generating Report...")
    
    # Read Results
    images = []
    if os.path.exists(output_dir):
        images = [str(p) for p in sorted(Path(output_dir).glob("*.jpg"))]
    
    if not hasattr(detector, 'final_summary'):
        return [], "❌ Analysis failed or produced no results."
        
    summary = detector.final_summary
    
    # Build Markdown Report
    status_msg = f"### 📊 Analysis Results\n"
    status_msg += f"**Processed:** {summary.get('processed_frames', 0)} frames | **FPS Used:** {summary.get('video_fps', 30):.2f}\n\n"
    
    if not summary.get("plates"):
        status_msg += "#### ⚠️ No vehicles detected."
    else:
        for p in summary["plates"]:
            plate = p['plate']
            
            # DB Lookup
            db_info = "Unknown"
            if vehicle_db:
                v = vehicle_db.get_vehicle_info(plate)
                if v: db_info = f"✅ {v['make']} {v['type']} ({v['location']})"
                else: db_info = "❌ Not in DB"
            
            status_msg += f"---\n"
            status_msg += f"#### 🚗 Plate: `{plate}`\n"
            status_msg += f"- **Database:** {db_info}\n"
            status_msg += f"- **Type:** {p.get('vehicle_type', 'N/A')}\n"
            status_msg += f"- **Seen:** {p.get('first_seen')} ➔ {p.get('last_seen')} (Duration: {p.get('duration_sec')}s)\n"
            status_msg += f"- **Detections:** {p.get('detection_count')}\n"
            
            # Timeline Visualization
            timestamps = p.get('timestamps', [])
            if len(timestamps) > 10:
                status_msg += f"- **Timeline:** {timestamps[0]}, {timestamps[1]} ... {timestamps[-2]}, {timestamps[-1]}\n"
            else:
                status_msg += f"- **Timeline:** {', '.join(timestamps)}\n"
            status_msg += "\n"

    return images, status_msg

# --- UI SETUP ---
def create_ui():
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER, exist_ok=True)
    
    with gr.Blocks(title="DeepStream LPR") as demo:
        gr.Markdown("# 🇬🇧 DeepStream UK LPR & Vehicle Tracking")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Database Section
                gr.Markdown("### 1. Database")
                db_file = gr.File(label="Vehicle DB (.xlsx)")
                load_btn = gr.Button("Load Database", size="sm")
                db_status = gr.Markdown("No database loaded.")
                
                # Video Section
                gr.Markdown("### 2. Input Source")
                src_radio = gr.Radio(["Upload Video", "Select from Folder"], value="Upload Video", label="Source Type")
                
                with gr.Group(visible=True) as upload_group:
                    vid_upload = gr.Video(label="Upload", sources=["upload"])
                
                with gr.Group(visible=False) as folder_group:
                    vid_dropdown = gr.Dropdown(label="Select Video", choices=get_videos_from_folder())
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")

                # Settings
                gr.Markdown("### 3. Settings")
                targets = gr.Textbox(label="Target Plates (Optional)", placeholder="AB12 CDE, XY55 ZZZ")
                with gr.Row():
                    chk_correct = gr.Checkbox(label="OCR Correction", value=True)
                    chk_valid = gr.Checkbox(label="Validate UK Format", value=True)
                slide_skip = gr.Slider(1, 10, value=2, step=1, label="Frame Skip (Performance)")
                
                run_btn = gr.Button("🚀 Start Analysis", variant="primary", size="lg")

            with gr.Column(scale=2):
                result_gallery = gr.Gallery(label="Captured Vehicles", columns=3, height="auto")
                result_text = gr.Markdown("### Waiting for analysis...")

        # Interactions
        load_btn.click(load_vehicle_database, inputs=[db_file], outputs=[db_status])
        
        def toggle_source(choice):
            return {
                upload_group: gr.update(visible=(choice == "Upload Video")),
                folder_group: gr.update(visible=(choice == "Select from Folder"))
            }

        src_radio.change(toggle_source, inputs=[src_radio], outputs=[upload_group, folder_group])
        
        refresh_btn.click(lambda: gr.update(choices=get_videos_from_folder()), outputs=[vid_dropdown])
        
        run_btn.click(
            process_video_ui,
            inputs=[src_radio, vid_upload, vid_dropdown, targets, chk_correct, chk_valid, slide_skip],
            outputs=[result_gallery, result_text]
        )

    return demo

if __name__ == "__main__":
    # Preload DB if exists
    if os.path.exists(EXCEL_DATABASE):
        load_vehicle_database(open(EXCEL_DATABASE, 'rb'))
        
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7555)
