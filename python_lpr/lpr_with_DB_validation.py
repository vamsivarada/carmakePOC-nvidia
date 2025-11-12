#!/usr/bin/env python3

import gradio as gr
import sys
import os
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import re

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds

INPUT_FOLDER = "/workspaces/cartheft/python_lpr/input_videos"
EXCEL_DATABASE = "/workspaces/cartheft/python_lpr/vehicle_database.xlsx"
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

# Precompile regex for speed
UK_PLATE_PATTERN = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$')

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
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            required_cols = ['license_plate', 'car_make', 'car_type', 'location']
            if any(col not in df.columns for col in required_cols):
                return False
            
            valid_count = 0
            for _, row in df.iterrows():
                plate = str(row['license_plate']).strip().upper().replace(' ', '')
                
                if not UKPlateValidator.is_valid_uk_plate(plate):
                    continue
                
                valid_count += 1
                plate_with_space = UKPlateValidator.format_uk_plate(plate)
                
                vehicle_info = {
                    'make': str(row['car_make']).strip(),
                    'type': str(row['car_type']).strip(),
                    'location': str(row['location']).strip(),
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
        
        for i in [0, 1]:
            if corrected[i].isdigit():
                if corrected[i] == '0': corrected[i] = 'O'
                elif corrected[i] == '1': corrected[i] = 'I'
        
        for i in [2, 3]:
            if corrected[i].isalpha():
                if corrected[i] == 'O': corrected[i] = '0'
                elif corrected[i] == 'I': corrected[i] = '1'
                elif corrected[i] == 'S': corrected[i] = '5'
                elif corrected[i] == 'Z': corrected[i] = '2'
                elif corrected[i] == 'B': corrected[i] = '8'
                elif corrected[i] == 'G': corrected[i] = '6'
        
        for i in [4, 5, 6]:
            if corrected[i].isdigit():
                if corrected[i] == '0': corrected[i] = 'O'
                elif corrected[i] == '1': corrected[i] = 'I'
        
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
        
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        
        print(f"\n🚀 Optimization: Processing every {frame_skip} frames for speed")
        if self.target_plates:
            print(f"🎯 Target plates: {len(self.target_plates)}")
        
    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("\n✅ Processing complete")
            self.save_results()
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
        if not nbin:
            return None
        
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        if not uri_decode_bin:
            return None
        
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", self.cb_newpad, nbin)
        uri_decode_bin.connect("child-added", self.decodebin_child_added, nbin)
        
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            return None
        return nbin

    def cb_newpad(self, decodebin, decoder_src_pad, data):
        caps = decoder_src_pad.get_current_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        source_bin = data

        if gstname.find("video") != -1:
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad\n")

    def save_frame_info(self, frame_number, obj_meta, plate_text, vehicle_type):
        try:
            if plate_text not in self.frames_to_save:
                bbox = None
                if obj_meta.parent:
                    vehicle_meta = obj_meta.parent
                    left = int(vehicle_meta.rect_params.left)
                    top = int(vehicle_meta.rect_params.top)
                    width = int(vehicle_meta.rect_params.width)
                    height = int(vehicle_meta.rect_params.height)
                    
                    if width > 0 and height > 0 and left >= 0 and top >= 0:
                        bbox = {'left': left, 'top': top, 'width': width, 'height': height}
                
                self.frames_to_save[plate_text] = {
                    'frame_number': frame_number,
                    'vehicle_type': vehicle_type,
                    'bbox': bbox
                }
                return True
            return False
        except:
            return False

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
            
            # Skip frames for speed
            if self.frame_count % self.frame_skip != 0:
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
                continue
            
            self.processed_frames += 1
            
            if self.processed_frames % 500 == 0:
                valid_plates = len([p for p in self.plate_detections if self.plate_detections[p].get('is_valid_uk', True)])
                print(f"⏱️  Frame {self.frame_count} | Valid plates: {valid_plates}")
            
            vehicles = {}
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                if obj_meta.unique_component_id == 1:
                    vehicles[obj_meta.object_id] = {
                        "type": obj_meta.obj_label,
                        "id": obj_meta.object_id,
                        "meta": obj_meta
                    }

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                if obj_meta.unique_component_id == 2:
                    parent_id = obj_meta.parent.object_id if obj_meta.parent else None
                    vehicle_type = vehicles.get(parent_id, {}).get("type", "Unknown") if parent_id else "Unknown"
                    
                    l_classifier = obj_meta.classifier_meta_list
                    while l_classifier is not None:
                        try:
                            classifier_meta = pyds.NvDsClassifierMeta.cast(l_classifier.data)
                        except StopIteration:
                            break
                        
                        if classifier_meta.unique_component_id == 3:
                            l_label = classifier_meta.label_info_list
                            while l_label is not None:
                                try:
                                    label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                                    plate_text = label_info.result_label
                                    
                                    if plate_text and plate_text.strip():
                                        plate_raw = plate_text.strip()
                                        plate = plate_raw
                                        was_corrected = False
                                        
                                        if self.enable_corrections and self.corrector:
                                            plate, was_corrected = self.corrector.process(plate_raw)
                                        
                                        self.validation_stats["total"] += 1
                                        is_valid_uk = UKPlateValidator.is_valid_uk_plate(plate)
                                        
                                        if is_valid_uk:
                                            self.validation_stats["valid"] += 1
                                        else:
                                            self.validation_stats["invalid"] += 1
                                            if self.validate_uk_format:
                                                try:
                                                    l_label = l_label.next
                                                except StopIteration:
                                                    break
                                                continue
                                        
                                        is_target = (not self.target_plates) or (plate in self.target_plates)
                                        
                                        if plate not in self.plate_detections:
                                            self.plate_detections[plate] = {
                                                "first_seen_frame": self.frame_count,
                                                "vehicle_type": vehicle_type,
                                                "confidence": label_info.result_prob,
                                                "count": 1,
                                                "is_target": is_target,
                                                "is_valid_uk": is_valid_uk,
                                                "raw_text": plate_raw if was_corrected else None
                                            }
                                            
                                            if is_target:
                                                print(f"🎯 FOUND: {plate} | {vehicle_type}")
                                        else:
                                            self.plate_detections[plate]["count"] += 1
                                        
                                        saved = False
                                        if is_valid_uk and is_target and plate not in self.saved_plates:
                                            saved = self.save_frame_info(self.frame_count, obj_meta, plate, vehicle_type)
                                            if saved:
                                                self.saved_plates.add(plate)
                                        
                                        result = {
                                            "frame": self.frame_count,
                                            "plate": plate,
                                            "plate_raw": plate_raw if was_corrected else plate,
                                            "was_corrected": was_corrected,
                                            "is_valid_uk": is_valid_uk,
                                            "vehicle_type": vehicle_type,
                                            "vehicle_id": parent_id,
                                            "confidence": label_info.result_prob,
                                            "is_target": is_target,
                                            "marked_for_save": saved
                                        }
                                        self.results["license_plates"].append(result)
                                        
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
        
        print(f"\n📸 Extracting {len(self.frames_to_save)} images...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
        
        frames_sorted = sorted(self.frames_to_save.items(), key=lambda x: x[1]['frame_number'])
        
        extracted = 0
        for plate, info in frames_sorted:
            frame_num = info['frame_number']
            bbox = info['bbox']
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            ret, frame = cap.read()
            
            if not ret or frame is None or frame.size == 0:
                continue
            
            vehicle_img = frame
            if bbox:
                padding = 20
                left = max(0, bbox['left'] - padding)
                top = max(0, bbox['top'] - padding)
                right = min(frame.shape[1], bbox['left'] + bbox['width'] + padding)
                bottom = min(frame.shape[0], bbox['top'] + bbox['height'] + padding)
                
                if left < right and top < bottom:
                    vehicle_img = frame[top:bottom, left:right]
                    if vehicle_img.size == 0:
                        vehicle_img = frame
            
            safe_plate = "".join(c if c.isalnum() else "_" for c in plate)
            filename = f"{safe_plate}_frame_{frame_num}.jpg"
            filepath = os.path.join(self.output_folder, filename)
            
            if cv2.imwrite(filepath, vehicle_img):
                extracted += 1
                self.frames_to_save[plate]['image_path'] = filepath
        
        cap.release()
        print(f"✅ Extracted {extracted} images")

    def save_results(self):
        self.extract_frames()
        
        valid_plate_detections = {
            plate: info for plate, info in self.plate_detections.items()
            if info.get('is_valid_uk', True)
        }
        
        self.results["summary"] = {
            "total_frames": self.frame_count,
            "processed_frames": self.processed_frames,
            "frame_skip": self.frame_skip,
            "total_detections": len(self.results["license_plates"]),
            "valid_uk_plates": len(valid_plate_detections),
            "invalid_plates_filtered": len(self.plate_detections) - len(valid_plate_detections),
            "target_plates": list(self.target_plates),
            "matched_targets": len([p for p, info in valid_plate_detections.items() if info.get("is_target")]),
            "images_saved": len(self.saved_plates),
            "output_folder": self.output_folder,
            "validation_stats": self.validation_stats,
            "plates": []
        }
        
        for plate, info in valid_plate_detections.items():
            plate_summary = {
                "plate": plate,
                "vehicle_type": info["vehicle_type"],
                "confidence": info["confidence"],
                "first_seen_frame": info["first_seen_frame"],
                "detection_count": info["count"],
                "is_target": info.get("is_target", False),
                "is_valid_uk": info.get("is_valid_uk", True)
            }
            
            if info.get("raw_text"):
                plate_summary["raw_text"] = info["raw_text"]
            
            if plate in self.frames_to_save and 'image_path' in self.frames_to_save[plate]:
                plate_summary["image_path"] = self.frames_to_save[plate]['image_path']
            
            self.results["summary"]["plates"].append(plate_summary)
        
        with open(self.output_json, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"RESULTS")
        print(f"{'='*50}")
        print(f"Frames: {self.frame_count} (processed: {self.processed_frames})")
        print(f"Valid UK Plates: {len(valid_plate_detections)}")
        print(f"Images Saved: {len(self.saved_plates)}")
        print(f"{'='*50}\n")

    def run(self):
        Gst.init(None)
        
        pipeline = Gst.Pipeline()
        if not pipeline:
            return

        uri = f"file://{os.path.abspath(self.video_path)}"
        source_bin = self.create_source_bin(0, uri)
        if not source_bin:
            return

        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
        sgie1 = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine1")
        sgie2 = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine2")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        sink = Gst.ElementFactory.make("fakesink", "fake-sink")

        if not all([streammux, pgie, sgie1, sgie2, nvvidconv, nvosd, sink]):
            return

        pipeline.add(source_bin)
        pipeline.add(streammux)
        pipeline.add(pgie)
        pipeline.add(sgie1)
        pipeline.add(sgie2)
        pipeline.add(nvvidconv)
        pipeline.add(nvosd)
        pipeline.add(sink)

        streammux.set_property('width', 1920)
        streammux.set_property('height', 1080)
        streammux.set_property('batch-size', 1)
        streammux.set_property('batched-push-timeout', 4000000)
        
        pgie.set_property('config-file-path', 
            "/workspaces/cartheft/python_lpr/config/pgie_trafficcamnet_config.txt")
        sgie1.set_property('config-file-path',
            "/workspaces/cartheft/python_lpr/config/sgie_lpd_DetectNet2_us.txt")
        sgie2.set_property('config-file-path',
            "/workspaces/cartheft/python_lpr/config/sgie_lpr_us_config.txt")

        sink.set_property('sync', 0)

        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad:
            return
        
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            return
        
        srcpad.link(sinkpad)
        streammux.link(pgie)
        pgie.link(sgie1)
        sgie1.link(sgie2)
        sgie2.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(sink)

        loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, loop)

        osdsinkpad = nvosd.get_static_pad("sink")
        if not osdsinkpad:
            return
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

        print("🚀 Starting processing...")
        pipeline.set_state(Gst.State.PLAYING)
        
        try:
            loop.run()
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted")
        except Exception as e:
            print(f"\n❌ Error: {e}")

        pipeline.set_state(Gst.State.NULL)


vehicle_db = None

def get_videos_from_folder():
    if not os.path.exists(INPUT_FOLDER):
        return []
    return sorted([f for f in os.listdir(INPUT_FOLDER) 
                   if any(f.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS)])

def load_vehicle_database(excel_file=None):
    global vehicle_db
    db_path = excel_file if excel_file else EXCEL_DATABASE
    vehicle_db = VehicleDatabase(db_path)
    
    if vehicle_db.database:
        valid_plates = len(set([k.replace(' ', '') for k in vehicle_db.database.keys()])) // 2
        return f"✅ Database loaded: {valid_plates} valid UK plates"
    return "⚠️ Failed to load database"

def process_video_ui(video_source, uploaded_video, folder_video, target_plates_text, 
                     enable_corrections, validate_uk, frame_skip, progress=gr.Progress()):
    global vehicle_db
    
    video_path = None
    if video_source == "Upload Video":
        if uploaded_video is None:
            return [], "❌ Upload a video"
        video_path = uploaded_video
    else:
        if not folder_video or folder_video == "No videos found":
            return [], "❌ Select a video"
        video_path = os.path.join(INPUT_FOLDER, folder_video)
        if not os.path.exists(video_path):
            return [], f"❌ Video not found"
    
    progress(0, desc="Starting...")
    
    temp_dir = tempfile.mkdtemp(prefix="lpr_")
    output_dir = os.path.join(temp_dir, "detected_vehicles")
    output_json = os.path.join(temp_dir, "results.json")
    
    try:
        target_plates = None
        if target_plates_text and target_plates_text.strip():
            plates = [p.strip() for p in target_plates_text.replace('\n', ',').split(',') if p.strip()]
            if enable_corrections and plates:
                corrector = LicensePlateCorrector(region='UK')
                target_plates = [corrector.process(p)[0] for p in plates]
            else:
                target_plates = plates
        
        progress(0.1, desc="Detecting...")
        
        detector = LPRDetector(
            video_path=video_path,
            target_plates=target_plates,
            output_json=output_json,
            output_folder=output_dir,
            enable_corrections=enable_corrections,
            validate_uk_format=validate_uk,
            frame_skip=frame_skip
        )
        
        import io
        from contextlib import redirect_stdout
        
        with redirect_stdout(io.StringIO()):
            detector.run()
        
        progress(0.9, desc="Finalizing...")
        
        with open(output_json, 'r') as f:
            results = json.load(f)
        
        images = []
        if os.path.exists(output_dir):
            images = [str(p) for p in sorted(Path(output_dir).glob("*.jpg"))]
        
        progress(1.0, desc="Done!")
        
        summary = results.get("summary", {})
        
        status_msg = f"### 📊 Results\n\n"
        status_msg += f"- **Frames:** {summary.get('total_frames', 0):,} (processed: {summary.get('processed_frames', 0):,})\n"
        status_msg += f"- **Valid UK Plates:** {summary.get('valid_uk_plates', 0)}\n"
        status_msg += f"- **Images Saved:** {len(images)}\n"
        
        if summary.get('invalid_plates_filtered', 0) > 0:
            status_msg += f"- **Filtered:** {summary.get('invalid_plates_filtered', 0)}\n"
        
        if not images:
            status_msg += "\n⚠️ No valid plates found"
        else:
            status_msg += f"\n### 🚗 Detected Vehicles\n\n"
            status_msg += "| Plate | Make | Type | Location | Status |\n"
            status_msg += "|-------|------|------|----------|--------|\n"
            
            for plate_info in summary.get('plates', []):
                if plate_info.get('is_valid_uk', True) and plate_info.get('is_target', True):
                    plate = plate_info.get('plate', '')
                    
                    if vehicle_db and vehicle_db.database:
                        vehicle_info = vehicle_db.get_vehicle_info(plate)
                        if vehicle_info:
                            make = vehicle_info['make']
                            car_type = vehicle_info['type']
                            location = vehicle_info['location']
                            db_status = "✅"
                        else:
                            make = "Unknown"
                            car_type = "Unknown"
                            location = "Unknown"
                            db_status = "❌"
                    else:
                        make = "N/A"
                        car_type = "N/A"
                        location = "N/A"
                        db_status = "⚠️"
                    
                    status_msg += f"| {plate} | {make} | {car_type} | {location} | {db_status} |\n"
        
        return images, status_msg
    
    except Exception as e:
        return [], f"❌ Error: {str(e)}"

def update_video_input(choice):
    if choice == "Upload Video":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        videos = get_videos_from_folder()
        if not videos:
            videos = ["No videos found"]
        return gr.update(visible=False), gr.update(visible=True, choices=videos, value=videos[0] if videos else None)

def create_ui():
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER, exist_ok=True)
    
    with gr.Blocks(title="LPR System", theme=gr.themes.Soft()) as demo:
        with gr.Row():
            gr.Markdown("# 🚗 License Plate Recognition (UK)")
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("**📊 Database**")
                    db_upload = gr.File(label="Excel Database", file_types=[".xlsx", ".xls"], type="filepath")
                    db_status = gr.Markdown(f"*{EXCEL_DATABASE}*")
                    load_db_btn = gr.Button("📂 Load", size="sm")
                
                with gr.Group():
                    gr.Markdown("**📹 Video**")
                    video_source = gr.Radio(["Upload Video", "Select from Folder"], value="Upload Video", label="Source")
                    video_upload = gr.Video(label="Upload", sources=["upload"], visible=True)
                    
                    videos = get_videos_from_folder() or ["No videos found"]
                    video_dropdown = gr.Dropdown(choices=videos, label="Select", value=videos[0], visible=False)
                    refresh_btn = gr.Button("🔄 Refresh", size="sm", visible=False)
                
                target_plates_input = gr.Textbox(
                    label="Target Plates (Optional)",
                    placeholder="PM16 XEH, MA09 LNO",
                    lines=3
                )
                
                with gr.Group():
                    enable_corrections = gr.Checkbox(label="OCR Corrections", value=True)
                    validate_uk = gr.Checkbox(label="UK Validation", value=True)
                    frame_skip = gr.Slider(1, 5, value=2, step=1, label="Frame Skip (higher = faster)")
                
                process_btn = gr.Button("🔍 Detect", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                status_output = gr.Markdown("*Ready*")
                gallery_output = gr.Gallery(label="Results", columns=3, height="auto")
        
        load_db_btn.click(load_vehicle_database, inputs=[db_upload], outputs=[db_status])
        video_source.change(update_video_input, inputs=[video_source], outputs=[video_upload, video_dropdown])
        video_source.change(lambda x: gr.update(visible=(x == "Select from Folder")), inputs=[video_source], outputs=[refresh_btn])
        refresh_btn.click(lambda: gr.update(choices=get_videos_from_folder()), outputs=[video_dropdown])
        process_btn.click(
            process_video_ui,
            inputs=[video_source, video_upload, video_dropdown, target_plates_input, enable_corrections, validate_uk, frame_skip],
            outputs=[gallery_output, status_output]
        )
    
    return demo

def main():
    if os.path.exists(EXCEL_DATABASE):
        load_vehicle_database(EXCEL_DATABASE)
    
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7555, share=False)

if __name__ == "__main__":
    main()