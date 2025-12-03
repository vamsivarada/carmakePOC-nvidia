#!/usr/bin/env python3

import sys
import gi
import argparse
import json
import os
import cv2
import numpy as np
from pathlib import Path
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds

class LicensePlateCorrector:
    """Post-processing corrections for common OCR errors"""
    
    def __init__(self, region='UK'):
        self.region = region
        
    def fix_missing_i_before_digit(self, plate_text):
        """Fix common issue where 'I' is missing before a digit"""
        clean = plate_text.replace(' ', '').upper()
        
        # UK format: LLDDLLL (2 letters, 2 digits, 3 letters)
        if self.region == 'UK' and len(clean) == 6:
            # Check if we have: LL D LLL (missing one char in digit section)
            if (len(clean) >= 3 and 
                clean[0:2].isalpha() and 
                clean[2].isdigit() and
                len(clean) >= 6 and
                clean[3:6].isalpha()):
                
                # Insert 'I' before the digit at position 2
                fixed = clean[:2] + 'I' + clean[2:]
                return fixed
        
        return clean
    
    def correct_uk_format(self, plate_text):
        """Enforce UK plate format: LLDDLLL"""
        clean = plate_text.replace(' ', '').upper()
        
        if len(clean) != 7:
            return clean
        
        corrected = list(clean)
        
        # Positions 0-1: Must be LETTERS
        for i in [0, 1]:
            if corrected[i].isdigit():
                if corrected[i] == '0': corrected[i] = 'O'
                elif corrected[i] == '1': corrected[i] = 'I'
        
        # Positions 2-3: Must be DIGITS  
        for i in [2, 3]:
            if corrected[i].isalpha():
                if corrected[i] == 'O': corrected[i] = '0'
                elif corrected[i] == 'I': corrected[i] = '1'
                elif corrected[i] == 'S': corrected[i] = '5'
                elif corrected[i] == 'Z': corrected[i] = '2'
                elif corrected[i] == 'B': corrected[i] = '8'
                elif corrected[i] == 'G': corrected[i] = '6'
        
        # Positions 4-6: Must be LETTERS
        for i in [4, 5, 6]:
            if corrected[i].isdigit():
                if corrected[i] == '0': corrected[i] = 'O'
                elif corrected[i] == '1': corrected[i] = 'I'
        
        return ''.join(corrected)
    
    def format_plate(self, plate_text):
        """Add space for readability: LLDD LLL"""
        clean = plate_text.replace(' ', '')
        if len(clean) == 7:
            return f"{clean[0:4]} {clean[4:7]}"
        return clean
    
    def process(self, plate_text):
        """Apply all corrections"""
        original = plate_text.strip()
        
        # Fix missing I
        plate_text = self.fix_missing_i_before_digit(plate_text)
        
        # Apply format rules
        if self.region == 'UK':
            plate_text = self.correct_uk_format(plate_text)
        
        # Format with space
        plate_text = self.format_plate(plate_text)
        
        return plate_text, plate_text != original.replace(' ', '')

class LPRDetector:
    def __init__(self, video_path, target_plates=None, output_json="lpr_results.json", output_folder="detected_vehicles", enable_corrections=True, use_opencv_color=False):
        self.video_path = video_path
        self.output_json = output_json
        self.output_folder = output_folder
        self.target_plates = set(target_plates) if target_plates else set()
        self.results = {"license_plates": []}
        self.frame_count = 0
        self.plate_detections = {}
        self.number_sources = 1
        self.saved_plates = set()  # Track which plates we've already saved
        self.frames_to_save = {}  # Store frame numbers and metadata for later extraction
        self.enable_corrections = enable_corrections
        self.corrector = LicensePlateCorrector(region='UK') if enable_corrections else None
        self.correction_stats = {"total": 0, "corrected": 0}
        self.use_opencv_color = use_opencv_color  # Use OpenCV for simple color detection
        
        # Create output folder if it doesn't exist
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        
        if self.target_plates:
            print(f"\n🎯 Searching for {len(self.target_plates)} target plate(s):")
            for plate in sorted(self.target_plates):
                print(f"   - '{plate}'")
            print(f"\n⚠️  ONLY plates matching the above will be saved!")
        else:
            print(f"\n📸 No target plates specified - will save images of ALL detected vehicles")
        
        if self.enable_corrections:
            print(f"✨ OCR corrections enabled (fixes missing 'I', format validation)")
        
        if self.use_opencv_color:
            print(f"🎨 OpenCV color detection enabled (basic color analysis)")
        
    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("\n" + "="*60)
            print("End-of-stream")
            self.save_results()
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}: {debug}")
            loop.quit()
        return True

    def decodebin_child_added(self, child_proxy, Object, name, user_data):
        print(f"Decodebin child added: {name}")
        if name.find("decodebin") != -1:
            Object.connect("child-added", self.decodebin_child_added, user_data)

    def create_source_bin(self, index, uri):
        bin_name = f"source-bin-{index:02d}"
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            sys.stderr.write(" Unable to create source bin \n")
            return None
        
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        if not uri_decode_bin:
            sys.stderr.write(" Unable to create uri decode bin \n")
            return None
        
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", self.cb_newpad, nbin)
        uri_decode_bin.connect("child-added", self.decodebin_child_added, nbin)
        
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            sys.stderr.write(" Failed to add ghost pad in source bin \n")
            return None
        return nbin

    def cb_newpad(self, decodebin, decoder_src_pad, data):
        caps = decoder_src_pad.get_current_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        source_bin = data
        features = caps.get_features(0)

        if gstname.find("video") != -1:
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")

    def detect_color_opencv(self, frame, bbox):
        """Simple color detection using OpenCV (no ML model needed)"""
        try:
            if not bbox:
                return "Unknown"
            
            # Extract vehicle region
            left = max(0, bbox['left'])
            top = max(0, bbox['top'])
            right = min(frame.shape[1], bbox['left'] + bbox['width'])
            bottom = min(frame.shape[0], bbox['top'] + bbox['height'])
            
            if left >= right or top >= bottom:
                return "Unknown"
            
            vehicle_roi = frame[top:bottom, left:right]
            
            if vehicle_roi.size == 0:
                return "Unknown"
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
            
            # Get the average color (focusing on center 50% to avoid shadows/reflections)
            h, w = hsv.shape[:2]
            center_roi = hsv[h//4:3*h//4, w//4:3*w//4]
            
            if center_roi.size == 0:
                return "Unknown"
            
            # Calculate mean HSV values
            mean_hue = np.mean(center_roi[:, :, 0])
            mean_sat = np.mean(center_roi[:, :, 1])
            mean_val = np.mean(center_roi[:, :, 2])
            
            # Simple color classification based on HSV
            # Low saturation = grayscale colors
            if mean_sat < 50:
                if mean_val < 60:
                    return "Black"
                elif mean_val < 120:
                    return "Grey"
                else:
                    return "White"
            
            # High saturation = chromatic colors
            if mean_hue < 15 or mean_hue > 165:
                return "Red"
            elif mean_hue < 25:
                return "Orange"
            elif mean_hue < 35:
                return "Yellow"
            elif mean_hue < 85:
                return "Green"
            elif mean_hue < 130:
                return "Blue"
            else:
                return "Purple"
                
        except Exception as e:
            return "Unknown"
    
    def save_frame_info(self, frame_number, obj_meta, plate_text, vehicle_type, vehicle_color="Unknown", vehicle_make="Unknown"):
        """Store frame info for later extraction"""
        try:
            # Store metadata for this frame/plate combination
            if plate_text not in self.frames_to_save:
                bbox = None
                # Get vehicle bounding box if obj_meta has parent (the vehicle)
                if obj_meta.parent:
                    vehicle_meta = obj_meta.parent
                    left = int(vehicle_meta.rect_params.left)
                    top = int(vehicle_meta.rect_params.top)
                    width = int(vehicle_meta.rect_params.width)
                    height = int(vehicle_meta.rect_params.height)
                    
                    # Validate bbox values
                    if width > 0 and height > 0 and left >= 0 and top >= 0:
                        bbox = {
                            'left': left,
                            'top': top,
                            'width': width,
                            'height': height
                        }
                        print(f"  📝 Marked frame {frame_number} for extraction (bbox: {width}x{height} at {left},{top})")
                    else:
                        print(f"  📝 Marked frame {frame_number} for extraction (invalid bbox, will use full frame)")
                else:
                    print(f"  📝 Marked frame {frame_number} for extraction (no parent vehicle, will use full frame)")
                
                self.frames_to_save[plate_text] = {
                    'frame_number': frame_number,
                    'vehicle_type': vehicle_type,
                    'vehicle_color': vehicle_color,
                    'vehicle_make': vehicle_make,
                    'bbox': bbox
                }
                return True
            return False
            
        except Exception as e:
            print(f"  ⚠️  Error storing frame info: {e}")
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
            
            if self.frame_count % 100 == 0:
                print(f"Processing frame {self.frame_count}... (Plates detected: {len(self.plate_detections)})")
            
            # First pass: collect vehicle info including color and type
            vehicles = {}
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Store vehicle info (gie_unique_id=1 is TrafficCamNet)
                if obj_meta.unique_component_id == 1:
                    vehicle_info = {
                        "type": obj_meta.obj_label,
                        "id": obj_meta.object_id,
                        "meta": obj_meta,
                        "color": "Unknown",
                        "make": "Unknown",
                        "confidence": obj_meta.confidence
                    }
                    
                    # Extract color and make from classifier metadata if available
                    l_classifier = obj_meta.classifier_meta_list
                    while l_classifier is not None:
                        try:
                            classifier_meta = pyds.NvDsClassifierMeta.cast(l_classifier.data)
                        except StopIteration:
                            break
                        
                        # Vehicle color classifier (gie_unique_id=4 if configured)
                        if classifier_meta.unique_component_id == 4:
                            l_label = classifier_meta.label_info_list
                            if l_label is not None:
                                try:
                                    label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                                    if label_info.result_label:
                                        vehicle_info["color"] = label_info.result_label
                                except StopIteration:
                                    pass
                        
                        # Vehicle make/model classifier (gie_unique_id=5 if configured)
                        elif classifier_meta.unique_component_id == 5:
                            l_label = classifier_meta.label_info_list
                            if l_label is not None:
                                try:
                                    label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                                    if label_info.result_label:
                                        vehicle_info["make"] = label_info.result_label
                                except StopIteration:
                                    pass
                        
                        try:
                            l_classifier = l_classifier.next
                        except StopIteration:
                            break
                    
                    vehicles[obj_meta.object_id] = vehicle_info

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            # Second pass: find license plates and their text
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Check if this is a license plate (gie_unique_id=2 is LPDNet)
                if obj_meta.unique_component_id == 2:
                    parent_id = obj_meta.parent.object_id if obj_meta.parent else None
                    
                    # Get full vehicle info
                    vehicle_info = vehicles.get(parent_id, {}) if parent_id else {}
                    vehicle_type = vehicle_info.get("type", "Unknown")
                    vehicle_color = vehicle_info.get("color", "Unknown")
                    vehicle_make = vehicle_info.get("make", "Unknown")
                    
                    # Check for LPR classification (gie_unique_id=3 is LPRNet)
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
                                        
                                        # Apply OCR corrections if enabled
                                        if self.enable_corrections and self.corrector:
                                            plate, was_corrected = self.corrector.process(plate_raw)
                                            self.correction_stats["total"] += 1
                                            if was_corrected:
                                                self.correction_stats["corrected"] += 1
                                        
                                        # Check if this is a target plate or if we're saving all
                                        is_target = (not self.target_plates) or (plate in self.target_plates)
                                        
                                        # Debug output
                                        if self.frame_count < 10 or plate not in self.plate_detections:  # Only show for first few frames or new plates
                                            if self.target_plates:
                                                print(f"  🔍 DEBUG: Plate='{plate}', In targets={plate in self.target_plates}, is_target={is_target}")
                                            else:
                                                print(f"  🔍 DEBUG: No target plates specified, is_target={is_target}")
                                        
                                        if plate not in self.plate_detections:
                                            self.plate_detections[plate] = {
                                                "first_seen_frame": self.frame_count,
                                                "vehicle_type": vehicle_type,
                                                "vehicle_color": vehicle_color,
                                                "vehicle_make": vehicle_make,
                                                "confidence": label_info.result_prob,
                                                "count": 1,
                                                "is_target": is_target,
                                                "raw_text": plate_raw if was_corrected else None
                                            }
                                            
                                            correction_info = f" (corrected from '{plate_raw}')" if was_corrected else ""
                                            vehicle_desc = f"{vehicle_color} {vehicle_type}" if vehicle_color != "Unknown" else vehicle_type
                                            if vehicle_make != "Unknown":
                                                vehicle_desc = f"{vehicle_color} {vehicle_make} {vehicle_type}"
                                            
                                            if is_target:
                                                print(f"\n🎯 MATCH FOUND: {plate}{correction_info}")
                                                print(f"   Vehicle: {vehicle_desc} | Confidence: {label_info.result_prob:.2f}")
                                            else:
                                                print(f"\n🚗 NEW PLATE: {plate}{correction_info}")
                                                print(f"   Vehicle: {vehicle_desc} | Confidence: {label_info.result_prob:.2f}")
                                        else:
                                            self.plate_detections[plate]["count"] += 1
                                        
                                        # Save frame info if this is a target plate and we haven't saved it yet
                                        saved = False
                                        if is_target and plate not in self.saved_plates:
                                            saved = self.save_frame_info(self.frame_count, obj_meta, plate, vehicle_type, vehicle_color, vehicle_make)
                                            if saved:
                                                self.saved_plates.add(plate)
                                        elif is_target and plate in self.saved_plates:
                                            print(f"  ⏭️  Already saved image for {plate}, skipping")
                                        elif not is_target:
                                            print(f"  ⏭️  Not a target plate ({plate}), not saving")
                                        
                                        result = {
                                            "frame": self.frame_count,
                                            "plate": plate,
                                            "plate_raw": plate_raw if was_corrected else plate,
                                            "was_corrected": was_corrected,
                                            "vehicle_type": vehicle_type,
                                            "vehicle_color": vehicle_color,
                                            "vehicle_make": vehicle_make,
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
        """Extract frames from video using OpenCV after processing"""
        if not self.frames_to_save:
            print("\nNo frames to extract.")
            return
        
        print(f"\n{'='*60}")
        print(f"EXTRACTING FRAMES FROM VIDEO")
        print(f"{'='*60}")
        print(f"Opening video: {self.video_path}")
        
        if self.use_opencv_color:
            print(f"🎨 OpenCV color detection: ENABLED")
        else:
            print(f"⚠️  OpenCV color detection: DISABLED (use --use-opencv-color to enable)")
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"  ⚠️  Error: Could not open video file")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {total_frames} frames, {frame_width}x{frame_height}")
        print(f"Frames to extract: {len(self.frames_to_save)}")
        
        # Sort frames to extract by frame number for efficient seeking
        frames_sorted = sorted(self.frames_to_save.items(), 
                              key=lambda x: x[1]['frame_number'])
        
        extracted = 0
        for plate, info in frames_sorted:
            frame_num = info['frame_number']
            bbox = info['bbox']
            
            # Seek to frame (0-indexed)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print(f"  ⚠️  Could not read frame {frame_num} for plate {plate}")
                continue
            
            if frame.size == 0:
                print(f"  ⚠️  Empty frame {frame_num} for plate {plate}")
                continue
            
            # Crop to vehicle if bbox available
            vehicle_img = None
            detected_color = "Unknown"
            
            if bbox:
                padding = 20
                left = max(0, bbox['left'] - padding)
                top = max(0, bbox['top'] - padding)
                right = min(frame.shape[1], bbox['left'] + bbox['width'] + padding)
                bottom = min(frame.shape[0], bbox['top'] + bbox['height'] + padding)
                
                # Validate cropping coordinates
                if left < right and top < bottom and right <= frame.shape[1] and bottom <= frame.shape[0]:
                    vehicle_img = frame[top:bottom, left:right]
                    
                    # Check if crop resulted in empty image
                    if vehicle_img.size == 0:
                        print(f"  ⚠️  Cropping resulted in empty image for {plate}, using full frame")
                        vehicle_img = frame
                    else:
                        # Detect color using OpenCV if enabled
                        if self.use_opencv_color:
                            detected_color = self.detect_color_opencv(frame, bbox)
                            if detected_color != "Unknown":
                                # Update the stored info with detected color
                                info['vehicle_color'] = detected_color
                                # Also update plate_detections so it shows in summary
                                if plate in self.plate_detections:
                                    self.plate_detections[plate]['vehicle_color'] = detected_color
                                print(f"  🎨 Detected color: {detected_color}")
                else:
                    print(f"  ⚠️  Invalid bbox for {plate} (left:{left}, top:{top}, right:{right}, bottom:{bottom}), using full frame")
                    vehicle_img = frame
            else:
                vehicle_img = frame
            
            # Final validation before saving
            if vehicle_img is None or vehicle_img.size == 0:
                print(f"  ⚠️  Cannot save empty image for plate {plate}")
                continue
            
            # Save image
            safe_plate = "".join(c if c.isalnum() else "_" for c in plate)
            filename = f"{safe_plate}_frame_{frame_num}.jpg"
            filepath = os.path.join(self.output_folder, filename)
            
            try:
                success = cv2.imwrite(filepath, vehicle_img)
                if success:
                    extracted += 1
                    print(f"  📸 [{extracted}/{len(self.frames_to_save)}] Saved: {filename} ({vehicle_img.shape[1]}x{vehicle_img.shape[0]})")
                    # Update results with image path
                    self.frames_to_save[plate]['image_path'] = filepath
                else:
                    print(f"  ⚠️  Failed to write image for plate {plate}")
            except Exception as e:
                print(f"  ⚠️  Error saving {plate}: {e}")
        
        cap.release()
        print(f"\n✅ Successfully extracted {extracted}/{len(self.frames_to_save)} images")
        
        if extracted > 0:
            print(f"\n📋 Images saved for these plates:")
            for plate in sorted(self.frames_to_save.keys()):
                if 'image_path' in self.frames_to_save[plate]:
                    print(f"   - {plate}")
        
        print(f"{'='*60}\n")

    def save_results(self):
        # Extract frames from video for matched plates
        self.extract_frames()
        
        self.results["summary"] = {
            "total_frames": self.frame_count,
            "total_detections": len(self.results["license_plates"]),
            "unique_plates": len(self.plate_detections),
            "target_plates": list(self.target_plates),
            "matched_targets": len([p for p, info in self.plate_detections.items() if info.get("is_target")]),
            "images_saved": len(self.saved_plates),
            "output_folder": self.output_folder,
            "corrections_enabled": self.enable_corrections,
            "correction_stats": self.correction_stats if self.enable_corrections else None,
            "plates": []
        }
        
        for plate, info in self.plate_detections.items():
            plate_summary = {
                "plate": plate,
                "vehicle_type": info["vehicle_type"],
                "vehicle_color": info.get("vehicle_color", "Unknown"),
                "vehicle_make": info.get("vehicle_make", "Unknown"),
                "confidence": info["confidence"],
                "first_seen_frame": info["first_seen_frame"],
                "detection_count": info["count"],
                "is_target": info.get("is_target", False)
            }
            
            # Add raw text if it was corrected
            if info.get("raw_text"):
                plate_summary["raw_text"] = info["raw_text"]
            
            # Add image path if available
            if plate in self.frames_to_save and 'image_path' in self.frames_to_save[plate]:
                plate_summary["image_path"] = self.frames_to_save[plate]['image_path']
            
            self.results["summary"]["plates"].append(plate_summary)
        
        with open(self.output_json, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Total Frames: {self.frame_count}")
        print(f"Total Detections: {len(self.results['license_plates'])}")
        print(f"Unique Plates: {len(self.plate_detections)}")
        
        if self.enable_corrections:
            corrected = self.correction_stats["corrected"]
            total = self.correction_stats["total"]
            if total > 0:
                pct = (corrected / total) * 100
                print(f"OCR Corrections: {corrected}/{total} ({pct:.1f}%)")
        
        if self.target_plates:
            print(f"Target Plates: {len(self.target_plates)}")
            print(f"Matched Targets: {len([p for p, info in self.plate_detections.items() if info.get('is_target')])}")
        
        print(f"Images Saved: {len(self.saved_plates)}")
        print(f"Output Folder: {self.output_folder}")
        print(f"\n{'='*60}")
        print(f"UNIQUE LICENSE PLATES:")
        print(f"{'='*60}")
        
        for plate, info in sorted(self.plate_detections.items(), key=lambda x: x[1]['count'], reverse=True):
            marker = "🎯" if info.get("is_target") else "📋"
            color = info.get('vehicle_color', 'Unknown')
            make = info.get('vehicle_make', 'Unknown')
            v_type = info['vehicle_type']
            
            # Build vehicle description
            if color != "Unknown" and make != "Unknown":
                vehicle_desc = f"{color} {make} {v_type}"
            elif color != "Unknown":
                vehicle_desc = f"{color} {v_type}"
            elif make != "Unknown":
                vehicle_desc = f"{make} {v_type}"
            else:
                vehicle_desc = v_type
            
            print(f"  {marker} {plate:15s} | {vehicle_desc:30s} | "
                  f"Confidence: {info['confidence']:.2f} | Detected: {info['count']} times")
        
        if self.target_plates:
            print(f"\n{'='*60}")
            print(f"TARGET PLATE STATUS:")
            print(f"{'='*60}")
            found = set([p for p in self.plate_detections.keys() if self.plate_detections[p].get("is_target")])
            not_found = self.target_plates - found
            
            if found:
                print(f"  ✅ Found: {', '.join(sorted(found))}")
            if not_found:
                print(f"  ❌ Not Found: {', '.join(sorted(not_found))}")
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {self.output_json}")
        print(f"Images saved to: {self.output_folder}/")
        print(f"{'='*60}\n")

    def run(self):
        Gst.init(None)

        print(f"\n{'='*60}")
        print(f"DeepStream License Plate Recognition")
        print(f"{'='*60}")
        print(f"Input Video: {self.video_path}")
        print(f"Output JSON: {self.output_json}")
        print(f"Output Folder: {self.output_folder}")
        print(f"{'='*60}\n")
        
        pipeline = Gst.Pipeline()
        if not pipeline:
            print("Unable to create Pipeline")
            return

        # Create source bin
        uri = f"file://{os.path.abspath(self.video_path)}"
        source_bin = self.create_source_bin(0, uri)
        if not source_bin:
            print("Unable to create source bin")
            return

        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
        sgie1 = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine1")
        sgie2 = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine2")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        sink = Gst.ElementFactory.make("fakesink", "fake-sink")

        if not streammux or not pgie or not sgie1 or not sgie2 or not nvvidconv or not nvosd or not sink:
            print("Unable to create elements")
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
            "/workspaces/numberplate-detection/python_lpr/config/pgie_trafficcamnet_config.txt")
        sgie1.set_property('config-file-path',
            "/workspaces/numberplate-detection/python_lpr/config/sgie_lpd_DetectNet2_us.txt")
        sgie2.set_property('config-file-path',
            "/workspaces/numberplate-detection/python_lpr/config/sgie_lpr_us_config.txt")

        sink.set_property('sync', 0)

        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad:
            print("Unable to get the sink pad of streammux")
            return
        
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            print("Unable to get source pad of source bin")
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
            print("Unable to get sink pad of nvosd")
            return
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

        print("Starting pipeline...\n")
        pipeline.set_state(Gst.State.PLAYING)
        
        try:
            loop.run()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\n\nError: {e}")

        pipeline.set_state(Gst.State.NULL)


def load_target_plates(filepath, normalize=True):
    """Load target license plates from a text file (one per line)"""
    plates = []
    with open(filepath, 'r') as f:
        for line in f:
            plate = line.strip()
            if plate and not plate.startswith('#'):  # Skip empty lines and comments
                plates.append(plate)
    
    # Normalize plates if requested (to match correction format)
    if normalize:
        corrector = LicensePlateCorrector(region='UK')
        normalized = []
        for plate in plates:
            corrected, _ = corrector.process(plate)
            normalized.append(corrected)
            if corrected != plate:
                print(f"  📝 Target plate normalized: '{plate}' → '{corrected}'")
        return normalized
    
    return plates


def main():
    parser = argparse.ArgumentParser(description='DeepStream License Plate Recognition with Image Capture')
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('-t', '--target-plates', 
                        help='Path to text file with target plates (one per line), or comma-separated plates')
    parser.add_argument('-o', '--output', default='lpr_results.json', 
                        help='Output JSON file (default: lpr_results.json)')
    parser.add_argument('-d', '--output-dir', default='detected_vehicles',
                        help='Output directory for vehicle images (default: detected_vehicles)')
    parser.add_argument('--no-corrections', action='store_true',
                        help='Disable OCR post-processing corrections')
    parser.add_argument('--use-opencv-color', action='store_true',
                        help='Enable basic color detection using OpenCV (no ML model needed)')
    
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1

    # Load target plates
    target_plates = None
    enable_corrections = not args.no_corrections
    
    if args.target_plates:
        if os.path.exists(args.target_plates):
            # Load from file
            target_plates = load_target_plates(args.target_plates, normalize=enable_corrections)
            print(f"Loaded {len(target_plates)} target plates from {args.target_plates}")
        else:
            # Treat as comma-separated list
            target_plates = [p.strip() for p in args.target_plates.split(',')]
            
            # Normalize if corrections are enabled
            if enable_corrections:
                corrector = LicensePlateCorrector(region='UK')
                normalized = []
                for plate in target_plates:
                    corrected, _ = corrector.process(plate)
                    normalized.append(corrected)
                    if corrected != plate:
                        print(f"  📝 Target plate normalized: '{plate}' → '{corrected}'")
                target_plates = normalized
            
            print(f"Using {len(target_plates)} target plates from command line")

    detector = LPRDetector(args.video, target_plates, args.output, args.output_dir, enable_corrections, args.use_opencv_color)
    detector.run()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())