#!/usr/bin/env python3

import sys
import gi
import argparse
import json
import os
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

# DeepStream imports
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds

class LPRWithCarMakeDetector:
    def __init__(self, video_path, output_json="lpr_carmake_results.json"):
        self.video_path = video_path
        self.output_json = output_json
        self.results = {
            "detections": []
        }
        self.frame_count = 0
        self.unique_detections = {}  # Track unique combinations
        
    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("\n" + "="*70)
            print("End-of-stream")
            self.save_results()
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}: {debug}")
            loop.quit()
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
            
            # Print progress every 100 frames
            if self.frame_count % 100 == 0:
                print(f"Processing frame {self.frame_count}... (Unique detections: {len(self.unique_detections)})")
            
            l_obj = frame_meta.obj_meta_list
            
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Check for vehicle detection (gie_unique_id=1 is TrafficCamNet)
                if obj_meta.unique_component_id == 1:
                    vehicle_type = obj_meta.obj_label
                    vehicle_id = obj_meta.object_id
                    
                    # Get car make from classifier
                    car_make = "Unknown"
                    car_make_confidence = 0.0
                    
                    l_classifier = obj_meta.classifier_meta_list
                    while l_classifier is not None:
                        try:
                            classifier_meta = pyds.NvDsClassifierMeta.cast(l_classifier.data)
                        except StopIteration:
                            break
                        
                        # Check if this is VehicleMakeNet (gie_unique_id=4)
                        if classifier_meta.unique_component_id == 4:
                            l_label = classifier_meta.label_info_list
                            while l_label is not None:
                                try:
                                    label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                                    if label_info.result_label:
                                        car_make = label_info.result_label
                                        car_make_confidence = label_info.result_prob
                                        break
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
                    
                    # Check for license plate on this vehicle
                    license_plate = None
                    plate_confidence = 0.0
                    
                    l_obj_sec = obj_meta.obj_meta_list
                    while l_obj_sec is not None:
                        try:
                            obj_meta_sec = pyds.NvDsObjectMeta.cast(l_obj_sec.data)
                        except StopIteration:
                            break
                        
                        # Check if this is a license plate (gie_unique_id=2 is LPDNet)
                        if obj_meta_sec.unique_component_id == 2:
                            # Check for LPR results (gie_unique_id=3 is LPRNet)
                            l_classifier_lpr = obj_meta_sec.classifier_meta_list
                            while l_classifier_lpr is not None:
                                try:
                                    classifier_meta_lpr = pyds.NvDsClassifierMeta.cast(l_classifier_lpr.data)
                                except StopIteration:
                                    break
                                
                                if classifier_meta_lpr.unique_component_id == 3:
                                    l_label_lpr = classifier_meta_lpr.label_info_list
                                    while l_label_lpr is not None:
                                        try:
                                            label_info_lpr = pyds.NvDsLabelInfo.cast(l_label_lpr.data)
                                            plate_text = label_info_lpr.result_label
                                            
                                            if plate_text and plate_text.strip():
                                                license_plate = plate_text.strip()
                                                plate_confidence = label_info_lpr.result_prob
                                                break
                                                
                                        except StopIteration:
                                            break
                                        try:
                                            l_label_lpr = l_label_lpr.next
                                        except StopIteration:
                                            break
                                
                                try:
                                    l_classifier_lpr = l_classifier_lpr.next
                                except StopIteration:
                                    break
                        
                        try:
                            l_obj_sec = l_obj_sec.next
                        except StopIteration:
                            break
                    
                    # If we have a license plate, store the complete detection
                    if license_plate:
                        detection_key = f"{license_plate}_{car_make}"
                        
                        if detection_key not in self.unique_detections:
                            self.unique_detections[detection_key] = {
                                "plate": license_plate,
                                "car_make": car_make,
                                "vehicle_type": vehicle_type,
                                "plate_confidence": plate_confidence,
                                "make_confidence": car_make_confidence,
                                "first_seen_frame": self.frame_count,
                                "count": 1
                            }
                            print(f"\n🚗 NEW DETECTION:")
                            print(f"   Plate: {license_plate} | Make: {car_make.upper()} | Type: {vehicle_type}")
                            print(f"   Confidence - Plate: {plate_confidence:.2f}, Make: {car_make_confidence:.2f}")
                        else:
                            self.unique_detections[detection_key]["count"] += 1
                        
                        # Add to detailed results
                        result = {
                            "frame": self.frame_count,
                            "license_plate": license_plate,
                            "car_make": car_make,
                            "vehicle_type": vehicle_type,
                            "vehicle_id": vehicle_id,
                            "plate_confidence": plate_confidence,
                            "make_confidence": car_make_confidence
                        }
                        self.results["detections"].append(result)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def save_results(self):
        """Save results to JSON file"""
        
        # Add summary
        self.results["summary"] = {
            "total_frames": self.frame_count,
            "total_detections": len(self.results["detections"]),
            "unique_vehicles": len(self.unique_detections),
            "vehicles": []
        }
        
        for key, info in self.unique_detections.items():
            self.results["summary"]["vehicles"].append(info)
        
        with open(self.output_json, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Total Frames: {self.frame_count}")
        print(f"Total Detections: {len(self.results['detections'])}")
        print(f"Unique Vehicles: {len(self.unique_detections)}")
        print(f"\n{'='*70}")
        print(f"DETECTED VEHICLES:")
        print(f"{'='*70}")
        
        for key, info in sorted(self.unique_detections.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"  📋 Plate: {info['plate']:12s} | "
                  f"Make: {info['car_make'].upper():12s} | "
                  f"Type: {info['vehicle_type']:8s}")
            print(f"     Confidence - Plate: {info['plate_confidence']:.2f}, Make: {info['make_confidence']:.2f} | "
                  f"Seen: {info['count']} times")
            print()
        
        print(f"{'='*70}")
        print(f"Results saved to: {self.output_json}")
        print(f"{'='*70}\n")

    def run(self):
        Gst.init(None)

        print(f"\n{'='*70}")
        print(f"DeepStream LPR + Car Make Recognition")
        print(f"{'='*70}")
        print(f"Input Video: {self.video_path}")
        print(f"Output File: {self.output_json}")
        print(f"{'='*70}\n")
        print("Building pipeline with 4 inference engines:")
        print("  1. TrafficCamNet - Car Detection")
        print("  2. LPDNet - License Plate Detection")
        print("  3. LPRNet - License Plate Recognition")
        print("  4. VehicleMakeNet - Car Make Classification")
        print()
        
        pipeline = Gst.Pipeline()

        if not pipeline:
            print("Unable to create Pipeline")
            return

        # Create elements
        source = Gst.ElementFactory.make("filesrc", "file-source")
        h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        
        # Primary GIE - Car Detection (TrafficCamNet)
        pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
        
        # Secondary GIE 1 - License Plate Detection (LPDNet)
        sgie1 = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine1")
        
        # Secondary GIE 2 - License Plate Recognition (LPRNet)
        sgie2 = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine2")
        
        # Secondary GIE 3 - Car Make Classification (VehicleMakeNet)
        sgie3 = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine3")
        
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        sink = Gst.ElementFactory.make("fakesink", "fake-sink")

        if not source or not h264parser or not decoder or not streammux or not pgie \
                or not sgie1 or not sgie2 or not sgie3 or not nvvidconv or not nvosd or not sink:
            print("Unable to create elements")
            return

        # Set properties
        source.set_property('location', self.video_path)
        streammux.set_property('width', 1920)
        streammux.set_property('height', 1080)
        streammux.set_property('batch-size', 1)
        streammux.set_property('batched-push-timeout', 4000000)
        
        # Config files - ABSOLUTE PATHS
        pgie.set_property('config-file-path', 
            "/opt/nvidia/deepstream/deepstream-8.0/sources/apps/sample_apps/deepstream_tao_apps/configs/nvinfer/trafficcamnet_tao/pgie_trafficcamnet_config.txt")
        sgie1.set_property('config-file-path',
            "/opt/nvidia/deepstream/deepstream-8.0/sources/apps/sample_apps/deepstream_tao_apps/configs/nvinfer/LPD_us_tao/sgie_lpd_DetectNet2_us.txt")
        sgie2.set_property('config-file-path',
            "/opt/nvidia/deepstream/deepstream-8.0/sources/apps/sample_apps/deepstream_tao_apps/configs/nvinfer/lpr_us_tao/sgie_lpr_us_config.txt")
        sgie3.set_property('config-file-path',
            "/opt/nvidia/deepstream/deepstream-8.0/sources/apps/sample_apps/deepstream_tao_apps/configs/nvinfer/vehiclemakenet_tao/sgie_vehiclemakenet_config.txt")

        sink.set_property('sync', 0)

        # Add elements to pipeline
        pipeline.add(source)
        pipeline.add(h264parser)
        pipeline.add(decoder)
        pipeline.add(streammux)
        pipeline.add(pgie)
        pipeline.add(sgie1)
        pipeline.add(sgie2)
        pipeline.add(sgie3)
        pipeline.add(nvvidconv)
        pipeline.add(nvosd)
        pipeline.add(sink)

        # Link elements
        source.link(h264parser)
        h264parser.link(decoder)

        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad:
            print("Unable to get the sink pad of streammux")
            return
        srcpad = decoder.get_static_pad("src")
        if not srcpad:
            print("Unable to get source pad of decoder")
            return
        srcpad.link(sinkpad)

        streammux.link(pgie)
        pgie.link(sgie1)
        sgie1.link(sgie2)
        sgie2.link(sgie3)
        sgie3.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(sink)

        # Create an event loop
        loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, loop)

        # Add probe to get metadata
        osdsinkpad = nvosd.get_static_pad("sink")
        if not osdsinkpad:
            print("Unable to get sink pad of nvosd")
            return
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self.osd_sink_pad_buffer_probe, 0)

        # Start pipeline
        print("Starting pipeline...\n")
        pipeline.set_state(Gst.State.PLAYING)
        
        try:
            loop.run()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\n\nError: {e}")

        # Cleanup
        pipeline.set_state(Gst.State.NULL)


def main():
    parser = argparse.ArgumentParser(
        description='DeepStream LPR + Car Make Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('video', help='Path to input video file (H264/MP4)')
    parser.add_argument('-o', '--output', default='lpr_carmake_results.json', 
                        help='Output JSON file (default: lpr_carmake_results.json)')
    
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1

    dict_path = "/opt/nvidia/deepstream/deepstream-8.0/sources/apps/sample_apps/deepstream_tao_apps/apps/tao_others/deepstream_lpr_app/dict.txt"
    if not os.path.exists(dict_path):
        print(f"Error: Dictionary file not found at: {dict_path}")
        return 1

    detector = LPRWithCarMakeDetector(args.video, args.output)
    detector.run()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())