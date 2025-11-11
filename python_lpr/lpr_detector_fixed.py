#!/usr/bin/env python3

import sys
import gi
import argparse
import json
import os
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds

class LPRDetector:
    def __init__(self, video_path, output_json="lpr_results.json"):
        self.video_path = video_path
        self.output_json = output_json
        self.results = {"license_plates": []}
        self.frame_count = 0
        self.plate_detections = {}
        self.number_sources = 1
        
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
            
            # First pass: collect vehicle info
            vehicles = {}
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                # Store vehicle info (gie_unique_id=1 is TrafficCamNet)
                if obj_meta.unique_component_id == 1:
                    vehicles[obj_meta.object_id] = {
                        "type": obj_meta.obj_label,
                        "id": obj_meta.object_id
                    }

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
                    vehicle_type = vehicles.get(parent_id, {}).get("type", "Unknown") if parent_id else "Unknown"
                    
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
                                        plate = plate_text.strip()
                                        
                                        if plate not in self.plate_detections:
                                            self.plate_detections[plate] = {
                                                "first_seen_frame": self.frame_count,
                                                "vehicle_type": vehicle_type,
                                                "confidence": label_info.result_prob,
                                                "count": 1
                                            }
                                            print(f"\n🚗 NEW PLATE: {plate} | Vehicle: {vehicle_type} | Confidence: {label_info.result_prob:.2f}")
                                        else:
                                            self.plate_detections[plate]["count"] += 1
                                        
                                        result = {
                                            "frame": self.frame_count,
                                            "plate": plate,
                                            "vehicle_type": vehicle_type,
                                            "vehicle_id": parent_id,
                                            "confidence": label_info.result_prob
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

    def save_results(self):
        self.results["summary"] = {
            "total_frames": self.frame_count,
            "total_detections": len(self.results["license_plates"]),
            "unique_plates": len(self.plate_detections),
            "plates": []
        }
        
        for plate, info in self.plate_detections.items():
            self.results["summary"]["plates"].append({
                "plate": plate,
                "vehicle_type": info["vehicle_type"],
                "confidence": info["confidence"],
                "first_seen_frame": info["first_seen_frame"],
                "detection_count": info["count"]
            })
        
        with open(self.output_json, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Total Frames: {self.frame_count}")
        print(f"Total Detections: {len(self.results['license_plates'])}")
        print(f"Unique Plates: {len(self.plate_detections)}")
        print(f"\n{'='*60}")
        print(f"UNIQUE LICENSE PLATES:")
        print(f"{'='*60}")
        
        for plate, info in sorted(self.plate_detections.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"  📋 {plate:15s} | Vehicle: {info['vehicle_type']:10s} | "
                  f"Confidence: {info['confidence']:.2f} | Detected: {info['count']} times")
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {self.output_json}")
        print(f"{'='*60}\n")

    def run(self):
        Gst.init(None)

        print(f"\n{'='*60}")
        print(f"DeepStream License Plate Recognition")
        print(f"{'='*60}")
        print(f"Input Video: {self.video_path}")
        print(f"Output File: {self.output_json}")
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


def main():
    parser = argparse.ArgumentParser(description='DeepStream License Plate Recognition')
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('-o', '--output', default='lpr_results.json', 
                        help='Output JSON file (default: lpr_results.json)')
    
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1

    dict_path = "dict.txt"
    if not os.path.exists(dict_path):
        print(f"Error: Dictionary file not found at: {dict_path}")
        return 1

    detector = LPRDetector(args.video, args.output)
    detector.run()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())