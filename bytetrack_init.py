import argparse
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("--track_high_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", type=float, default=0.1, help="tracking confidence threshold")
    parser.add_argument("--new_track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--aspect_ratio_thresh', type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=3000, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def bytetrack(bboxes, scores,cls,tracker):
    if bboxes.shape[0]>0:
        online_targets = tracker.update(bboxes, scores,cls)
        return np.array([t[:4] for t in online_targets]),[t[4] for t in online_targets]
    else:
        return None,None