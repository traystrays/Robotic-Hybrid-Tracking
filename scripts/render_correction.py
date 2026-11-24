"""Render predictions from study_correction.py over the original video."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def render(directory):
    report = json.loads((directory/'study.json').read_text())
    with np.load(directory/'dataset.npz') as data:
        observed, raw, corrected = [data[k] for k in ('observed_px','raw_px','corrected_px')]
        held = data['held_ecm']
    cap = cv2.VideoCapture(report['config']['input']['video'])
    size = tuple(int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT))
    destination = directory/'corrected.mp4'
    temporary = destination.with_suffix('.partial.mp4')
    writer = cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),cap.get(cv2.CAP_PROP_FPS),size)
    if not writer.isOpened(): raise RuntimeError('Could not open output video')
    preview_frame = report['frame_ranges_exclusive']['train'][0]
    test_start = report['frame_ranges_exclusive']['test'][0]
    try:
        for i in range(len(raw)):
            ok, frame = cap.read()
            if not ok: raise ValueError(f'Missing source frame {i}')
            for points, color, radius, thickness in [(raw[i],(0,0,255),6,-1),(observed[i],(0,255,0),6,-1),(corrected[i],(0,255,255),6,-1)]:
                for pixel in points[np.isfinite(points).all(axis=1)]:
                    x,y = np.rint(pixel).astype(int)
                    if -radius <= x < size[0]+radius and -radius <= y < size[1]+radius:
                        cv2.circle(frame,(x,y),radius,color,thickness,cv2.LINE_AA)
            split = next((s for s,(a,b) in report['frame_ranges_exclusive'].items() if a<=i<b),'gap')
            status = 'HELD ECM' if held[i] else split.upper()
            legend = f'Frame {i} | {status} | red: raw NDI   green: observed   yellow: corrected NDI'
            cv2.putText(frame,legend,(20,34),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,0),4,cv2.LINE_AA)
            cv2.putText(frame,legend,(20,34),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),1,cv2.LINE_AA)
            writer.write(frame)
            if i == preview_frame:
                cv2.imwrite(str(directory/'test_frame.jpg'), frame)
            if i == test_start:
                cv2.imwrite(str(directory/f'held_out_frame_{i}.jpg'), frame)
    finally:
        cap.release()
        writer.release()
    temporary.replace(destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study_directory',type=Path)
    render(parser.parse_args().study_directory)
