# object_detection_tracking.py
import cv2
import torch
import numpy as np
from torchvision import transforms
from collections import deque


class ObjectDetectorTracker:
    """基于YOLO的目标检测与追踪系统"""

    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # 加载YOLOv5模型
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = confidence_threshold
        self.model.iou = nms_threshold

        # 追踪器
        self.trackers = {}
        self.track_id = 0
        self.track_history = {}

    def detect_objects(self, frame):
        """检测帧中的目标"""
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        return detections

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        intersect_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        intersect_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersect_area = intersect_w * intersect_h

        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersect_area

        return intersect_area / union_area if union_area > 0 else 0

    def update_trackers(self, detections):
        """更新目标追踪器"""
        current_boxes = []

        for _, detection in detections.iterrows():
            box = [detection['xmin'], detection['ymin'],
                   detection['xmax'], detection['ymax']]
            current_boxes.append({
                'box': box,
                'class': detection['name'],
                'confidence': detection['confidence']
            })

        # 匹配现有追踪器
        matched_ids = set()
        new_trackers = {}

        for obj in current_boxes:
            best_iou = 0
            best_id = None

            for track_id, tracker in self.trackers.items():
                iou = self.calculate_iou(obj['box'], tracker['box'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_id = track_id

            if best_id is not None:
                new_trackers[best_id] = obj
                matched_ids.add(best_id)

                if best_id not in self.track_history:
                    self.track_history[best_id] = deque(maxlen=30)

                center = (
                    int((obj['box'][0] + obj['box'][2]) / 2),
                    int((obj['box'][1] + obj['box'][3]) / 2)
                )
                self.track_history[best_id].append(center)
            else:
                # 新目标
                self.track_id += 1
                new_trackers[self.track_id] = obj
                self.track_history[self.track_id] = deque(maxlen=30)

        self.trackers = new_trackers
        return self.trackers

    def draw_detections(self, frame, trackers):
        """在帧上绘制检测结果"""
        colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

        for track_id, obj in trackers.items():
            box = obj['box']
            color = tuple(map(int, colors[track_id % 100]))

            # 绘制边界框
            cv2.rectangle(frame,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          color, 2)

            # 绘制标签
            label = f"ID:{track_id} {obj['class']} {obj['confidence']:.2f}"
            cv2.putText(frame, label,
                        (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 绘制轨迹
            if track_id in self.track_history:
                points = list(self.track_history[track_id])
                for i in range(1, len(points)):
                    cv2.line(frame, points[i - 1], points[i], color, 2)

        return frame

    def process_video(self, video_path, output_path=None):
        """处理视频文件"""
        cap = cv2.VideoCapture(video_path)

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 检测目标
            detections = self.detect_objects(frame)

            # 更新追踪
            trackers = self.update_trackers(detections)

            # 绘制结果
            frame = self.draw_detections(frame, trackers)

            # 显示帧数和目标数
            info_text = f"Frame: {frame_count} | Objects: {len(trackers)}"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if output_path:
                out.write(frame)

            cv2.imshow('Object Detection & Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()


# 使用示例
if __name__ == "__main__":
    detector = ObjectDetectorTracker(confidence_threshold=0.5)

    # 处理视频
    # detector.process_video('input_video.mp4', 'output_video.mp4')

    # 处理摄像头
    # detector.process_video(0)  # 0表示默认摄像头

    print("Object Detector & Tracker initialized successfully!")
