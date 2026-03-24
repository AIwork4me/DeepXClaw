"""COCO 80-class labels and color palette for YOLOv26n visualization."""

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

NUM_CLASSES = len(COCO_LABELS)


def generate_colors(n: int = NUM_CLASSES, seed: int = 42):
    """Generate n distinct BGR colors for visualization."""
    import numpy as np
    rng = np.random.RandomState(seed)
    colors = rng.randint(0, 255, size=(n, 3), dtype=np.uint8)
    # Ensure brightness: at least one channel > 100
    for i in range(n):
        if colors[i].max() < 100:
            colors[i][rng.randint(3)] = 200
    return colors


COLORS = generate_colors()