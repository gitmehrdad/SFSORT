# SFSORT: Scene Features-based Simple Online Real-Time Tracker

## What is Multi-object tracking?
Multi-object tracking (MOT) is the process of detecting and tracking multiple moving objects over time in a video. The objects can belong to the same class (e.g., all humans) or different classes (e.g., humans and cars).

## What are SFSORT’s specifications?
- SFSORT follows the tracking-by-detection paradigm, where an arbitrary object detector identifies objects in a video frame. SFSORT then assigns unique IDs to the detected objects, ensuring that objects retain the same ID across frames. It is compatible with most object detectors, with the only requirement being that the average prediction score for occluded or blurred objects must be lower than for clear, normal objects.
- SFSORT is a real-time tracker, meaning it can process frames at speeds greater than 30 frames per second. Its exceptionally high processing speed, capable of tracking over 200 objects at more than 300 frames per second, sets it apart from other trackers.
- SFSORT can operate both online and offline. In online scenarios, such as tracking objects in live streams, SFSORT uses only information from the last and next-to-last frames to assign IDs to objects. In offline scenarios, like tracking objects in videos, it initially assigns IDs using online tracking. Then, if an object is undetected for a short period across some frames, its position in those frames is estimated based on where it appears in visible frames, ensuring it retains a unique ID. This post-processing improves tracking accuracy, and SFSORT is the first to account for camera movement and scene depth in this step.

## Finding the Path to the Comprehensive Explanation
Full details are available at https://arxiv.org/abs/2404.07553

## Citation
```
@misc{sfsort,
    title={SFSORT: Scene Features-based Simple Online Real-Time Tracker},
    author={M. M. Morsali and Z. Sharifi and F. Fallah and S. Hashembeiki and H. Mohammadzade and S. Bagheri Shouraki},
    year={2024},
    doi = {10.48550/arXiv.2404.07553}, 
    url = {https://arxiv.org/abs/2404.07553}, 
    eprint={2404.07553},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
