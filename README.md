# SFSORT: Scene Features-based Simple Online Real-Time Tracker

## What is multi-object tracking?
Multi-object tracking (MOT) is the process of detecting and tracking multiple moving objects over time in a video. The objects can belong to the same class (e.g., all humans) or different classes (e.g., humans and cars).

## What are SFSORT’s specifications?
- SFSORT follows the tracking-by-detection paradigm, where an arbitrary object detector identifies objects in a video frame. SFSORT then assigns unique IDs to the detected objects, ensuring that objects retain the same ID across frames. It is compatible with most object detectors, with the only requirement being that the average prediction score for occluded or blurred objects must be lower than for clear, normal objects.
- SFSORT is a real-time tracker, meaning it can process frames at speeds greater than 30 frames per second. Its exceptionally high processing speed, capable of tracking over 200 objects at more than 300 frames per second, sets it apart from other trackers.
- SFSORT can operate both online and offline. In online scenarios, such as tracking objects in live streams, SFSORT uses only information from the last and next-to-last frames to assign IDs to objects. In offline scenarios, like tracking objects in videos, it initially assigns IDs using online tracking. Then, if an object is undetected for a short period across some frames, its position in those frames is estimated based on where it appears in visible frames, ensuring it retains a unique ID. This post-processing improves tracking accuracy, and SFSORT is the first to account for camera movement and scene depth in this step.

## How to tune tracking parameters?
Usually, a set of videos similar to the test videos, known as the validation set, is used to tune tracking parameters. To optimize tracking accuracy, an iterative experiment is conducted where parameter values are changed in each iteration, and tracking accuracy on the validation videos is measured to find the values that maximize accuracy. Since the number of possible parameter combinations in such an experiment is high, the following points about each parameter can significantly reduce the number of iterations:
1. **dynamic_tuning**: Set this to `True` if the tracker needs to process frames with a large difference in the number of objects.
2. **cth**: This parameter is only effective when `dynamic_tuning` is enabled. Set it to the average prediction score reported by the object detector for all objects. Default value: `0.7`.
3. **high_th**: Set this to the lowest prediction score reported by the object detector for normal and clear objects.
4. **high_th_m**: This parameter is only effective when `dynamic_tuning` is enabled. Set it to a value between `0.02` and `0.1`. If you observe more drops in `high_th` for crowded scenes compared to normal scenes, set this parameter to higher values.
5. **match_th_first**: Set it to a value between `0` and `0.67`. Higher values relax the association conditions, which can be useful when there is poor overlap between the bounding boxes of the same object across video frames or when the object’s shape changes significantly. It is recommended to use higher values for this parameter.
6. **match_th_first_m**: This parameter is only effective when `dynamic_tuning` is enabled. Set it to a value between `0.02` and `0.08`. Increase the value if the association for high-score detections is too strict, causing ID switches in crowded scenes.
7. **match_th_second**: Set it to a value between `0` and `1`. Higher values relax the association conditions, which can be useful when there is poor overlap between the bounding boxes of the same object across video frames. It is recommended to use lower values for this parameter.
8. **low_th**: Set this to the lowest prediction score reported by the object detector for occluded or blurred objects.
9. **new_track_th**: When `dynamic_tuning` is enabled, set this parameter to a value below `high_th`. Otherwise, set it to a value slightly above `high_th`.
10. **new_track_th_m**: This parameter is only effective when `dynamic_tuning` is enabled. Set it to a value between `0.02` and `0.08`. Increase the value if you observe too many ID switches for objects within the crowd.
11. **marginal_timeout**: Set this parameter to determine how many frames the tracker attempts to revisit a track lost at frame margins. Set it to an integer value between `0.1 * frame_rate` and `0.9 * frame_rate`. Higher values allow a track to disappear for a longer time.
12. **central_timeout**: Set this parameter to determine how many frames the tracker attempts to revisit a track lost at the frame center. Set it to an integer value between `0.5 * frame_rate` and `1.5 * frame_rate`. Higher values allow a track to disappear for a longer time.
13. **horizontal_margin**: Determines the horizontal margins used in attempts to revisit a track lost at frame margins. Set it to an integer value between `0.05 * frame_width` and `0.1 * frame_width`. 
14. **vertical_margin**: Determines the vertical margins used in attempts to revisit a track lost at frame margins. Set it to an integer value between `0.05 * frame_ height` and `0.1 * frame_ height`. 
15. **frame_width**: An integer indicating the width of video frames.
16. **frame_ height**: An integer indicating the width of video frames.

## How to access scientific details about SFSORT?
[Our paper](https://arxiv.org/abs/2404.07553) provides full details.

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
