import logging

from tools.snippets import (quick_log_setup, mkdir)
from tools.voc import (VOC_ocv, transforms_voc_ocv_eval)
from tools.dogs import (
        eval_stats_at_threshold, read_metadata,
        produce_gt_dog_boxes, produce_fake_centered_dog_boxes,
        visualize_dog_boxes)
from tools.gradcam import GradcamBBoxPredictor, produce_gradcam_bboxes


def dog_detection():
    """
    Here are two simple baselines for dog detection:
      - predicting a center box (50% of the area, center of the image)
      - predicting a center box, peeking at GT dog presence

    Two helper functions here:
      - a function that evaluates Average Precision and Recall
      - a function that visualizes predictions and ground truth
    """
    # / Config
    # This folder will be used to save VOC2007 dataset
    voc_folder = 'voc_dataset'

    # Dataset and Dataloader to quickly access the VOC2007 data
    dataset_test = VOC_ocv(
            voc_folder, year='2007', image_set='test',
            download=True, transforms=transforms_voc_ocv_eval)

    # Load first 500 items of metadata, create GT boxes
    metadata_test = read_metadata(dataset_test)
    metadata_test = dict(list(metadata_test.items())[:500])
    all_gt_dogs = produce_gt_dog_boxes(metadata_test)

    # / Baselines
    # Produce fake centerboxd dogs
    all_centerbox_dogs = produce_fake_centered_dog_boxes(
            metadata_test, scale=0.5)
    stats_df = eval_stats_at_threshold(all_centerbox_dogs, all_gt_dogs)
    log.info('CENTERBOX dogs:\n{}'.format(stats_df))

    # Cheat and constrain centerboxed dogs to images with GT dogs
    all_cheating_centerbox_dogs = {
            k: v for k, v in all_centerbox_dogs.items() if len(all_gt_dogs[k])}
    stats_df = eval_stats_at_threshold(all_cheating_centerbox_dogs, all_gt_dogs)
    log.info('CHEATING CENTERBOX dogs:\n{}'.format(stats_df))

    # Visualize the centerboxes boxes
    fold = mkdir('visualize/cheating_centerbox_dogs')
    visualize_dog_boxes(fold,
            all_cheating_centerbox_dogs, all_gt_dogs, metadata_test)
    
    #gradcam colution
    dataset_test_cropped = [dataset_test[i] for i in range(500)]
    bbox_predictor = GradcamBBoxPredictor()
    gradcam_bboxes = produce_gradcam_bboxes(dataset_test_cropped, bbox_predictor,class_number=4)
    stats_df = eval_stats_at_threshold(gradcam_bboxes, all_gt_dogs)
    log.info('GRADCAM dogs:\n{}'.format(stats_df))
    fold = mkdir('visualize/gradcam_centerbox_dogs')
    visualize_dog_boxes(fold,
            gradcam_bboxes, all_gt_dogs, metadata_test)
    


if __name__ == "__main__":
    # Establish logging to STDOUT
    log = quick_log_setup(logging.INFO)
    dog_detection()
