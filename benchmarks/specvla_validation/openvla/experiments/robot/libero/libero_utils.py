"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import cv2
import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openvla.experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def get_libero_wrist_image(obs, resize_size):
    """Extracts wrist/eye-in-hand camera image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, frame_annotations=None, camera_name="agentview"):
    """Saves an MP4 replay of an episode with optional frame annotations.
    
    Args:
        rollout_images: List of images
        idx: Episode index
        success: Whether episode succeeded
        task_description: Task description string
        log_file: Optional log file handle
        frame_annotations: Optional list of dicts with keys:
            - 'composite_metric': float or None
            - 'mode': str (e.g., 'DB', 'AR')
            - 'action': numpy array of action values
        camera_name: Camera identifier (e.g., 'agentview', 'wrist') for filename
    """
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--camera={camera_name}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    
    for i, img in enumerate(rollout_images):
        # 创建图像副本以便在上面绘制
        annotated_img = img.copy()
        
        # 如果有标注信息，在图像上绘制
        if frame_annotations is not None and i < len(frame_annotations):
            annotation = frame_annotations[i]
            
            # 设置文本参数
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            line_height = 20
            y_offset = 20
            
            # 绘制模式（Verify/NoVerify）
            mode = annotation.get('mode', 'N/A')
            # 转换模式名称：DB相关 -> NoVerify, AR -> Verify
            if mode.startswith('DB'):
                display_mode = 'NoVerify'
            elif mode == 'AR':
                display_mode = 'Verify'
            else:
                display_mode = mode
            mode_text = f"Mode: {display_mode}"
            cv2.putText(annotated_img, mode_text, (10, y_offset), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            y_offset += line_height
            
            # 绘制综合指标
            composite_metric = annotation.get('composite_metric')
            if composite_metric is not None:
                metric_text = f"Composite: {composite_metric:.4f}"
            else:
                metric_text = "Composite: N/A"
            cv2.putText(annotated_img, metric_text, (10, y_offset), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            y_offset += line_height
            
            # 绘制action (分两行显示)
            action = annotation.get('action')
            if action is not None:
                # 第一行：前4个值 (xyz + rotation)
                action_text1 = f"Act: [{action[0]:.3f},{action[1]:.3f},{action[2]:.3f},{action[3]:.3f}"
                cv2.putText(annotated_img, action_text1, (10, y_offset), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                y_offset += line_height
                
                # 第二行：后3个值 (rotation + gripper)
                action_text2 = f"     {action[4]:.3f},{action[5]:.3f},{action[6]:.3f}]"
                cv2.putText(annotated_img, action_text2, (10, y_offset), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        
        video_writer.append_data(annotated_img)
    
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
