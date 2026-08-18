import copy
import cv2 
import numpy as np
import os
import sys
import threading
from threading import Condition

import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, JointState, Joy
from geometry_msgs.msg import TwistStamped, PoseStamped, Twist, TransformStamped
try:
    from control_msgs.msg import DynamicJointState
except ModuleNotFoundError:
    DynamicJointState = None
from datetime import datetime, timezone
from medcvr_il.common.config_schema import DataWriterConfig, RecorderConfig, SynchSubscriberNodeConfig, TopicConfig

def encoding_to_dtype_with_channels(encoding):
    """
    Map ROS image encoding strings to (numpy dtype, number of channels).
    Add or edit encodings as needed.
    """
    if not hasattr(encoding_to_dtype_with_channels, "encoding_map"): # Static
        encoding_map = {
            'mono8':    ('uint8', 1),
            'mono16':   ('uint16', 1),
            'bgr8':     ('uint8', 3),
            'rgb8':     ('uint8', 3),
            'rgba8':    ('uint8', 4),
            'bgra8':    ('uint8', 4),
            '32FC1':    ('float32', 1),
            '32FC3':    ('float32', 3),
            # Add more encodings if needed
        }
    if encoding not in encoding_map:
        raise ValueError(f"Unsupported encoding: {encoding}")
    return encoding_map[encoding]

def ros_image_to_numpy(img_msg):
    """
    Convert a ROS2 Image message-like object to a numpy ndarray using only numpy.
    The output image will be in RGB8 format.
    
    img_msg must have these attributes:
    - data: bytes or list of bytes
    - height: int
    - width: int
    - step: int (full row length in bytes)
    - encoding: str (e.g. 'rgb8', 'mono8', 'bgr8', '32FC1')
    - is_bigendian: bool
    """

    # 1. Get dtype and channels from encoding
    dtype_str, n_channels = encoding_to_dtype_with_channels(img_msg.encoding)
    dtype = np.dtype(dtype_str)

    # 2. Set byte order according to is_bigendian flag
    byteorder = '>' if img_msg.is_bigendian else '<'
    dtype = dtype.newbyteorder(byteorder)

    # 3. Convert raw data buffer to numpy array
    # If data is a list (unlikely for ROS), convert to bytes first
    if isinstance(img_msg.data, list):
        img_buf = bytes(img_msg.data)
    else:
        img_buf = img_msg.data

    arr = np.frombuffer(img_buf, dtype=dtype)

    # 4. Reshape depending on channels and step
    # step is total bytes per row, so width may be less than step / (dtype size * channels)
    if n_channels == 1:
        row_len = img_msg.step // dtype.itemsize
        im = arr.reshape((img_msg.height, row_len))
        im = np.ascontiguousarray(im[:, :img_msg.width])
    else:
        row_len = img_msg.step // (dtype.itemsize * n_channels)
        im = arr.reshape((img_msg.height, row_len, n_channels))
        im = np.ascontiguousarray(im[:, :img_msg.width, :])

    # 5. Handle system endianness vs message endianness
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        im = im.byteswap().newbyteorder()

    # 7. Convert image to rgb8 order based on encoding message:
    
    if img_msg.encoding == 'rgb8':
        pass
    elif img_msg.encoding == "bgr8":
        im = im[..., ::-1]
    
    elif img_msg.encoding == 'rgba8':
        im = im[..., :3][..., ::-1]
    elif img_msg.encoding == 'bgra8':
        im = im[..., :3][..., ::-1]    
    elif n_channels == 1:
        pass
    else:
        raise NotImplementedError(f"Conversion from {img_msg.encoding} to rgb8 not implemented")
    return im

class SynchSubscriberNode(Node):
    class CircularBuffer:
        def __init__(self, max_size):
            assert max_size > 1
            self.max_size = max_size
            self.idx = 0
            self.locks = [threading.Lock() for _ in range(max_size)]
            self.data_buffer = [None for _ in range(max_size)]
        def put(self, item):
            with self.locks[self.idx]:
                self.data_buffer[self.idx] = item
            self.idx = (self.idx + 1) % self.max_size
        def get_latest(self):
            latest_idx = (self.idx - 1) % self.max_size
            return self.data_buffer[latest_idx], self.locks[latest_idx]
            
    def __init__(self, topics_config: dict[str, TopicConfig], config: SynchSubscriberNodeConfig):
        super().__init__('sync_subscriber')
        self.topics_config = topics_config

        self.is_data_collected = False
        self.data_condition = Condition()
        # 20 should be enough time for the datawriter to peek, and save images?
        self.data_dict_buffer = SynchSubscriberNode.CircularBuffer(max_size=20) 
        self.sync_subscribers = self.__get_synchronous_subscribers()

        # Create all required subscribers 
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [x[1] for x in list(self.sync_subscribers.values())], 
            queue_size=10, slop=config.slop_sec)
        callback_fn = self.__create_dynamic_callback(topic_keys = list(self.sync_subscribers.keys()))
        self.ts.registerCallback(callback_fn)

        self.spin_thread = threading.Thread(target=lambda : rclpy.spin(self))
        self.spin_thread.start()

        # Wait for one data
        self.get_logger().info("Waiting for one synchronized data ... ")
        self.wait_for_one_data()
        self.get_logger().info("Finished Waiting for one synchronized data ... ")

        data = self.get_latest_data_dict()
        self.topic_key_to_image_sizes_dict = {}
        for topic_key in data.keys():
            if topic_key == "timestamp":
                continue
            topic_type = self.topics_config[topic_key].type
            if topic_type == "sensor_msgs/Image":
                image = data[topic_key]
                height, width = image.shape[:2]
                self.topic_key_to_image_sizes_dict[topic_key] = (width, height)

    def join(self):
        self.spin_thread.join() 
         
    # Do not do this unless you are sure the consumer will consume data in time.
    def peek_latest_data_dict(self):
        with self.data_condition:
            if not self.is_data_collected:
                return None   
            data_dict, lock = self.data_dict_buffer.get_latest()
            return data_dict, lock

    def get_latest_data_dict(self):
        with self.data_condition:
            if not self.is_data_collected:
                return None                
            data_dict, lock = self.data_dict_buffer.get_latest()
            with lock:
                return copy.deepcopy(data_dict)

    def wait_for_one_data(self):
        with self.data_condition:
            self.is_data_collected = False
        with self.data_condition:
            while not self.is_data_collected:
                self.data_condition.wait()
            
    def __create_dynamic_callback(self, topic_keys):
        def data_collection_callback(*args):
            self.get_logger().debug("Synchronized callback running")
            assert len(args) == len(topic_keys)
            data_dict = {}
            current_time = self.get_clock().now()
            data_dict["timestamp"] = current_time.to_msg()
            for topic_key, msg in zip(topic_keys, args):
                topic_type_str = self.topics_config[topic_key].type
                if topic_type_str == "sensor_msgs/Image":
                    try:
                        data = ros_image_to_numpy(msg)
                    except Exception as e:
                        self.get_logger().error("Error parsing image {}: {}".format(topic_key, e))
                        return              
                else:
                    data = msg
                data_dict[topic_key] = data
            with self.data_condition:
                self.data_dict_buffer.put(data_dict)
                self.is_data_collected = True
                self.data_condition.notify_all()                              
        return data_collection_callback
        
    def __get_synchronous_subscribers(self):
        subscribers = {}
        for topic_key, topic_info in self.topics_config.items():
            topic_name = topic_info.name
            msg_type = topic_info.type
            msg_class = None
            if msg_type == "geometry_msgs/PoseStamped":
                msg_class =  PoseStamped
            elif msg_type == "sensor_msgs/JointState":
                msg_class =  JointState
            elif msg_type == "geometry_msgs/Twist":
                msg_class = Twist
            elif msg_type == "geometry_msgs/TwistStamped":
                msg_class = TwistStamped
            elif msg_type == "geometry_msgs/TransformStamped":
                msg_class = TransformStamped              
            elif msg_type == "sensor_msgs/Image":
                msg_class = Image
            elif msg_type == "sensor_msgs/Joy":
                msg_class = Joy                
            elif msg_type == "control_msgs/DynamicJointState":
                if DynamicJointState is None:
                    raise RuntimeError(
                        "Topic '{}' requires control_msgs/DynamicJointState, but the "
                        "control_msgs Python package is unavailable. On ROS 2 Jazzy, "
                        "install it with: sudo apt-get install ros-jazzy-control-msgs"
                        .format(topic_name)
                    )
                msg_class = DynamicJointState
            else:
                self.get_logger().error("No callback defined for topic key: {}".format(topic_key))
                break
            
            subscribers[topic_key] = (msg_type, message_filters.Subscriber(self, msg_class, topic_name), topic_name)
            self.get_logger().info("Subscribed to {} with type {} topic_key {}".format(topic_name, msg_type, topic_key))
        return subscribers

class DataWriter:     
    class EpisodeFileWriterComponent:
        def __init__(self, file_path, parser_fn):
            self.file_path = file_path
            self.parser_fn = parser_fn
            # If the file exists from a previous run, you might want to clear it.
            open(self.file_path, 'w').close()

        def save(self, data):
            data_line = self.parser_fn(data)
            with open(self.file_path, 'a') as f:
                f.write(data_line + "\n")
        def finish(self):
            pass   
    class EpisodeFolderImageWriterComponent:
        def __init__(self, folder_path, parser_fn, image_size, fps=30):
            self.folder_path = folder_path
            self.parser_fn = parser_fn
            os.makedirs(folder_path, exist_ok=True)

            video_filename = os.path.join(folder_path, "video.mp4")
            # OpenCV expects frame_size=(width, height). Normalize common shape formats.
            if len(image_size) == 3:
                frame_size = (image_size[1], image_size[0])  # (H, W, C) -> (W, H)
            elif len(image_size) == 2:
                frame_size = image_size
            else:
                raise ValueError(f"Unsupported image_size format: {image_size}")
            codec = cv2.VideoWriter_fourcc(*'mp4v')  # use 'XVID' for .avi, or 'avc1'/'mp4v' for .mp4
            self.video_writer = cv2.VideoWriter(video_filename, codec, fps, frame_size)

            self.filenames_and_debug_images = []
            self.iteration_count = 0

        def save(self, image):
            image = self.parser_fn(image)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #  ----- For debugging ----- 
            image_filename = os.path.join(self.folder_path, f"{self.iteration_count}.jpg")
            resized_image_bgr = image_bgr
            resize_ratio = 2
            resized_image_bgr = cv2.resize(image_bgr, (image_bgr.shape[1] // resize_ratio, 
                                                image_bgr.shape[0] // resize_ratio))   
            if self.iteration_count == 0:
                self.filenames_and_debug_images = [(image_filename, resized_image_bgr)]
            else:
                self.filenames_and_debug_images.append((image_filename, resized_image_bgr))             
            # ----------------------------
            self.video_writer.write(image_bgr)
            self.iteration_count += 1

        def finish(self):
            cv2.imwrite(*self.filenames_and_debug_images[0]) # First image
            len_filenames_and_debug_image = len(self.filenames_and_debug_images)
            if len_filenames_and_debug_image > 1:
                cv2.imwrite(*self.filenames_and_debug_images[-1]) # Last image
            if len_filenames_and_debug_image > 2:
                half = len_filenames_and_debug_image // 2 
                cv2.imwrite(*self.filenames_and_debug_images[half]) # Middle image
            self.video_writer.release() # Writes the video

    def __init__(self, config: DataWriterConfig, topic_key_to_image_sizes_dict, topics: dict[str, TopicConfig], fps):
        self.config = config

        # Constants
        self.topic_key_to_image_sizes_dict = topic_key_to_image_sizes_dict
        self.recorded_topics_config = {k: v for k, v in topics.items() if v.recorder is not None} # filter recorded topics
        self.recorded_topic_key_to_files = {}
        self.recorded_topic_key_to_folders = {}
        for topic_key, topic_info in self.recorded_topics_config.items():
            recorder_cfg = topic_info.recorder
            if recorder_cfg.file is not None:
                self.recorded_topic_key_to_files[topic_key] = recorder_cfg.file
            if recorder_cfg.folder is not None:
                self.recorded_topic_key_to_folders[topic_key] = recorder_cfg.folder

        self.fps = fps
        # Per episode variables
        self.iteration_count = 0
        self.topic_key_to_current_episode_writer_components = {}
        self.current_timestamp_file = None

        def parse_header(header):
            t_sec = header.stamp.sec
            t_nsec = header.stamp.nanosec
            return t_sec, t_nsec 
        def pose_stamped_to_data(msg):
            t_sec, t_nsec = parse_header(msg.header)
            pos = msg.pose.position
            quat = msg.pose.orientation
            return f"{t_sec},{t_nsec}," \
                    f"{pos.x:.5f},{pos.y:.5f},{pos.z:.5f}," \
                    f"{quat.x:.5f},{quat.y:.5f},{quat.z:.5f},{quat.w:.5f}"
        def transform_stamped_to_data(msg):
            t_sec, t_nsec = parse_header(msg.header)
            pos = msg.transform.translation
            quat = msg.transform.rotation
            return f"{t_sec},{t_nsec}," \
                    f"{pos.x:.5f},{pos.y:.5f},{pos.z:.5f}," \
                    f"{quat.x:.5f},{quat.y:.5f},{quat.z:.5f},{quat.w:.5f}," \
                    f"{msg.header.frame_id},{msg.child_frame_id}"
        def joint_state_to_data(msg):
            t_sec, t_nsec = parse_header(msg.header)
            joints_str = ",".join([f"{p:.5f}" for p in msg.position])
            return f"{t_sec},{t_nsec},{joints_str}"
        def joy_to_data(msg):
            t_sec, t_nsec = parse_header(msg.header)
            return f"{t_sec},{t_nsec},{msg.buttons[0]}"
        def twist_to_data(msg):
            t_sec, t_nsec = parse_header(msg.header)
            return f"{t_sec},{t_nsec}" \
                    f"{msg.linear.x:.5f}," \
                    f"{msg.linear.y:.5f}," \
                    f"{msg.linear.z:.5f}," \
                    f"{msg.angular.x:.5f}," \
                    f"{msg.angular.y:.5f}," \
                    f"{msg.angular.z:.5f}" 
        def dynamic_joint_state_to_data(msg):
            t_sec, t_nsec = parse_header(msg.header)
            if len(msg.interface_values) > 0:
                iface = msg.interface_values[0]  # First tracker only
                iface_map = dict(zip(iface.interface_names, iface.values))
                # Define your desired, consistent order
                ordered_keys  = [
                    "position.x",
                    "position.y",
                    "position.z",
                    "orientation.x",
                    "orientation.y",
                    "orientation.z",
                    "orientation.w"
                ]
                # Extract in that order (if missing, insert NaN)
                ordered_values = [iface_map.get(k, float('nan')) for k in ordered_keys]
                values_str = ",".join([f"{v:.5f}" for v in ordered_values])
                return f"{t_sec},{t_nsec},{values_str}"

        def image_to_data(numpy):
            return numpy # passthrough
    
        self.topic_type_str_to_data_parser = {
            "geometry_msgs/PoseStamped": pose_stamped_to_data,
            "geometry_msgs/TransformStamped": transform_stamped_to_data,
            "sensor_msgs/JointState": joint_state_to_data,
            "sensor_msgs/Joy": joy_to_data,
            "geometry_msgs/TwistStamped": twist_to_data,
            "sensor_msgs/Image": image_to_data,
            "control_msgs/DynamicJointState": dynamic_joint_state_to_data
        }

    def save_data_iteration(self, data_dict, lock):
        time_msg = data_dict["timestamp"]
        sec = time_msg.sec
        nsec = time_msg.nanosec
        dt = datetime.fromtimestamp(sec + nsec / 1e9, tz=timezone.utc)
        timestamp_line = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # '2026-02-19 12:34:56.123'
        timestamp_line += "\n"
        with open(self.current_timestamp_file, 'a') as tf:
            tf.write(timestamp_line)

        with lock:
            for topic_key, episode_writer_component in self.topic_key_to_current_episode_writer_components.items():
                data = data_dict[topic_key]
                episode_writer_component.save(data)
                    
            print("Saved iteration: {} at time: {} sec, {} nanosec".format(self.iteration_count, sec, nsec))    
            self.iteration_count += 1
        
    def __create_episode_folders(self, episode_num):
        """Create the folder structure for the current episode."""
        assert type(episode_num) == int

        episode_folder = os.path.join(self.config.saving_folder, f"e{episode_num}")
        os.makedirs(episode_folder, exist_ok=True)

        for topic_key in self.recorded_topic_key_to_files:
            file_path = os.path.join(episode_folder, self.recorded_topic_key_to_files[topic_key])
            episode_file_path = os.path.join(episode_folder, file_path)
            topic_type = self.recorded_topics_config[topic_key].type
            self.topic_key_to_current_episode_writer_components[topic_key] = \
                DataWriter.EpisodeFileWriterComponent(episode_file_path, 
                                                      self.topic_type_str_to_data_parser[topic_type])
        for topic_key in self.recorded_topic_key_to_folders:
            folder = self.recorded_topic_key_to_folders[topic_key]
            full_episode_folder_path = os.path.join(episode_folder, folder)
            topic_type = self.recorded_topics_config[topic_key].type
            if topic_type == "sensor_msgs/Image": # There is only image now
                self.topic_key_to_current_episode_writer_components[topic_key] = \
                    DataWriter.EpisodeFolderImageWriterComponent(full_episode_folder_path, 
                                                                self.topic_type_str_to_data_parser[topic_type], 
                                                                self.topic_key_to_image_sizes_dict[topic_key],
                                                                self.fps)

        # Create the timestamp file.
        self.current_timestamp_file = os.path.join(episode_folder, self.config.timestamp_file)
        open(self.current_timestamp_file, 'w').close()

    def create_episode(self, episode_num):
        self.__create_episode_folders(episode_num)
        self.iteration_count = 0

    def finish_episode(self):
        for _, episode_writer_component in self.topic_key_to_current_episode_writer_components.items():
            episode_writer_component.finish()
        self.topic_key_to_current_episode_writer_components = {}
        self.current_timestamp_file = None

class Recorder:
    def __init__(self, config: RecorderConfig, sync_subscriber: SynchSubscriberNode):
        self.config = config
        self.sync_subscriber = sync_subscriber            
        self.data_writer = DataWriter(self.config.data_writer,
                                      self.sync_subscriber.topic_key_to_image_sizes_dict,
                                      self.sync_subscriber.topics_config,
                                      self.config.frequency)

        assert self.config.episode >= 0
        self.current_episode_num = self.config.episode

        self.started_episode = False
        self.key_listener_thread = threading.Thread(target=self.key_listener, daemon=True)
        self.key_listener_thread.start()

        self.saving_thread = None
        self.is_saving_lock = threading.Lock()
        self.is_saving = False
        self.saving_thread = threading.Thread(target=self.data_saving_thread, daemon=True)
        self.saving_thread.start()

    def key_listener(self):
        """Listens for key input in a separate thread. On pressing 'n', a new episode is started."""
        while rclpy.ok():
            key = input("Press 'q' to stop current episode collection and 's' to start a new demo episode, and then press enter")
            if key.lower() == 'q':
                if self.started_episode: 
                    self.sync_subscriber.get_logger().info("Key 'q' pressed: stopping collection and saving current episode.")
                    self.stop_current_episode()
                    self.started_episode = False
                else: 
                    self.sync_subscriber.get_logger().error("No episode was started, press 's' first and then enter")

            if key.lower() == 's':
                if self.started_episode: 
                    self.sync_subscriber.get_logger().error("Episode was already started, stop it by pressing 'q' first and then enter")
                else: 
                    self.started_episode = True
                    self.sync_subscriber.get_logger().info("Key 's' pressed: Starting new episode")
                    self.start_new_episode()

    def start_new_episode(self):
        self.sync_subscriber.get_logger().info("Switching to episode: {}".format(self.current_episode_num))
        self.sync_subscriber.wait_for_one_data()
        self.data_writer.create_episode(self.current_episode_num)
        with self.is_saving_lock:
            self.is_saving = True

    def stop_current_episode(self):
        with self.is_saving_lock:
            self.is_saving = False
            self.data_writer.finish_episode()
            self.current_episode_num += 1

    def data_saving_thread(self):
        rate = self.sync_subscriber.create_rate(self.config.frequency)
        while rclpy.ok():
            with self.is_saving_lock:
                if self.is_saving:
                    try:
                        data, lock = self.sync_subscriber.peek_latest_data_dict()
                        self.data_writer.save_data_iteration(data, lock)
                        self.sync_subscriber.wait_for_one_data()
                    except Exception as e:
                        self.sync_subscriber.get_logger().error("Error in saving loop: {}".format(e))
                        break
            rate.sleep()

    def join(self):
        self.saving_thread.join()
        self.key_listener_thread.join()
