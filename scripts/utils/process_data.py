import sys
import os
import struct
import numpy as np
import ffmpeg
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def read_floats_from_files_txt(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return lines


def read_floats_from_file_bin(file_path):
    floats = []
    with open(file_path, 'rb') as f:
        while chunk := f.read(4):
            floats.append(struct.unpack('f', chunk)[0])
    return floats


def read_floats_from_folder_bin(folder_path):
    all_floats = []
    file_size = len(os.listdir(folder_path))
    for filenum in range(file_size):
        filename = f'{filenum}.bin'
        file_path = os.path.join(folder_path, filename)
        all_floats.extend([read_floats_from_file_bin(file_path)])
    return np.array(all_floats)


def convert_nsec_to_sec(data):
    return (data*1e-9).astype(np.float32)


def get_data_difference(data):
    data_shift = np.zeros_like(data)
    rows, cols = data.shape
    data_shift[:rows-1] = np.copy(data[1:rows])
    data_shift[rows-1] = data[rows-1]

    print(np.sum(data_shift == data))
    data_diff = data_shift - data 
    return data_diff



def get_image_in_order_unity(image_folder):
    """
    For image collected with Unity script, image in format of "image_xxxx.png"
    """
    image_name_lst_1 = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    image_name_lst = []

    for i in range(len(image_name_lst_1)):
        image_name = "image_" + f"{i:04}"+".png"
        if image_name not in image_name_lst_1:
            print(image_name + " not in the folder!!!")
            pass
        else:
            image_name_lst.append(image_name)
    return image_name_lst


def get_image_in_order_ros(image_folder):
    
    """
    For image collected with ROS script, image in format of "x.png",
    """
    
    image_name_lst_1 = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    image_name_lst = []

    for i in range(len(image_name_lst_1)):
        image_name = str(i)+".png"
        if image_name not in image_name_lst_1:
            print(image_name + " not in the folder!!!")
            pass
        else:
            image_name_lst.append(image_name)
    return image_name_lst


def load_image_from_episode_to_np(path, episode_folder, image_folder_name="cam0/", show_image=False, image_sorted_func=get_image_in_order_ros):
    
    image_folder = path + episode_folder + image_folder_name
    image_values = []

    image_name_lst = image_sorted_func(image_folder)
    # images_str = read_from_files(FOLDER_PATH + episode_name + )
    # image_name_sorted = (sorted(image_name_lst_1))
    
    for img_path in image_name_lst:
        image = cv2.imread(image_folder + img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320,240))
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cropped_image = image[:,40:320-40,:]
        resized_image = cv2.resize(cropped_image, (96,96))
        image_values.append(np.array(resized_image))
        if show_image:
            print("Process Image: ", image_folder + img_path)
            cv2.imshow("RGB resized image: " + str(image.shape), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            cv2.imshow("RGB cropped image", cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
    return np.array(image_values)  


def load_eef_position_from_episode_txt(folder_path, episode_name):
    eef_file_path = folder_path + episode_name + "end_effector_positions.txt"
    eef_positions = read_floats_from_files_txt(eef_file_path)
    state_data = np.array([np.fromstring(line, sep=',') for line in eef_positions])
    return state_data[:,[0,2]]


def get_action_from_eef_position(eef_positions):
    action = np.zeros_like(eef_positions)
    action[:-1] = eef_positions[1:]
    action[-1] = eef_positions[-1]
    return action


def reencode_video(input_file, output_file):
    (
        ffmpeg
        .input(input_file)
        .output(output_file, vcodec='libx264', crf=23, acodec='copy')
        .run()
    )



if __name__ == '__main__':
    
    folder_path = "./Data/real-dvrk-push-block/"
    episode_name = "e0/"
    

    # eef_position = read_floats_from_folder(folder_path+POSITION_FOLDER)
    # robot_joint = read_floats_from_folder(folder_path+JOINT_FOLDER)
    # timestamp = read_floats_from_folder(folder_path+TIMESTAMP_FOLDER)
    # mili_time = convert_nsec_to_milisec(timestamp)
    # teleop = read_floats_from_folder(folder_path+VELOCITY_FOLDER)

    img = load_image_from_episode_to_np(folder_path, "e3/", "cam0/", True)
    print(img.shape)
    # print(t1, t2)
    # print(mili_time)
    # print(robot_joint)
    # print(eef_position)
    # # print(teleop)
    # # print(len(mili_time))
    # print(eef_position[0], eef_position.shape)
    # print(robot_joint[0], robot_joint.shape)
    # print(timestamp, timestamp.shape)
    # print(timestamp[0], timestamp)
    # for i in range(len(timestamp)):
    #     print(timestamp[i,0])
    #     print(mili_time[i,0])
