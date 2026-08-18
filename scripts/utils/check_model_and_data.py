# Loads ACT Policy trained using the diffusion_policy pipeline

import yaml
import torch
# from diffusion_policy.policy.vae_act_policy import VAE_ACT_ImagePolicy
import zarr
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time


def load_policy_from_yaml(yaml_path, checkpoint_path):
    """Loads a policy from a YAML config and checkpoint."""
    
    # Load YAML config
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract policy configuration
    policy_config = config['policy']
    shape_meta = config['shape_meta']
    
    # Initialize policy
    policy = VAE_ACT_ImagePolicy(
        shape_meta=shape_meta,
        horizon=policy_config['horizon'],
        n_action_steps=policy_config['n_action_steps'],
        n_obs_steps=policy_config['n_obs_steps'],
        crop_shape=tuple(policy_config['crop_shape']),
        obs_encoder_group_norm=policy_config['obs_encoder_group_norm'],
        eval_fixed_crop=policy_config['eval_fixed_crop']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(checkpoint['state_dicts']['model'])
    
    # Set to evaluation mode
    policy.eval()
    
    return policy

def load_normalizer_stats(input_stats):
    """ Extracts normalizer stats from a ParameterDict and converts them to a usable dictionary. """
    normalizer_stats = {}

    for key, param_dict in input_stats.items():
        normalizer_stats[key] = {
            'min': param_dict['min'].detach().cpu(),  
            'max': param_dict['max'].detach().cpu(),  
            'mean': param_dict['mean'].detach().cpu(),  
            'std': param_dict['std'].detach().cpu()
        }
    
    return normalizer_stats


def manual_normalize(input_data, input_stats, mode='limits'):
    """ 
    Normalizes the input data manually based on 'limits' or 'gaussian' mode.

    Args:
        input_data (dict[str, torch.Tensor]): Dictionary of tensors (e.g. 'image', 'agent_pos').
        input_stats (dict[str, dict[str, torch.Tensor]]): Extracted normalization statistics.
        mode (str): Normalization mode ('limits' or 'gaussian').

    Returns:
        dict[str, torch.Tensor]: Normalized data.
    """
    normalized_data = {}

    for key, data in input_data.items():
        if key not in input_stats:
            raise ValueError(f"Missing normalization stats for key: {key}")

        min_val = input_stats[key]['min']
        max_val = input_stats[key]['max']
        mean_val = input_stats[key]['mean']
        std_val = input_stats[key]['std']

        if mode == 'limits':
            # Normalize using min-max scaling: (x - min) / (max - min) scaled to [-1,1]
            scale = (max_val - min_val).clamp(min=1e-6)  # Avoid division by zero
            normalized_data[key] = 2 * ((data - min_val) / scale) - 1  # Scale to [-1,1]

        elif mode == 'gaussian':
            # Normalize using standardization (zero mean, unit variance)
            std_val = std_val.clamp(min=1e-6)  # Avoid division by zero
            normalized_data[key] = (data - mean_val) / std_val

        else:
            raise ValueError(f"Invalid mode for manual normalizer: {mode}. Pick 'limits' or 'gaussian'.")
    
    return normalized_data

class ONNXWrappedPolicy(torch.nn.Module):
    """
    A wrapper around VAE_ACT_ImagePolicy to define a proper ONNX forward method.
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, image, agent_pos):
        """
        ONNX-friendly forward pass.

        Args:
            image (Tensor): Shape (B, 1, 3, 96, 96) - Normalized RGB image
            agent_pos (Tensor): Shape (B, 1, 2) - Normalized agent position

        Returns:
            action (Tensor): Shape (B, n_action_steps, 2) - Unnormalized predicted action
        """
        # Inputs normalized in predict_action: Scale image by 255 
        obs_dict = {
            # "image": image * 255,
            "image": image,
            "agent_pos": agent_pos
        }

        # Run prediction
        output = self.policy.predict_action({"obs": obs_dict}, training=False)

        return output

def export_policy_to_onnx(policy, onnx_filename="vae_act_policy.onnx"):
    """
    Exports the VAE_ACT_ImagePolicy to ONNX format.
    
    Args:
        policy (VAE_ACT_ImagePolicy): Loaded policy.
        onnx_filename (str): Filename for saving ONNX model.
    """
    onnx_policy = ONNXWrappedPolicy(policy)
    
    # Dummy inputs (batch_size=1)
    dummy_image = torch.randn(1, 1, 3, 96, 96, dtype=torch.float32)  # Image input
    dummy_agent_pos = torch.randn(1, 1, 2, dtype=torch.float32)  # Agent position input

    # Export ONNX
    torch.onnx.export(
        onnx_policy, 
        (dummy_image, dummy_agent_pos),
        onnx_filename,
        input_names=["image", "agent_pos"],
        output_names=["action"],
        dynamic_axes={"image": {0: "batch_size"}, "agent_pos": {0: "batch_size"}, "action": {0: "batch_size"}},
        opset_version=11
    )
    
    print(f"✅ ONNX model exported successfully: {onnx_filename}")

if __name__ == "__main__":
    yaml_path = "configs/experiments/unity_act_larger_chunk.yaml"  # Update with actual path
    # For float rgb 0-1: Doesnt work
    # checkpoint_path = "outputs/pushblock_rl_float_rgb/25-03-10_09_30_42/checkpoints/epoch=0250-test_mean_score=0.083.ckpt"  # Update with actual path
    # For chunk size 8 RGB 0-255 images
    # checkpoint_path = "outputs/pushblock_rl_v4/25-03-09_20_22_03/checkpoints/epoch=0200-test_mean_score=0.083.ckpt"
    # For chunk size 30 RGB 0-255 images
    # checkpoint_path = "/home/medcvr/Amey/medcvr-il/diffusion_policy/outputs/unity_push_block_image/epoch_100.ckpt"
    # policy = load_policy_from_yaml(yaml_path, checkpoint_path)
    print("Policy loaded successfully!")
    
    # # Export to ONNX
    # export_policy_to_onnx(policy, onnx_filename="vae_act_policy_largerchunk_255_images.onnx")
    # print("ONNX export completed successfully!")
    # quit()
    # Load norm values (for debug, this is done in the policy class)

    # #### ONNX Evaluation ####
    # # Load the ONNX model
    # onnx_model_path = "vae_act_policy.onnx"  # Update with actual path
    # ort_session = ort.InferenceSession(onnx_model_path)
    # # Get some test data from the testset
    # testset_dir = "/mnt/d/UnityDemos/pb_rl_downsampled_testset"
    # episode = 2
    # ep_dir = f"{testset_dir}/e{episode}"
    # # Test at timestep 0
    # img = f"{ep_dir}/image_0000.png"
    # img = plt.imread(img)
    # # Reorder the image from [96, 96, 3] to [1, 1, 3, 96, 96]
    # img = np.expand_dims(np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0), axis=0)
    # eepos = f"{ep_dir}/end_effector_positions.txt"
    # with open(eepos, "r") as f:
    #     lines = f.readlines()
    # eepos_values = np.array((lines[0].strip().split(",")[0], lines[0].strip().split(",")[2]), dtype=np.float32)
    # # Reshape eepos_values from [2] to [1, 1, 2]
    # eepos_values = np.expand_dims(np.expand_dims(eepos_values, axis=0), axis=0)
    # onnx_inputs = {
    #     'image': img,
    #     'agent_pos': eepos_values
    # }
    # # print(img.shape)
    # # print(eepos_values.shape)
    # # quit()
    # # onnx_inputs = {
    # #     "image": np_obs_dict['image'].numpy(),
    # #     "agent_pos": np_obs_dict['agent_pos'].numpy()
    # # }
    # onnx_outputs = ort_session.run(None, onnx_inputs)
    # print("ONNX Predicted action:")
    # print(onnx_outputs) 
    # quit()

    # input_stats = policy.normalizer.get_input_stats()
    # normalizer_stats = load_normalizer_stats(input_stats)

    # Try a sample from the train set (or validation set)
    # zarr_path = "/home/medcvr/Downloads/pusht/pusht_cchi_v7_replay.zarr"

    zarr_path = "/home/medcvr/Amey/medcvr-il/diffusion_policy/data/replay_buffers/unity-push-block"
    zarr_path = "/home/medcvr/Amey/medcvr-il/diffusion_policy/data/replay_buffers/push_block_unity_two_cam"
    # zarr_path = "/home/medcvr/Amey/medcvr-il/diffusion_policy/data/replay_buffers/real-dvrk-pushT-Image"
    
    zarr_file = zarr.open_group(zarr_path, mode='r')  # Read mode
    end_indices = zarr_file['meta']['episode_ends']

    episode = 1
    n_obs_steps = 2000
    print(n_obs_steps)
    print(end_indices)
    img = None
    eepos = None
    # For debug: Visualize the image
    
    img_list = zarr_file['data']['cam0']
    print(img_list.shape)
    # plt.imshow(zarr_file['data']['img'][end_indices[episode-1]])
    # plt.show()
    # quit()

    for timestep in range(n_obs_steps):
        img = zarr_file['data']['cam0'][timestep]
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(np.amax(img), np.amin(img))
        cv2.imshow("RGB Image shape: " + str(bgr_img.shape), bgr_img)
        cv2.waitKey(1)
        # time.sleep(0.1)

    np_obs_dict = {
        'image': torch.from_numpy(img).permute(0, 3, 1, 2).unsqueeze(0),
        'agent_pos': torch.from_numpy(eepos).unsqueeze(0)
    }
    
    #### Execute eval and Test denormalization ####
    print(np_obs_dict['image'].shape)
    print(np_obs_dict['agent_pos'].shape)
    batch_size = len(np_obs_dict['image'])
    print(batch_size)

    # print(np_obs_dict['image'])
    # quit()
    # np_obs_dict['image'] *= 255
    # pred = policy.predict_action(np_obs_dict)
    # print("Correct action:")
    # for i in range(policy.n_action_steps):
    #     timestep = end_indices[episode-1] + i
    #     print(zarr_file['data']['action'][timestep])

    # print("\n")
    # print("Predicted action:")
    # print(pred['action'])

    # #### ONNX Evaluation ####
    # # Load the ONNX model
    # onnx_model_path = "vae_act_policy.onnx"  # Update with actual path
    # ort_session = ort.InferenceSession(onnx_model_path)
    # onnx_inputs = {
    #     "image": np_obs_dict['image'].numpy(),
    #     "agent_pos": np_obs_dict['agent_pos'].numpy()
    # }
    # onnx_outputs = ort_session.run(None, onnx_inputs)
    # print("ONNX Predicted action:")
    # print(onnx_outputs)