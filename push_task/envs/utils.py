import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import os
def plot_info_on_frame(pil_image, info, font_size=30):
    # TODO: this is a hard-coded path
    location = os.path.dirname(os.path.realpath(__file__))

    font = ImageFont.truetype(os.path.join(location, "arial.ttf"), font_size)
    draw = ImageDraw.Draw(pil_image)

    x = font_size  # X position of the text
    y = font_size  # Beginning of the y position of the text
    
    i = 0
    for k in info:
        # TODO: This is pretty ugly
        if not any([text in k for text in ["TimeLimit", "render_array", "TimeLimit.truncated", "jnt_states"]]):
            reward_text = f"{k}:{info[k]}"
            # Plot the text from bottom to top
            text_position = (x, y + 30*(i+1))
            draw.text(text_position, reward_text, fill=(255, 255, 255), font=font)
        i += 1
    return np.array(pil_image)

def select_representative_numbers(numbers, k=5):
    numbers = np.array(numbers).reshape(-1, 1)  # Reshape for clustering
    
    if len(numbers) <= k:
        return list(numbers.flatten())  # Return all if less than or equal to k numbers
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(numbers)
    
    # Find the closest number to each cluster center
    centers = kmeans.cluster_centers_.flatten()
    representatives = []
    
    for center in centers:
        closest = min(numbers.flatten(), key=lambda x: abs(x - center))
        if closest not in representatives:
            representatives.append(closest)
    
    return sorted(representatives)




def linear_decay_sample(size, decay_factor):
    """
    1 is uniform
    0 is weighted
    """
    if decay_factor > 1:
        decay_factor = 1
    if decay_factor < 0:
        decary_factor = 0
    # Create an array of weights that favors later indices
    weights = np.arange(1, size + 1)

    # Normalize weights to create a starting probability distribution
    probabilities = weights / weights.sum()

    # Create a uniform distribution
    uniform_probabilities = np.ones(size) / size

    # Linearly interpolate between the biased and uniform probabilities
    mixed_probabilities = (1 - decay_factor) * probabilities + decay_factor * uniform_probabilities

    # Normalize the mixed probabilities
    mixed_probabilities /= mixed_probabilities.sum()

    # Sample an index based on the mixed probabilities
    sampled_index = np.random.choice(size, p=mixed_probabilities)

    return sampled_index

def linear_gaussian_sample(size, bias):
    """
    1 is later in the trajectory
    -1 is earlier
    """
    if bias < -1:
        bias = -1
    if bias > 1:
        bias = 1

    mean = size // 2 + (bias * (size // 2))
    if mean > size - 1:
        mean = size - 1
    std_dev = size / 8
    indices = np.arange(size)

    weights = np.exp(-0.5 * ((indices - mean) / std_dev) ** 2)
    weights = weights / np.sum(weights)

    sampled_index = np.random.choice(size, p=weights)
    return sampled_index




def seq_obs(obs_list, sequence_len):
    """
    Takes the last sequence_len observations and concatenates them.
    """
    obs_list_len = len(obs_list)
    start_index = obs_list_len - sequence_len
    new_obs_list = obs_list[start_index:]
    key_list = obs_list[0].keys()
    final_obs = {}
    for key in key_list:
        final_obs[key] = []
    for obs in new_obs_list:
        for key in key_list:
            final_obs[key].append(obs[key])
    for f_key in final_obs.keys():
        final_obs[f_key] = np.array(final_obs[f_key])
    return final_obs

# def get_history_obs(observation_history):
#     handle_0_tried, handle_1_tried, handle_2_tried = 0, 0, 0
#     for obs in observation_history:
#         if obs["handle_displacement_0"] > 0.07:
#             handle_0_tried = 1
#         if obs["handle_displacement_1"] > 0.07:
#             handle_1_tried = 1
#         if obs["handle_displacement_2"] > 0.07:
#             handle_2_tried = 1
#     return handle_0_tried, handle_1_tried, handle_2_tried

def get_history_obs(observation_history):
    target_done_0, target_done_1, target_done_2 = 0, 0, 0
    for obs in observation_history:
        if np.linalg.norm(obs["target_0"] - obs["red_box"]) < 0.06:
            target_done_0 = 1
        if np.linalg.norm(obs["target_1"] - obs["red_box"]) < 0.06:
            target_done_1 = 1
        if np.linalg.norm(obs["target_2"] - obs["red_box"]) < 0.06:
            target_done_2 = 1
    return target_done_0, target_done_1, target_done_2

def get_obs_split(obs, env=None):
    teacher_obs_keys = None
    student_obs_keys = None
    if env is None:
        teacher_obs_keys = [
            "jnt_states", 
            "target_0", 
            "target_1",
            "target_2", 
            "red_box",
            "red_target",
        ]
        student_obs_keys = [
            "jnt_states", 
            "target_0", 
            "target_1",
            "target_2", 
            "red_box",
            "red_history",
        ]
    else:
        teacher_obs_keys = [k for k in env.observation_space if k not in env.unwrapped.teacher_filter_list()]
        student_obs_keys = [k for k in env.observation_space if k not in env.unwrapped.student_filter_list()]
    teacher_obs = {key: obs[key] for key in teacher_obs_keys if key in obs}
    student_obs = {key: obs[key] for key in student_obs_keys if key in obs}
    return teacher_obs, student_obs

class Trajectory:
    def __init__(self, seq_length):
        """
        traj (Dict)
            obs (Dict): keys of each observation has shape (traj length, feature size)
            actions (List): list of shape (traj length, action dim)
        """
        self.traj = {"obs": {}, "actions": []}
        self.size = 0 # traj_length
        self.seq_length = seq_length
        
    def add(self, obs, action):
        if len(self.traj["obs"].keys()) == 0:
            self.traj["obs"].update(obs)
        else:
            for key in self.traj["obs"].keys():
                k = np.concatenate((self.traj["obs"][key], obs[key]))
                self.traj["obs"][key] = k
            self.traj["actions"].extend(action)

        self.traj["actions"].append(action)
        self.size += 1
        return self.traj


class Buffer:
    def __init__(self):
        """
        Bank (List of Dict): Each list contains a saved reset state
            Keys:
                data stretch joint information
                handle positions
                goal drawer (0, 1, 2)
        """
        self.buffer = []
        

    def add_from_sim(self, data, goal):
        state_dict = {}
        for data_joint in self.data_joints:
            state_dict[data_joint] = data.joint(data_joint).qpos
        state_dict["goal"] = goal
        self.bank.append(state_dict)
        return self.bank
    
    def add(self, entry):
        self.bank.append(entry)
        return self.bank
    
    def extend(self, bank):
        self.bank.append(deepcopy(bank.bank))
        return self.bank

    def sample(self, ind=None, pop=True):
        if len(self.bank) == 0:
            print("Reset bank empty")
            return None
        if ind is None:
            ind = np.random.randint(0, len(self.bank))
        if pop:
            reset_state = self.bank.pop(ind)
        else:
            reset_state = self.bank[ind]
        return reset_state

    def __len__(self):
        return len(self.bank)

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
    
    def load(self, file_path):
        with open(file_path, 'rb') as file:
            self = pickle.load(file)
