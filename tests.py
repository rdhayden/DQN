import argparse
import torch
import torch.optim as optim
import collections
from lib.conv_net import ConvNet
from lib.experience_buffer import ExperienceBuffer
from lib.extended_game import ExtendedGame
from lib.agent import Agent
from lib.utils import calc_loss
from tqdm import trange
import numpy as np
import torch.nn as nn
from skimage import io
from matplotlib import pyplot as plt

if __name__ == "__main__":

    print("Test ExtendedGame: sample_actions returns int between 1 and 3")
    device = torch.device("cuda")
    game = ExtendedGame('scenarios/basic.cfg')
    for i in range(10):
        action = game.sample_actions()
        assert type(action) is int, "fail: action is not an int"
        print("pass: action is an int")
        assert 0 <= action < 3, "fail: action is not between 1 and 3"
        print("pass: action is between 1 and 3")

    print("Visually test ExtendedGame: start game, move right 12 times, fire 12 times, move left 12 times")
    device = torch.device("cuda")
    game = ExtendedGame('scenarios/basic.cfg')
    game.set_window_visible(True)
    game.init()
    number_of_times = 4
    
    input("Press any key to make left action {} times, each action repeats for 4 frames".format(number_of_times))
    reward = 0
    for i in range(number_of_times):
        reward += game.make_action_int(0,4)
    print("reward {}".format(reward))

    input("Press any key to make fire action {} times, each action repeats for 4 frames".format(number_of_times))
    reward = 0
    for i in range(number_of_times):
        reward += game.make_action_int(2,4)
    print("reward {}".format(reward))

    input("Press any key to make right action {} times, each action repeats for 4 frames".format(number_of_times))
    reward = 0
    for i in range(number_of_times):
        reward += game.make_action_int(1,4)
    print("reward {}".format(reward))

    input("Press any key to continue")  
    
    capacity = 10
    nsteps = 2
    gamma = 0.99
    experience_buffer = ExperienceBuffer(capacity, nsteps, gamma)
    num_frames_stacked = 4
    width = 54
    height = 54
    repeat_for_frames = 4
    net_shape = (num_frames_stacked, width, height)
    train_net = ConvNet(net_shape, game.n_actions).to(device)
    target_net = ConvNet(net_shape, game.n_actions).to(device)
    agent = Agent(game, train_net, experience_buffer, width, height, num_frames_stacked, repeat_for_frames)

    print("Test Agent: state should be shaped num frames {} width {} height {}".format(num_frames_stacked, width, height))
    assert agent.state.shape == (num_frames_stacked, width, height), "fail: agent state should be shaped {} {} {}".format(num_frames_stacked, width, height)
    print("pass: agent state shaped {} {} {}".format(num_frames_stacked, width, height))

    print("Test Agent: initially all frames in the stack except the top frame should be zero")
    assert sum(agent.state[0:num_frames_stacked-1].flatten()) == 0, "fail: first 3 frames in the stack should be zero when initialised"
    print("pass: first 3 frames in the stack are zero when initialised")
    assert sum(agent.state[num_frames_stacked-1].flatten()) != 0, "fail: the last frame in the stack should not be zero when initialised"
    print("pass: the last frame in the stack is not zero when initialised")

    print("Test Agent: reset after stack is full should result in all frames in the stack except the top frame being zero")
    agent.state = np.random.rand(num_frames_stacked, width, height)
    agent._reset()
    assert sum(agent.state[0:num_frames_stacked-1].flatten()) == 0, "fail: first 3 frames in the stack should be zero"
    print("pass: first 3 frames in the stack are zero")
    assert sum(agent.state[num_frames_stacked-1].flatten()) != 0, "fail: the last frame in the stack should not be zero"
    print("pass: the last frame in the stack should not be zero")

    print("Visualy test Agent, after reset top frame should be a reduced image of size {} by {}".format(width, height))
    agent._reset()
    print(agent.state[-1].shape)
    print(agent.state[-1])
    img = agent.state[-1].reshape(width, height)
    io.imshow(img)
    plt.show()
    input("Press any key to continue")
    plt.close()

    print("resetting experience buffer, then observe state and invoke agent play method 100 times ...")
    experience_buffer.buffer.clear()
    agent._reset()
    for i in range(100):
        play_random = True
        agent.play_step(play_random)
    print("number of experiences in the buffer: {}".format(len(experience_buffer)))
    
    input("Press any key to see a random initial state image from the experience buffer")
    idx = np.random.choice(len(experience_buffer))
    random_experience = experience_buffer[idx]
    io.imshow(random_experience.state[num_frames_stacked-1])
    plt.show()
    input("Press any key to see the next state image for the same experience")
    plt.close()
    io.imshow(random_experience.new_state[num_frames_stacked-1])
    plt.show()
    input("Press any key to continue")

    print("Test Agent: after some iterations check if the second frame in the stack is not zero")
    agent._reset()
    second_frame_non_zero = False
    for i in range(100):
        play_random = True
        agent.play_step(play_random)
        if sum(agent.state[1].flatten()) != 0:
            second_frame_non_zero = True
            break
    assert second_frame_non_zero, "fail: 2nd frame never non-zero after 100 steps"
    print("pass: 2nd frame non-zero at least some of the time")

    print("Network Agent: test network outputs with framestack input")
    state_a = np.array([agent.state], copy=False)
    state_tensor = torch.tensor(state_a, dtype=torch.float).to(device)
    q_vals_tensor = train_net(state_tensor)
    print("Q value predictions:")
    print(q_vals_tensor)
    print()
    _, index = torch.max(q_vals_tensor, 1) # index of action predicted with the highest value
    action = index.cpu().data.numpy()[0]
    print("Predicted action: {}".format(action))