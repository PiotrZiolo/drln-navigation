import numpy as np
from collections import namedtuple, deque

from ddqn_agent import DDQNAgent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
ALPHA = 1               # value from 0 to 1, where 0 - uniform distribution, 1 - distribution given by p
BETA = 1                # importance-sampling weight power

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PrioritizedDDQNAgent(DDQNAgent):
    """Prioritized double DQN agent. Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size,
                 buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 gamma=GAMMA, tau=TAU, lr=LR, update_every=UPDATE_EVERY,
                 alpha=ALPHA, beta=BETA, seed=0):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            buffer_size (int): replay buffer size
            batch_size (int): batch size
            batch_size (int): batch size
            gamma (int): time discount
            tau (int): target network soft update rate
            lr (int): learning rate
            update_every (int): learn every this number of steps
            alpha (float): value from 0 to 1, where 0 - uniform distribution, 1 - distribution given by p
            beta (float): importance-sampling weight power
            seed (int): random seed
        """
        super().__init__(state_size, action_size, buffer_size, batch_size,
                         gamma, tau, lr, update_every, seed)

        self.alpha = alpha
        self.beta = beta

        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, self.buffer_size, self.batch_size,
                                              self.alpha, self.beta, seed)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done, idx, w) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, idx, weights = experiences

        # Prepare output

        outputs = self.qnetwork_local(states).gather(1, actions)

        # Prepare target

        Q_targets = self.qnetwork_target(next_states).detach().max(1, keepdim=True)[0]
        targets = rewards + gamma * Q_targets * (1 - dones)

        # Update probabilities

        delta = (outputs - targets).abs()
        for i in range(len(idx)):
            self.memory.update_p(idx[i], delta[i])

        # Add importance-sampling weighing

        loss = ((outputs - targets).pow(2) * weights).mean()

        # Run training step

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)


class PrioritizedReplayBuffer:
    """Fixed-size prioritized buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, alpha, beta, seed):
        """Initialize a PrioritizedReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float): value from 0 to 1, where 0 - uniform distribution, 1 - distribution given by p
            beta (float): importance-sampling weight power
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "p"])
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory. Use previously maximal probability for the new transition."""
        max_p = np.max(np.array([e.p for e in self.memory if e is not None])) if len(self.memory) > 0 else 1.0
        e = self.experience(state, action, reward, next_state, done, max_p)
        self.memory.append(e)

    def update_p(self, idx, p):
        """Update probability of a given transition"""
        self.memory[idx]._replace(p=p + 1 / len(self.memory))
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        prob = np.array([e.p for e in self.memory if e is not None])
        prob = np.power(prob, self.alpha) / np.sum(np.power(prob, self.alpha))
        indices = self.rng.choice(range(len(self.memory)), size=self.batch_size, p=prob)
        experiences = [self.memory[index] for index in indices]
        prob = prob[indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        idx = list(range(len(experiences)))
        weights = 1 / np.power(len(self.memory) * prob, self.beta)
        weights = weights / np.max(weights)
        weights = torch.from_numpy(np.vstack(weights.tolist())).float().to(device)

        return states, actions, rewards, next_states, dones, idx, weights

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
