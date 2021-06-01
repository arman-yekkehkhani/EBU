import numpy as np


class ExperienceBuffer:
    """
    A replay buffer that stores (state, action, reward, next_state)
    implemented as a circular buffer
    :param buffer_size: capacity of buffer
    """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.states = [None] * buffer_size
        self.actions = np.zeros(buffer_size, np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = [None] * buffer_size
        self.done = np.zeros(buffer_size, dtype=bool)
        self.pos = 0
        self.full = False

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.pos

    def append(self, transition):
        """
        Append the last transition to the buffer.
        :param transition: a named tuple of (state, action, reward, new_state, done)
        :return: None
        """
        self.states[self.pos] = transition.state
        self.actions[self.pos] = transition.action
        self.rewards[self.pos] = transition.reward
        self.next_states[self.pos] = transition.new_state
        self.done[self.pos] = transition.done

        self.pos += 1

        if not self.full and self.pos >= self.buffer_size:
            self.full = True

        self.pos %= self.buffer_size

    def sample(self, batch_size):
        """
        Sample a random mini-batch from the replay buffer.
        :param batch_size: size of mini-batch
        :return: sampled mini-batch
        """
        if self.full:
            indices = np.random.choice(self.buffer_size, batch_size, replace=False)
        else:
            indices = np.random.choice(self.pos, batch_size, replace=False)

        actions, rewards, dones = list(map(lambda arr: np.take(arr, indices, axis=0), [self.actions,
                                                                                       self.rewards,
                                                                                       self.done]))
        states = np.stack([np.vstack(self.states[i]) for i in indices]).astype(np.float32) / 255.
        next_states = np.stack([np.vstack(self.next_states[i]) for i in indices]).astype(np.float32) / 255.

        return states, actions, rewards, dones, next_states

    def sample_episode(self, batch_size):
        """
        Sample a whole episode and fill it with transitions from another episode to make it
        multiple of batch_size

        :param batch_size: size of mini-batch
        :return:
        """
        assert batch_size <= self.buffer_size, "batch size must be smaller or eq to buffer size"
        done_idx = np.where(self.done == True)[0]

        batchnum = 0
        while batchnum == 0:
            # sample two episodes (ind1 for main, and ind2 for the remaining steps to make multiple of 32)
            ind = np.random.choice(range(1, len(done_idx) - 1), 2, replace=False)
            ind1 = ind[0]
            ind2 = ind[1]

            indice_array = range(done_idx[ind1 - 1] + 1, done_idx[ind1] + 1)
            epi_len = len(indice_array)  # finish - start
            batchnum = int(np.ceil(epi_len / float(batch_size)))

        remainindex = int(batchnum * batch_size - epi_len)

        # Normally an episode does not have steps=multiple of 32.
        # Fill last minibatch with redundant steps from another episode
        second_indice_arr = list(range(done_idx[ind2], done_idx[ind2] - remainindex, -1))
        second_indice_arr.reverse()
        indice_array = np.append(indice_array, second_indice_arr)
        indice_array = indice_array.astype(int)

        rewards = np.array([self.rewards[i] for i in indice_array])
        actions = np.array([self.actions[i] for i in indice_array])
        dones = np.array([self.done[i] for i in indice_array])

        states = np.array([np.vstack(self.states[i]) for i in indice_array]).astype(np.float32) / 255.
        next_states = np.array([np.vstack(self.next_states[i]) for i in indice_array]).astype(np.float32) / 255.

        return batchnum, states, actions, rewards, dones, next_states
