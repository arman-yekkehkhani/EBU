import torch
import torch.nn as nn
import torch.nn.functional as F


class EbuTrainer:
    def __init__(self, opt):
        self.batch_num = 0
        self.batch_count = 0
        self.y_ = None
        self.q_tilde = None
        self.actions = None
        self.states = None
        self.opt = opt
        self.criterion = nn.SmoothL1Loss()

    def ebu_train_step(self, model, tgt, num_actions, rb, batch_size, device,
                       beta=0.5, gamma=0.99):
        if self.batch_num == self.batch_count:

            self.batch_num, self.states, self.actions, rewards, dones, next_states = rb.sample_episode(batch_size)

            self.states = torch.from_numpy(self.states).to(device)  # N, 4, 84, 84
            self.actions = torch.LongTensor(self.actions).to(device)
            rewards = torch.from_numpy(rewards).to(device)
            dones = torch.from_numpy(dones).to(device)
            next_states = torch.from_numpy(next_states).to(device)

            gamma = torch.as_tensor(gamma)

            epi_len = self.batch_num * batch_size

            self.q_tilde = torch.zeros((epi_len, num_actions)).to(device)
            with torch.no_grad():
                for i in range(self.batch_num):
                    s, e = i * batch_size, (i + 1) * batch_size
                    self.q_tilde[s: e] = tgt(next_states[s: e])

            self.y_ = torch.zeros(epi_len).to(device)
            for i in range(epi_len - 1, -1, -1):
                if dones[i]:
                    self.y_[i] = rewards[i]
                else:
                    self.q_tilde[i, self.actions[i + 1]] = beta * self.y_[i + 1] \
                                                           + (1 - beta) * self.q_tilde[i, self.actions[i + 1]]
                    self.y_[i] = rewards[i] + gamma * self.q_tilde[i, :].max()

            self.batch_count = 1

            self.opt.zero_grad()
            q_vals = model(self.states[: batch_size])

            actions_one_hot = F.one_hot(self.actions[: batch_size], num_classes=num_actions)
            q_vals = (q_vals * actions_one_hot).sum(-1)

            loss = self.criterion(self.y_[:batch_size], q_vals)
            loss.backward()
            self.opt.step()
            return loss.detach().item()

        else:
            self.opt.zero_grad()
            s = self.batch_count * batch_size
            f = (self.batch_count + 1) * batch_size
            q_vals = model(self.states[s: f])

            actions_one_hot = F.one_hot(self.actions[s: f], num_classes=num_actions)
            q_vals = (q_vals * actions_one_hot).sum(-1)

            loss = self.criterion(self.y_[s: f], q_vals)
            loss.backward()
            self.opt.step()

            self.batch_count += 1

            return loss.detach().item()
