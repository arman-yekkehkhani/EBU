import torch
import torch.nn as nn
import torch.nn.functional as F


class EbuTrainer:
    def __init__(self, opts: list, betas):
        self.batch_num = 0
        self.batch_count = 0
        self.y_ = [None for _ in betas]
        self.q_tilde = [None for _ in betas]
        self.actions = None
        self.states = None
        self.betas = betas
        self.opts = opts
        self.criterion = nn.SmoothL1Loss()
        self.K = len(betas)

    def ebu_train_step(self, models, tgts, num_actions, rb, batch_size, device, gamma=0.99):
        if self.batch_num == self.batch_count:

            self.batch_num, self.states, self.actions, rewards, dones, next_states = rb.sample_episode(batch_size)

            self.states = torch.from_numpy(self.states).to(device)  # N, 4, 84, 84
            self.actions = torch.LongTensor(self.actions).to(device)
            rewards = torch.from_numpy(rewards).to(device)
            dones = torch.from_numpy(dones).to(device)
            next_states = torch.from_numpy(next_states).to(device)

            gamma = torch.as_tensor(gamma)

            epi_len = self.batch_num * batch_size

            for k in range(self.K):
                self.q_tilde[k] = torch.zeros((epi_len, num_actions)).to(device)
                with torch.no_grad():
                    for i in range(self.batch_num):
                        s, e = i * batch_size, (i + 1) * batch_size
                        self.q_tilde[k][s: e] = tgts[k](next_states[s: e])

                self.y_[k] = torch.zeros(epi_len).to(device)
                for i in range(epi_len - 1, -1, -1):
                    if dones[i]:
                        self.y_[k][i] = rewards[i]
                    else:
                        self.q_tilde[k][i, self.actions[i + 1]] = self.betas[k] * self.y_[k][i + 1] \
                                                                  + (1 - self.betas[k]) * self.q_tilde[k][
                                                                      i, self.actions[i + 1]]
                        self.y_[k][i] = rewards[i] + gamma * self.q_tilde[k][i, :].max()


                self.opts[k].zero_grad()
                # TODO: does q_vals need to be a list too? for multi-threading. and also actions_one_hot, fix it in
                #  'else' too
                q_vals = models[k](self.states[: batch_size])

                actions_one_hot = F.one_hot(self.actions[: batch_size], num_classes=num_actions)
                q_vals = (q_vals * actions_one_hot).sum(-1)

                loss = self.criterion(self.y_[k][:batch_size], q_vals)
                loss.backward()
                self.opts[k].step()

            self.batch_count = 1

            # FIXME: return loss list
            return None

        else:
            s = self.batch_count * batch_size
            f = (self.batch_count + 1) * batch_size

            for k in range(self.K):
                self.opts[k].zero_grad()
                q_vals = models[k](self.states[s: f])

                actions_one_hot = F.one_hot(self.actions[s: f], num_classes=num_actions)
                q_vals = (q_vals * actions_one_hot).sum(-1)

                loss = self.criterion(self.y_[k][s: f], q_vals)
                loss.backward()
                self.opts[k].step()

            self.batch_count += 1

            # FIXME: return loss list
            return None
