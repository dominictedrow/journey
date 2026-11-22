import torch as torch
import torch.nn as nn


class bloons_bot(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4272, 16384,bias = True)
        self.layer2 = nn.Linear(16384, 16384, bias = True)
        self.layer3 = nn.Linear(16384, 10499, bias = True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def init_params(self):
        nn.init.xavier_normal_(self.layer1)
        nn.init.xavier_normal_(self.layer2)
        nn.init.xavier_normal_(self.layer3)
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        nn.init.zeros_(self.layer3.bias)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return torch.softmax(x, dim=-1)

    def optimize(self, mini_batch, policy_dqn, target_dqn, discount):
        
        for state, action, reward, next_state, done, legal_mask in mini_batch:
            if done or not any(legal_mask): 
                target_q = reward
            else:
                with torch.no_grad():
                    output = target_dqn(next_state)
                    output = [-torch.inf for i in range(len(output)) if not legal_mask[i]]
                    target_q = reward + discount * torch.max(output)

            current_q = policy_dqn(state)[action]
            loss = nn.MSELoss()(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()





