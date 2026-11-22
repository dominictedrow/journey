import torch
from dqn import bloons_bot
from memory import ReplayMemory
from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from bloons.env import ACTIONS, Bot


@dataclass
class Configuration:
    maxlen: int = 100000
    bacth_size: int = 64
    epsilon_init : int = 1
    epsilon_decay : float = 0.995
    epsilon_min : float = 0.05
    sync_rate : int = 30
    discount : float = 0.99
    learning_rate : float = 0.001
    optimize_rate : int = 10

config = Configuration()
env = Bot()




class Agent:

    def train(self):
        agent = bloons_bot()
        target_agent = bloons_bot()
        target_agent.load_state_dict(agent.state_dict()) 

        step_count = 0
        env.h.new_game()
        book = ReplayMemory(config.maxlen)
        e = config.epsilon_init
        
        while True:
            current = env.state()
            done = False
            action = ""
            legal = [n for n, ok in env.actions().items() if ok]
            reward = 0
            
            if len(legal) > 0:
                rand = torch.rand(1, dtype=torch.float64)
                if e > rand.item():
                    
                    num = torch.randint(0, len(legal), size = (1,))
                    action = legal[num]
                    env.act(action)
                    
                    if env.state().round_ended:
                        reward = env.state().reward()
                else:
                    legal_mask = list(env.actions().values())
                    x = torch.tensor(current.to_vector())
                    output = agent(x)
                    
                    i = 0
                    maxx =  0
                    
                    for j in range(len(legal_mask)):
                        if legal_mask[j] and output[j] > maxx:
                            maxx = output[j]
                    
                    for i in range(len(legal_mask)):
                        if legal_mask[i] and output[i] == maxx:
                            break
                        
                    action = ACTIONS[i]
                    env.act(action)

            if env.state().outcome == 'loss' or env.state().outcome == 'win':
                done = True
                book.append((current, action, reward, env.state(), done, list(env.actions().values())))
                break
            
            book.append((current, action, reward, env.state(), done))
            e = max(e * config.epsilon_decay, config.epsilon_min)
            step_count+=1
            
            if step_count >= config.sync_rate:
                target_agent.load_state_dict(agent.state_dict())
                step_count = 0

            if step_count % config.optimize_rate == 0:
                agent.optimize(book.sample(config.bacth_size), agent, target_agent, config.discount)
            

        torch.save(agent.state_dict(), "model_weights.pth")

    def run(self):
        step_count = 0
        env.h.new_game()
        book = ReplayMemory(config.maxlen)
        agent = bloons_bot()
        agent.load_state_dict(torch.load("model_weights.pth", weights_only=True))
        
        with torch.no_grad():
            while True:
                current = env.state()
                done = False
                action = ""
                legal = [n for n, ok in env.actions().items() if ok]
                reward = 0
                
                if len(legal) > 0:
                    legal_mask = list(env.actions().values())
                    x = torch.tensor(current.to_vector())
                    output = agent(x)
                    
                    i = 0
                    maxx =  0
                    
                    for j in range(len(legal_mask)):
                        if legal_mask[j] and output[j] > maxx:
                            maxx = output[j]
                    
                    for i in range(len(legal_mask)):
                        if legal_mask[i] and output[i] == maxx:
                            break
                    action = ACTIONS[i]
                    env.act(action)

                if env.state().outcome == 'loss' or env.state().outcome == 'win':
                    done = True
                    book.append((current, action, reward, env.state(), done))
                    break    

                book.append((current, action, reward, env.state(), done, list(env.actions().values())))
                step_count += 1
    

if __name__ == "__main__":
    Agent().train()