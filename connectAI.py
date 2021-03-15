import connect
import copy
import numpy as np
import random
import time

class Q_Learning:
    def __init__(self, Connect):
        # List of tried state-action pairs
        self._state_action = []
        # Corresponding q(s, a), forms a q table together with self._state_action
        self._q_value = []
        # Split q table into 15 lists for more runtime-efficient search and append (1 for each possible amount of disk on grid)
        for i in range(1, 16):
            self._state_action.append([])
            self._q_value.append([])
        # Total number of disks on grid
        self._total_disks = 0
        self._game = Connect
        # Is game over after a move is played?
        self._game_over = False
        self._prev_move = 0
        self._prev_state = copy.deepcopy(self._game.grid)
        self._prev_index = 0
        # Percentage chance of exploration
        self._e = 10
        self._lamda = 1
        self._alpha = 0.2
        # Total number of moves by the agent for the current game/episode
        self._interactions = 0
    
    # Used by opponent only, uniformly randomly place disk in legal positions
    def random_move(self):
        self._total_disks += 1
        return self._game.act(random.choice(self._game.available_actions))
            
    # Used by agent, epsilon-greedy
    def opt_move(self):
        self._interactions += 1
        moves = self._game.available_actions
        # Indeces for current grid in state-action list
        temp_index = []
        # Maximum of q(s, a) from above indeces
        temp_max = 0
        # Indeces of those maximums
        temp_max_index = []
        self._prev_state = copy.deepcopy(self._game.grid)
        
        counter = 0
        
        for i in range(0, len(self._state_action[self._total_disks-1])):
            if np.array_equal(self._state_action[self._total_disks-1][i][0], self._game.grid) == True:
                temp_index.append(i)
                counter += 1
            
            if counter >= 5:
                break
        
        # If agent has no experience from current state
        if len(temp_index) == 0:
            self._prev_move = random.choice(moves)
            self._prev_index = len(self._state_action[self._total_disks-1])
            self._state_action[self._total_disks-1].append([copy.deepcopy(self._prev_state), self._prev_move])
            self._q_value[self._total_disks-1].append(0)
            self._total_disks += 1
            return self._game.act(self._prev_move)
        
        # If current state has been visited before
        else:
            # Exploit with self._e% chance
            if (random.randint(1,100) > self._e):
                for i in temp_index:
                    temp_max = max(temp_max, self._q_value[self._total_disks-1][i])
                    
                    if self._q_value[self._total_disks-1][i] == temp_max:
                        temp_max_index.append(i)
                    
                    if len(temp_max_index) == 0:
                        self._prev_move = random.choice(moves)
                        self._total_disks += 1
                        return self._game.act(self._prev_move)
                    
                    self._prev_move = self._state_action[self._total_disks-1][random.choice(temp_max_index)][1]
                    self._total_disks += 1
                    return self._game.act(self._prev_move)
            
            # Explore otherwise
            else:
                self._prev_move = random.choice(moves)
                self._total_disks += 1
                return self._game.act(self._prev_move)
    
    # Finds max action value for current state
    def max_reward(self):
        reward = None
        temp_index = []
        for i in range(0, len(self._state_action[self._total_disks-1])):
            if np.array_equal(self._state_action[self._total_disks-1][i][0], self._game.grid):
                temp_index.append(i)
                
        if len(temp_index) >= 1:
            reward = self._q_value[self._total_disks-1][temp_index[0]]
            
            for i in temp_index:
                reward = max(reward, self._q_value[self._total_disks-1][i])
                
        else:
            reward = 0

        return reward
    
    # Plays an episode while updating q table
    def play(self, epsilon = 10, interactions = -1):
        self._e = epsilon
        local_max_reward = None
        first = True
        
        while self._game_over == False:
            reward, self._game_over = self.random_move()

            local_max_reward = self.max_reward()
            
            # if not first pass through
            if first == False:
                for i in range(0, len(self._state_action[self._total_disks-3])):
                    if (np.array_equal(self._state_action[self._total_disks-3][i][0], self._prev_state) == True) and (self._state_action[self._total_disks-3][i][1] == self._prev_move):
                        self._q_value[self._total_disks-3][i] = self._q_value[self._total_disks-3][i] + (self._alpha*(reward + self._lamda*local_max_reward - self._q_value[self._total_disks-3][i]))
                        break
            first = False
                    
            # if game continues after opponent's move
            if self._game_over == False:
                reward, self._game_over = self.opt_move()

                # if agent wins after its move
                if self._game_over == True:
                    for i in range(0, len(self._state_action[self._total_disks-2])):
                        if (np.array_equal(self._state_action[self._total_disks-2][i][0], self._prev_state) == True) and (self._state_action[self._total_disks-2][i][1] == self._prev_move):
                            self._q_value[self._total_disks-2][i] = self._q_value[self._total_disks-2][i] + (self._alpha*(reward + self._lamda*local_max_reward - self._q_value[self._total_disks-2][i]))
                            break
                    
            # if opponent wins after its move
            else:
                for i in range(0, len(self._state_action[self._total_disks-3])):
                    if (np.array_equal(self._state_action[self._total_disks-3][i][0], self._prev_state) == True) and (self._state_action[self._total_disks-3][i][1] == self._prev_move):
                        self._q_value[self._total_disks-3][i] = self._q_value[self._total_disks-3][i] + (self._alpha*(reward + self._lamda*local_max_reward - self._q_value[self._total_disks-3][i]))
                        break
                
        game._game.reset()
        self._total_disks = 0
        self._game_over = False
        
    # Plays n scored games with current policy without changing q table, returns amount of games won
    def play_scored(self, n):
        # Set exploration chance to 0 to test current policy
        self._e = 0
        won = 0
        for i in range(0, n):
            while self._game_over == False:
                reward, self._game_over = self.random_move()

                if self._game_over == False:
                    reward, self._game_over = self.opt_move()

                    if self._game_over == True:
                        won += 1

            game._game.reset()
            self._total_disks = 0
            self._game_over = False
            
        return won
    
    # Plays n scored games between 2 players that places disk uniformly randomly (for base line comparison)
    # Returns amount of games won
    def play_scored_random(self, n):
        won = 0
        for i in range(0, n):
            while self._game_over == False:
                reward, self._game_over = self.random_move()

                if self._game_over == False:
                    reward, self._game_over = self.random_move()

                    if self._game_over == True:
                        won += 1

            game._game.reset()
            self._total_disks = 0
            self._game_over = False
            
        return won