import numpy as np
import torch

class ConnectFourBoard:
    def __init__(self, render:bool=True, rows:int = 6, columns:int = 8):
        self.board = torch.zeros(size=(rows, columns))
        self._render = render
        self.render((-1,-1))
        self.reward_won = 1.#10.
        self.reward_loose = -1.#-1.
        self.reward_default = 0.0#0.001

    def reset(self):
        self.board[:,:] = 0# = torch.zeros(size=(self.board.shape[0], self.board.shape[1]))

    def step(self, c:int, player:int):
        done = False
        won = False
        last_step = (-1,-1)
        for r in range(self.board.shape[0]+1):
            if done:
                break
            if r == self.board.shape[0]:
                done = True
                #print("invalid action", c)
                break
            if self.board[r, c] == 0:
                self.board[r,c] = player
                last_step = (r, c)
                won = self.check_won(player)
                if won:
                    done = True
                break
        self.render(last_step)
        reward = self.reward_default
        if done:
            if won:
                reward = self.reward_won
            else:
                reward = self.reward_loose
        
        return done, reward
    
    def check_won(self, player:int):
        won_vert = self.check_won_vert(player)
        won_hor = self.check_won_hor(player)
        won_diag = self.check_won_diag(player)
        return won_vert or won_hor or won_diag

    def check_won_vert(self, player:int):
        for c in range(self.board.shape[1]):
            score = 0
            for r in range(self.board.shape[0]):
                if self.board[r, c] == player:
                    score += 1
                else:
                    score = 0
                if score >= 4:
                    return True
        return False

    def check_won_hor(self, player:int):
        for r in range(self.board.shape[0]):
            score = 0
            for c in range(self.board.shape[1]):
                if self.board[r, c] == player:
                    score += 1
                else:
                    score = 0
                if score >= 4:
                    return True
        return False

    def check_won_diag(self, player:int):
        for r in range(self.board.shape[0]):
            for c in range(self.board.shape[1]):
                score_l = 0
                score_r = 0
                for i in range(self.board.shape[1]):
                    if r-i < 0:
                        continue
                    if c-i >= 0:
                        if self.board[r-i,c-i] == player:
                            score_l += 1
                        else:
                            score_l = 0
                        if score_l >= 4:
                            return True
                    if c+i < self.board.shape[1]:
                        if self.board[r-i,c+i] == player:
                            score_r += 1
                        else:
                            score_r = 0
                        if score_r >= 4:
                            return True
        return False

    def render(self, last_step):
        if not self._render:
            return
        # ─│
        board_str = self.horizontal_line()
        for r in range(self.board.shape[0]-1, -1, -1):
            board_str += "\n"
            for c in range(self.board.shape[1]):
                board_str += "│"
                if self.board[r, c] == 1:
                    if (r, c) == last_step:
                        board_str += ">X<"
                    else:
                        board_str += " X "
                elif self.board[r, c] == -1:
                    if (r, c) == last_step:
                        board_str += ">O<"
                    else:
                        board_str += " O "
                else:
                    board_str += "   "
            board_str += "│\n"
            board_str += self.horizontal_line()
            
        print(board_str)
        print("_______________________________")
    
    def horizontal_line(self):
        board_str = ""
        for c in range(2 * self.board.shape[1] + 1):
            if (c+1)%2 == 0:
                board_str += "───"
            else:
                board_str += " "
        return board_str