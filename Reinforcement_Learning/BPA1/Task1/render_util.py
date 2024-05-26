import numpy as np
from gymnasium import Env, spaces
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from screeninfo import get_monitors
from typing import Optional
from os import path
import time
import pygame
import gymnasium as gym

def render(self):
    try:
        import pygame
    except ImportError:
        raise DependencyNotInstalled(
            "pygame is not installed, run `pip install gymnasium[toy_text]`"
        )

    if self.window_surface is None:
        pygame.init()

    if self.render_mode == "human":
        pygame.display.init()
        pygame.display.set_caption("Basketball")
        self.window_surface = pygame.display.set_mode(self.window_size)
    elif self.render_mode == "rgb_array":
        self.window_surface = pygame.Surface(self.window_size)

    # Load the images
    file_name = path.join(path.dirname(__file__), "img/noline_robot_ball.png")
    self.noline_robo_ball = pygame.transform.scale(
        pygame.image.load(file_name), self.cell_size
    )
    file_name = path.join(path.dirname(__file__), "img/noline_robot_happy.png")
    self.noline_robo_happy = pygame.transform.scale(
        pygame.image.load(file_name), self.cell_size
    )
    file_name = path.join(path.dirname(__file__), "img/noline_robot_sad.png")
    self.noline_robo_sad = pygame.transform.scale(
        pygame.image.load(file_name), self.cell_size
    )
    file_name = path.join(path.dirname(__file__), "img/noline_field.png")
    self.noline_field = pygame.transform.scale(
        pygame.image.load(file_name), self.cell_size
    )
    file_name = path.join(path.dirname(__file__), "img/basket_empty.png")
    self.basket_empty = pygame.transform.scale(
        pygame.image.load(file_name), self.cell_size
    )
    file_name = path.join(path.dirname(__file__), "img/basket_miss.png")
    self.basket_miss = pygame.transform.scale(
        pygame.image.load(file_name), self.cell_size
    )
    file_name = path.join(path.dirname(__file__), "img/basket_hit.png")
    self.basket_hit = pygame.transform.scale(
        pygame.image.load(file_name), self.cell_size
    )
    file_name = path.join(path.dirname(__file__), "img/line_robot_ball.png")
    self.line_robo_ball = pygame.transform.scale(
        pygame.image.load(file_name), self.cell_size
    )
    file_name = path.join(path.dirname(__file__), "img/line_robot_happy.png")
    self.line_robo_happy = pygame.transform.scale(
        pygame.image.load(file_name), self.cell_size
    )
    file_name = path.join(path.dirname(__file__), "img/line_robot_sad.png")
    self.line_robo_sad = pygame.transform.scale(
        pygame.image.load(file_name), self.cell_size
    )
    file_name = path.join(path.dirname(__file__), "img/line_field.png")
    self.line_field = pygame.transform.scale(
        pygame.image.load(file_name), self.cell_size
    )
    # create a dictionary for allocation of the correct image
    images = {'line': {'robot': {'ball': self.line_robo_ball, 
                                    'happy': self.line_robo_happy, 
                                    'sad': self.line_robo_sad},
                        'field': {'placeholder': self.line_field}},
                'no_line': {'robot': {'ball': self.noline_robo_ball, 
                                        'happy': self.noline_robo_happy, 
                                        'sad': self.noline_robo_sad},
                        'field': {'placeholder': self.noline_field}},
                'basket': {'empty': {'placeholder': self.basket_empty},
                            'hit': {'placeholder': self.basket_hit},
                            'miss': {'placeholder': self.basket_miss}}}

    # Add the images
    rgb_arrays = []
    terminal = False
    for i in range(self.field_length+1):
        pos = (i*self.render_width, 0)
        # While on the field:
        if i < self.field_length:

            # Verify line
            if i == self.line_position:
                line = 'line'
            else: 
                line = 'no_line'
            # Choose field
            if i == self.state and i < self.field_length:
                type = 'robot'
                spec = 'ball' 
            else:
                type = 'field'
                spec = 'placeholder'
        # Choose basket and success animation
        else:
            line = 'basket'
            spec = 'placeholder'
            if self.state == self.field_length:
                terminal = True
                type = 'hit'
                # add the robot
                if self.laststate == self.line_position:
                    self.window_surface.blit(images['line']['robot']['happy'], 
                                                (self.laststate*self.render_width, 0))
                else: 
                    self.window_surface.blit(images['no_line']['robot']['happy'], 
                                                (self.laststate*self.render_width, 0))
            elif self.state == self.field_length+1:
                terminal = True
                type = 'miss'
                # add the robot
                if self.laststate == self.line_position:
                    self.window_surface.blit(images['line']['robot']['sad'], 
                                                (self.laststate*self.render_width, 0))
                else: 
                    self.window_surface.blit(images['no_line']['robot']['sad'], 
                                                (self.laststate*self.render_width, 0))
                # Note: Animation for leaving the field is the same as missing at the moment
            else:
                type = 'empty'
        self.window_surface.blit(images[line][type][spec], pos)
    
    if self.render_mode == "human":
        pygame.event.pump()
        pygame.display.update()
        if terminal:
            time.sleep(self.render_time*2)
        else:
            time.sleep(self.render_time)
    elif self.render_mode == "rgb_array":
        rgb_arrays.append(np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
        ))
    
    if not all(arr is None for arr in rgb_arrays):
        return rgb_arrays
    
def is_close_to_target(value_list, target, tolerance):
    for value in value_list:
        if abs(value - target) <= tolerance:
            return True
    return False

