#!/usr/env/bin python
#   
# Game Screen
# 

import pygame 
import sys
import os
import inspect
import pickle
import numpy as np
from environment import Environment
from spritesheet import SpriteSheet
from Background import Backgrounds
from King import King
from Babe import Babe
from Level import Levels
from Menu import Menus

from Start import Start

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time



class JKGame:
	""" Overall class to manga game aspects """
        
	def __init__(self, n_kings, max_step=float('inf')):

		pygame.init()

		self.environment = Environment()

		self.clock = pygame.time.Clock()

		self.fps = int(os.environ.get("fps"))

		self.bg_color = (0, 0, 0)

		self.screen = pygame.display.set_mode((int(os.environ.get("screen_width")) * int(os.environ.get("window_scale")), int(os.environ.get("screen_height")) * int(os.environ.get("window_scale"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)

		self.game_screen = pygame.Surface((int(os.environ.get("screen_width")), int(os.environ.get("screen_height"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)

		print(type(self.game_screen))
		print(type(self.screen))
		
		self.game_screen_x = 0

		pygame.display.set_icon(pygame.image.load("images/sheets/JumpKingIcon.ico"))

		self.levels = Levels(self.game_screen)

		#self.king = King(self.game_screen, self.levels)
		
		self.kings = []
		for _ in range(n_kings):
			self.kings.append(King(self.game_screen, self.levels))

		self.babe = Babe(self.game_screen, self.levels)

		#self.menus = Menus(self.game_screen, self.levels, self.king)

		#self.start = Start(self.game_screen, self.menus)

		self.step_counter = 0
		self.max_step = max_step

		self.visited = {}

		pygame.display.set_caption('Jump King At Home XD')

	def reset(self):
		
		for king in self.kings:
			king.reset()
		self.levels.reset()
		os.environ["start"] = "1"
		os.environ["gaming"] = "1"
		os.environ["pause"] = ""
		os.environ["active"] = "1"
		os.environ["attempt"] = str(int(os.environ.get("attempt")) + 1)
		os.environ["session"] = "0"

		self.step_counter = 0
		done = False
		
		for king in self.kings:
			state = [king.levels.current_level, king.x, king.y, king.jumpCount]

			self.visited = {}
			self.visited[(king.levels.current_level, king.y)] = 1	
		
		# self.king.reset()
		# state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]

		# self.visited = {}
		# self.visited[(self.king.levels.current_level, self.king.y)] = 1

		return done, state

	def move_available(self):
		for king in self.kings:
			available = not king.isFalling \
						and not king.levels.ending \
						and (not king.isSplat or king.splatCount > king.splatDuration)
			return True

	def step(self, actions):
		
		#old_y = (self.king.levels.max_level - self.king.levels.current_level) * 360 + self.king.y
		while True:
			self.clock.tick(self.fps)
			self._check_events()
			if not os.environ["pause"]:
				if not self.move_available():
					actions = None
				self._update_gamestuff(actions=actions)

			self._update_gamescreen()
			self._update_guistuff()
			self._update_audio()
			pygame.display.update()
			for king in self.kings:
				old_level = king.levels.current_level
				old_y = king.y
				if self.move_available():
					self.step_counter += 1
					state = [king.levels.current_level, king.x, king.y, king.jumpCount]
					##################################################################################################
					# Define the reward from environment                                                             #
					##################################################################################################
					if king.levels.current_level > old_level or (king.levels.current_level == old_level and king.y < old_y):
						reward = 0
					else:
						self.visited[(king.levels.current_level, king.y)] = self.visited.get((king.levels.current_level, king.y), 0) + 1
						if self.visited[(king.levels.current_level, king.y)] < self.visited[(old_level, old_y)]:
							self.visited[(king.levels.current_level, king.y)] = self.visited[(old_level, old_y)] + 1

						reward = -self.visited[(king.levels.current_level, king.y)]
					####################################################################################################

					done = True if self.step_counter > self.max_step else False
					return state, reward, done

	def running(self):
		"""
		play game with keyboard
		:return:
		"""
		self.reset()
		while True:
			#state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
			#print(state)
			self.clock.tick(self.fps)
			self._check_events()
			if not os.environ["pause"]:
				self._update_gamestuff()

			self._update_gamescreen()
			self._update_guistuff()
			self._update_audio()
			pygame.display.update()

	def _check_events(self):

		for event in pygame.event.get():

			if event.type == pygame.QUIT:

				self.environment.save()

				self.menus.save()

				sys.exit()

			if event.type == pygame.KEYDOWN:

				self.menus.check_events(event)

				if event.key == pygame.K_c:

					if os.environ["mode"] == "creative":

						os.environ["mode"] = "normal"

					else:

						os.environ["mode"] = "creative"
					
			if event.type == pygame.VIDEORESIZE:

				self._resize_screen(event.w, event.h)

	def _update_gamestuff(self, actions=None):

		self.levels.update_levels(self.kings, self.babe, agentCommand=actions)

	def _update_guistuff(self):

		# if self.menus.current_menu:

		# 	self.menus.update() menu

		if not os.environ["gaming"]:

			self.start.update()

	def _update_gamescreen(self):

		pygame.display.set_caption(f"Jump King At Home XD - {self.clock.get_fps():.2f} FPS")

		self.game_screen.fill(self.bg_color)

		if os.environ["gaming"]:

			self.levels.blit1()

		if os.environ["active"]:

			for king in self.kings:
				king.blitme()

		if os.environ["gaming"]:

			self.babe.blitme()

		if os.environ["gaming"]:

			self.levels.blit2()

		if os.environ["gaming"]:

			self._shake_screen()

		if not os.environ["gaming"]:

			self.start.blitme()

		# self.menus.blitme() menu

		self.screen.blit(pygame.transform.scale(self.game_screen, self.screen.get_size()), (self.game_screen_x, 0))

	def _resize_screen(self, w, h):

		self.screen = pygame.display.set_mode((w, h), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.SRCALPHA)

	def _shake_screen(self):

		try:

			if self.levels.levels[self.levels.current_level].shake:

				if self.levels.shake_var <= 150:

					self.game_screen_x = 0

				elif self.levels.shake_var // 8 % 2 == 1:

					self.game_screen_x = -1

				elif self.levels.shake_var // 8 % 2 == 0:

					self.game_screen_x = 1

			if self.levels.shake_var > 260:

				self.levels.shake_var = 0

			self.levels.shake_var += 1

		except Exception as e:

			print("SHAKE ERROR: ", e)

	def _update_audio(self):

		for channel in range(pygame.mixer.get_num_channels()):

			if not os.environ["music"]:

				if channel in range(0, 2):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			if not os.environ["ambience"]:

				if channel in range(2, 7):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			if not os.environ["sfx"]:

				if channel in range(7, 16):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			pygame.mixer.Channel(channel).set_volume(float(os.environ.get("volume")))


def train():

	action_dict = {
		0: 'right',
		1: 'left',
		2: 'right+space',
		3: 'left+space',
		# 4: 'idle',
		# 5: 'space',
	}

	env = JKGame(n_kings=4,max_step=1000)
	num_episode = 3
	action_keys = list(action_dict.keys())


	for i in range(num_episode):
		done, state = env.reset()
		yourmother = True
		yourcounter = 0
		while yourmother:     # game ends when loop exits
			action = np.random.choice(action_keys)
			next_state, reward, done = env.step(action)
			yourcounter += 1
			if yourcounter > 30:
				yourmother = False


def train(n_generations):

    action_dict = {
        0: 'right',
        1: 'left',
        2: 'right+space',
        3: 'left+space',
        # 4: 'idle',
        # 5: 'space',
    }

    env = JKGame(max_step=1000, n_kings=5)
    env.reset()
    action_keys = list(action_dict.keys())

    for generation in range(n_generations):
        env.reset()
        yourmother = True
        yourcounter = 0
        while yourmother:
            actions = []
            for king in env.kings:
                action = np.random.choice(action_keys)
                actions.append(action)
            env.step(actions)
            yourcounter += 1
            if yourcounter > 3000:
                yourmother = False

if __name__ == "__main__":
	#Game = JKGame()
	#Game.running()
	train(1)