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



import random
import time
import neat



class JKGame:
	""" Overall class to manga game aspects """
        
	def __init__(self, max_step=float('inf'),):

		pygame.init()

		self.environment = Environment()

		self.clock = pygame.time.Clock()

		self.fps = int(os.environ.get("fps"))
 
		self.bg_color = (0, 0, 0)

		self.screen = pygame.display.set_mode((int(os.environ.get("screen_width")) * int(os.environ.get("window_scale")), int(os.environ.get("screen_height")) * int(os.environ.get("window_scale"))), pygame.HWSURFACE|pygame.DOUBLEBUF)

		self.game_screen = pygame.Surface((int(os.environ.get("screen_width")), int(os.environ.get("screen_height"))), pygame.HWSURFACE|pygame.DOUBLEBUF)

		self.game_screen_x = 0

		pygame.display.set_icon(pygame.image.load("images\\sheets\\JumpKingIcon.ico"))

		self.levels = Levels(self.game_screen)

		self.king = King(self.game_screen, self.levels)

		self.babe = Babe(self.game_screen, self.levels)

		self.menus = Menus(self.game_screen, self.levels, self.king)

		self.start = Start(self.game_screen, self.menus)

		self.step_counter = 0
		self.max_step = max_step

		self.visited = {}

		pygame.display.set_caption('Jump King At Home XD')

	def reset(self):
		self.king.reset()
		self.levels.reset()
		os.environ["start"] = "1"
		os.environ["gaming"] = "1"
		os.environ["pause"] = ""
		os.environ["active"] = "1"
		os.environ["attempt"] = str(int(os.environ.get("attempt")) + 1)
		os.environ["session"] = "0"

		self.step_counter = 0
		done = False
		state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]

		self.visited = {}
		self.visited[(self.king.levels.current_level, self.king.y)] = 1

		return done, state

	def move_available(self):
		available = not self.king.isFalling \
					and not self.king.levels.ending \
					and (not self.king.isSplat or self.king.splatCount > self.king.splatDuration)
		return available

	def step(self, action):
		old_level = self.king.levels.current_level
		old_y = self.king.y
		#old_y = (self.king.levels.max_level - self.king.levels.current_level) * 360 + self.king.y
		while True:
			self.clock.tick(self.fps)
			self._check_events()
			if not os.environ["pause"]:
				if not self.move_available():
					action = None
				self._update_gamestuff(action=action)

			self._update_gamescreen()
			self._update_guistuff()
			self._update_audio()
			pygame.display.update()


			if self.move_available():
				self.step_counter += 1
				state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
				##################################################################################################
				# Define the reward from environment                                                             #
				##################################################################################################
				if self.king.levels.current_level > old_level or (self.king.levels.current_level == old_level and self.king.y < old_y):
					reward = 0
				else:
					self.visited[(self.king.levels.current_level, self.king.y)] = self.visited.get((self.king.levels.current_level, self.king.y), 0) + 1
					if self.visited[(self.king.levels.current_level, self.king.y)] < self.visited[(old_level, old_y)]:
						self.visited[(self.king.levels.current_level, self.king.y)] = self.visited[(old_level, old_y)] + 1

					reward = -self.visited[(self.king.levels.current_level, self.king.y)]
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

	def _update_gamestuff(self, action=None):

		self.levels.update_levels(self.king, self.babe, agentCommand=action)

	def _update_guistuff(self):

		if self.menus.current_menu:

			self.menus.update()

		if not os.environ["gaming"]:

			self.start.update()

	def _update_gamescreen(self):

		pygame.display.set_caption(f"Jump King At Home XD - {self.clock.get_fps():.2f} FPS")

		self.game_screen.fill(self.bg_color)

		if os.environ["gaming"]:

			self.levels.blit1()

		if os.environ["active"]:

			self.king.blitme()

		if os.environ["gaming"]:

			self.babe.blitme()

		if os.environ["gaming"]:

			self.levels.blit2()

		if os.environ["gaming"]:

			self._shake_screen()

		if not os.environ["gaming"]:

			self.start.blitme()

		self.menus.blitme()

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



nnetworks = []
kings = []
genomeslist = []

def eval_genomes(genomes, config):
	# CODE TO RUN EACH GENERATION
	Game = JKGame()
	for genome_id, genome in genomes:
		genome.fitness = 0
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		nnetworks.append(net)
		kings.append(King(Game.screen, Game.levels))
		genomeslist.append(genome)

# def prepare_game(n_genomes):
# 	Game = JKGame()
# 	for genome_id, genome in n_genomes:
# 		genome.fitness = 0
# 		net = neat.nn.FeedForwardNetwork.create(genome, config)
# 		nnetworks.append(net)
# 		kings.append(King(game.screen, game.levels))
# 		genomeslist.append(genome)



	# for i, king in enumerate(kings):
	# 	genomeslist[i].fitness += 0.1
	# 	while not False:
	# 		output = nnetworks[i].activate(next_state)
	# 		action = np.argmax(output)
	# 		next_state, reward, done = Game.step(action)
	# 		genomeslist[i].fitness += reward

	print("your mother")

def run(config_file):
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
	pop = neat.Population(config)

	pop.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	pop.add_reporter(stats)

	game = JKGame()
	winner = pop.run(eval_genomes, 5)
	print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
	#Game = JKGame()
	#Game.running()
	run(os.path.join(os.path.dirname(__file__), 'networkconfig.txt'))