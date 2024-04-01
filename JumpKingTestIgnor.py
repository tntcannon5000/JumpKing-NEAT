#!/usr/env/bin python
#   
# Game Screen
# 

import gc
import math
import pygame 
import os
import numpy as np
from environment import Environment
from King import King
from Babe import Babe
from Level import Levels
import random
import neat
import cProfile

generation = 0
n_moves = 8


class JKGame:
	""" Overall class to manga game aspects """
        
	def __init__(self, n_kings, n_levels):

		self.n_levels = n_levels
		pygame.init()

		self.environment = Environment(n_levels)

		self.clock = pygame.time.Clock()

		self.fps = 60

		self.bg_color = (0, 0, 0)

		self.screen = pygame.display.set_mode((int(os.environ.get("screen_width")) * int(os.environ.get("window_scale")), int(os.environ.get("screen_height")) * int(os.environ.get("window_scale"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)

		self.game_screen = pygame.Surface((int(os.environ.get("screen_width")), int(os.environ.get("screen_height"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)
		
		self.game_screen_x = 0

		pygame.display.set_icon(pygame.image.load("images/sheets/JumpKingIcon.ico"))

		self.levels_list = []
		self.levels_list = [Levels(self.game_screen, init_level=levelnum, n_levels=n_levels) for levelnum in range(n_levels)]
		self.levels = self.levels_list[0]

		self.kings = []
		self.kings = [King(self.game_screen, self.levels_list, n_levels) for _ in range(n_kings)]

		self.babe = Babe(self.game_screen, self.levels)

		#self.menus = Menus(self.game_screen, self.levels, self.king)

		#self.start = Start(self.game_screen, self.menus)

		self.action_dict = {
		0: 'right',
		1: 'left',
		2: 'right+space',
		3: 'left+space',
		#4: 'idle',
		# 5: 'space',
		} 
		self.action_keys = list(self.action_dict.keys())
		self.env_started = 0

		self.step_counter = 0

		self.visited = {}

		pygame.display.set_caption('Jump King At Home XD')


	def reset(self):
		
		for king in self.kings:
			king.reset()

		os.environ["start"] = "1"
		os.environ["gaming"] = "1" 
		os.environ["pause"] = ""
		os.environ["active"] = "1"
		os.environ["attempt"] = str(int(os.environ.get("attempt")) + 1)
		os.environ["session"] = "0"

		self.step_counter = 0
		done = False
		
		for king in self.kings:
			for level in self.levels_list:

				self.visited = {}
				self.visited[(level, king.y)] = 1    

		return done

	# Checks wether the king can move or not
	def move_available(self, king):
		available = not king.isFalling \
		and (not king.isSplat or king.splatCount > king.splatDuration)
		return available

	def step(self, actions):
		
		self.clock.tick(self.fps)
		self._check_events()
		#if not os.environ["pause"]:
		#Pass the actions to the game environment to make the king move
		self._update_gamestuff(actions=actions)
		self._update_gamescreen()
		self._update_guistuff()
		self._update_audio()
		pygame.display.update()

		#Update the max y value of the king if the king can move which means the king is on top of a platform
		for index,king in enumerate(self.kings):
			if king.maxy > king.y and self.move_available(king):
				king.update_max_y(king.y)
			

	def running(self):
		"""
		play game with keyboard
		:return:
		"""
		self.reset()
		while True:
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

				#sys.exit()

			if event.type == pygame.KEYDOWN:

				#self.menus.check_events(event)

				if event.key == pygame.K_c:

					if os.environ["mode"] == "creative":

						os.environ["mode"] = "normal"

					else:

						os.environ["mode"] = "creative"
					
			if event.type == pygame.VIDEORESIZE:

				self._resize_screen(event.w, event.h)

	def _update_gamestuff(self, actions=None):

		self.levels.update_levels(self.kings, agentCommand=actions)

	def _update_guistuff(self):

		# if self.menus.current_menu:

		# 	self.menus.update() menu

		if not os.environ["gaming"]:
			self.start.update()

	def _update_gamescreen(self):

		pygame.display.set_caption(f"Jump King At Home XD - {self.clock.get_fps():.2f} FPS")

		self.game_screen.fill(self.bg_color)

		if os.environ["gaming"]:

			for level1 in self.levels_list:
				level1.blit1()

		if os.environ["active"]:
			for king in self.kings:
				king.blitme()

		if os.environ["gaming"]:

			self.babe.blitme()

		if os.environ["gaming"]:
			
			for level2 in self.levels_list:
				level2.blit2()

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

def calculate_distance(king, platform):
	closest_x = min(abs(platform.x-king.x),abs(platform.x + platform.width-king.x))
	if platform.width <= 8 :
		distance = 1000
		closest_y = 1000
	# Find the closest point on the platform rectangle to the king
	else :
		
		closest_y = min(abs(platform.y-king.y),abs(platform.y + platform.height-king.y))

		# Calculate the distance between the king and the closest point
		distance = math.sqrt(closest_x ** 2 + closest_y ** 2)
	return closest_x,closest_y,distance

# Function to get the surrounding platforms of the king
def get_surrounding_platforms(env, king):
	surrounding_platforms = []
	MAX_PLATFORM_LEVELS = 40
	# Fetch all the platforms in the levels loaded
	for level in env.levels_list:
		for platform in level.levels[level.current_level].platforms:
			relative_x,relative_y,distance_to_platform = calculate_distance(king, platform)
			surrounding_platforms.append((relative_x/480, relative_y/(360*env.n_levels)))

	# Pad out the surrounding_platforms list with Max_platform_levels - len(surrounding_platforms) values
	surrounding_platforms += [(-1,-1)] * (MAX_PLATFORM_LEVELS - len(surrounding_platforms))
	return surrounding_platforms

# Function to generate moves for the king using the NEAT algorithm
def generate_ml_move(env, king, nets):
	surrounding_platforms = [item for sublist in get_surrounding_platforms(env, king) for item in sublist]
	inputs = surrounding_platforms
	output = nets[env.kings.index(king)].activate(inputs)
 
	# The 4th output is the number of steps the king will take. It is scaled to 31 because that is the max limit it can crouch to make a jump	
	length = int(round(output[4] * 31))
	# Choosing one of the steps form 0-4 based on the max output value
	number = output.index(max(output[0:4]))
 
	alist = [number] * length
	# If the move is 2 i.e right+space, add 0 to let go of the space key and move right since 0 is right
	if number == 2:
		alist.append(0)
  	# If the move is 3 i.e left+space, add 1 to let go of the space key and move left since 1 is left
	elif number == 3:
		alist.append(1)
	else:
		alist.append(4)
	return alist

# Function to generate random moves for the king to test if code is working correctly
def generate_random_move():
	length = np.random.randint(1, 30)
	number = random.randint(0, 3)
	random_list = [number] * length
	if number == 2:
		random_list.append(0)
	elif number == 3:
		random_list.append(1)
	else:
		random_list.append(4)
	return random_list 	
	
# Function to evaluate the genomes
def eval_genomes(genomes, config):

	# Setting up global variables to increase the number of moves a king can make in a generation as the generations progress
	global generation
	global n_moves
	# Moves available to the king and their corresponding numbers
	# 0: 'right',
	# 1: 'left',
	# 2: 'right+space',
	# 3: 'left+space',
	# 4: 'idle',
	# 5: 'space',             

	# Initializing the game environment with the number of genomes for the generation and the number of levels we want to play
	env = JKGame(n_kings=len(genomes), n_levels=2)
	env.reset()

	nets = []
	actions_queue = []
	for genome_id, genome in genomes:
		genome.fitness = 0  # start with fitness level of 0
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		nets.append(net)
		actions_queue.append([])


	actions = [0] * len(genomes)
	
	kings_move_count = [0] * len(genomes)
	kings_finished_list = [0] * len(genomes)

	# Increasing moves every 5 generations
	if generation % 5 == 0:
		n_moves += (int(generation/5)*5)
	running = True
	toquit = False
	print("Generation: " + str(generation))
	print("Number of Moves: " + str(n_moves))
	#Counting the generations
	generation += 1

	
	while True:
		for index, king in enumerate(env.kings):
			# If the king has finished all moves for the level, set the action to idle
			if kings_finished_list[index] == 1:
				actions[index] = 4
			else:
				# If the action set of steps for a move has not finished, get the next action from the queue
				if len(actions_queue[index]) > 0:
					actions[index] = actions_queue[index].pop(0)
				# If the action set of steps for a move has finished, check if the king has moves left for the generation
				elif len(actions_queue[index]) == 0 and kings_move_count[index] < n_moves:
					#Check if the king can move
					if env.move_available(king):
						#actions_queue[index] = generate_random_move()
						#Get a set of steps for a move from the network
						actions_queue[index] = generate_ml_move(env, king, nets)
						actions[index] = actions_queue[index].pop(0)
						kings_move_count[index] += 1
					#Stay idle if the king cannot move i.e if it is falling
					else:
						actions[index] = 4
				# If the king has finished all moves for the generation, set the action to idle and update the list of kings that have finished
				elif (len(actions_queue[index]) == 0):
					kings_finished_list[index] = 1
					actions[index] = 4

		# Check if all kings have finished all moves for the generation and if all kings can move
		if sum(kings_finished_list) >= len(env.kings)-1:
			if all(kings_move_count >= n_moves for kings_move_count in kings_move_count) and all(env.move_available(k) for k in env.kings):
				toquit = True
		
		# Calling step function to update step of each king with the actions obtained from the network.
		env.step(actions)
		for index, genome in enumerate(genomes):
			genome[1].fitness = env.kings[index].reward

		# Deleting the environment and the kings to free up memory after the generation has finished
		if toquit:
			for level in env.levels_list:
				del level.background_audio
				del level.channels
				del level.background
				del level.foreground
				del level.midground
				del level.platforms
				del level.props
				del level.wind
				del level.npcs
				del level.flyers
				del level.readables
				del level.Ending_Animation
				del level.levels
				del level.weather
				del level.hiddenwalls
				del level.scrollers
				del level
				gc.collect()

			for king in env.kings:
				del king.screen
				del king.levels_list
				del king.timer
				del king.sprites
				del king.mask
				del king.maxy
				del king.reward
				del king.channel
				del king.audio
				del king
				gc.collect()

			del env.levels_list
			del env.levels
			del env.kings
			del env.babe
			del env

			gc.collect()
			break

# Running the NEAT algorithm
def run(config_file):
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
	p = neat.Population(config)

	# Add a stdout reporter to show progress in the terminal.
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	
	winner = p.run(eval_genomes, 100)

# Fetching the configuration file and passing it to the run function
def run_game():
    run(os.path.join(os.path.dirname(__file__), 'networkconfig.txt'))

# def train_n_games(n_games):
# 	processes = []
# 	for i in range(n_games):
# 		#run(os.path.join(os.path.dirname(__file__), 'networkconfig.txt'))
# 		p = Process(target=run_game)
# 		processes.append(p)
# 	for p in processes:
# 		p.start()
# 	for p in processes:
# 		p.join()

# if __name__ == "__main__":
# train_n_games(1)
# if __name__ == "__main__":
#     p1 = Process(target=run_game)
#     p2 = Process(target=run_game)

#     p1.start()
#     p2.start()

#     p1.join()
#     p2.join()

if __name__ == "__main__":
# 	Game = JKGame(2)
# 	Game.running()
# 	#train(1)
 	#run(os.path.join(os.path.dirname(__file__), 'networkconfig.txt'))
	# cProfile.run('run_game()', 'profile_results')
	# import pstats
	# p = pstats.Stats('profile_results')
	# p.sort_stats('cumulative').print_stats()
	run_game()