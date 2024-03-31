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
        
	def __init__(self, n_kings, n_levels, max_step=float('inf')):

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
		#print("BP 2")
		self.env_started = 0

		self.step_counter = 0
		self.max_step = max_step

		self.visited = {}

		pygame.display.set_caption('Jump King At Home XD')
		#print("BP 3")


	def reset(self):
		
		for king in self.kings:
			king.reset()

		#self.levels.reset()
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
				# state = [king.levels.current_level, king.x, king.y, king.jumpCount]

				self.visited = {}
				self.visited[(level, king.y)] = 1    
		
		# self.king.reset()
		# state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]

		# self.visited = {}
		# self.visited[(self.king.levels.current_level, self.king.y)] = 1

		return done

	def move_available(self, king):
		available = not king.isFalling \
		and (not king.isSplat or king.splatCount > king.splatDuration)
		return available

	def step(self, actions):
		
		#old_y = (self.king.levels.max_level - self.king.levels.current_level) * 360 + self.king.y
		self.clock.tick(self.fps)
		self._check_events()
		#if not os.environ["pause"]:	To do if we want to do it xd
		self._update_gamestuff(actions=actions)
		self._update_gamescreen()
		self._update_guistuff()
		self._update_audio()
		pygame.display.update()

		for index,king in enumerate(self.kings):
			# if king.y < king.maxy and self.move_available(king):
			# 	print("LMFAO MINUS 100 FOR YOU")
			# 	king.reward -= 100
			if king.maxy > king.y and self.move_available(king):
				king.update_max_y(king.y)
			

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
			print("WHY THE FUCK IS THIS RUNNING LMAO")
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

# def calculate_distance(king, platform):
#     #Ignore the side walls as we do not need them
# 	closest_x = min(abs(platform.x-king.x),abs(platform.x + platform.width-king.x))
# 	if platform.width <= 8 :
# 		distance = 1000
# 		closest_y = 1000
# 	# Find the closest point on the platform rectangle to the king
# 	else :
		
# 		closest_y = min(abs(platform.y-king.y),abs(platform.y + platform.height-king.y))

# 		# Calculate the distance between the king and the closest point
# 		distance = math.sqrt(closest_x ** 2 + closest_y ** 2)
# 	return closest_x,closest_y,distance

# def get_surrounding_platforms(env, king):
# 	zone_of_vision_size_x = 480  # Adjust as needed
# 	zone_of_vision_size_y = 360
# 	surrounding_platforms = []
# 	MAX_PLATFORM_LEVELS = 40
# 	for level in env.levels_list:
# 		for platform in level.levels[level.current_level].platforms: 
# 			# Calculate relative distances to the king
# 			relative_x,relative_y,distance_to_platform = calculate_distance(king, platform)
# 			surrounding_platforms.append((relative_x/480, relative_y/(360*env.n_levels)))

# 	# Pad out the surrounding_platforms list with Max_platform_levels - len(surrounding_platforms) values
# 	surrounding_platforms += [(-1,-1)] * (MAX_PLATFORM_LEVELS - len(surrounding_platforms))
# 	return surrounding_platforms


def calculate_distances(env, king, platform):
    #Ignore the side walls as we do not need them

	if (platform.y < king.y) and (platform.width > 8) and abs(platform.y-king.y) < 154:
		deltax = platform.x - king.x
		if deltax <= 0:
			l = 1
			r = 0 
			closest_x = platform.x+platform.width
			closest_y = platform.y
			dist_x = abs(platform.x+platform.width - king.x)
			dist_y = abs(platform.y - king.y)
			distance = math.sqrt((dist_x/480) ** 2 + (dist_y/(360*env.n_levels)) ** 2)
			return l, r, closest_y, distance
		elif deltax > 0:
			l = 0
			r = 1	
			closest_x = platform.x
			closest_y = platform.y
			dist_x = abs(platform.x - king.x)
			dist_y = abs(platform.y - king.y)
			distance = math.sqrt((dist_x/480) ** 2 + (dist_y/(360*env.n_levels)) ** 2)
			return l, r, closest_y, distance
	else:
		return None, None, None, None
	# Find the closest point on the platform rectangle to the king
	


def get_surrounding_platforms(env, king):
	#zone_of_vision_size_x = 480  # Adjust as needed
	#zone_of_vision_size_y = 360
	surrounding_platforms = []
	MAX_PLATFORM_LEVELS = 6
	for level in env.levels_list:
		for platform in level.levels[level.current_level].platforms: 
			# Calculate relative distances to the king
			l, r, relative_y, distance_to_platform = calculate_distances(env, king, platform)
			if relative_y is not None and distance_to_platform is not None:
				surrounding_platforms.append((l, r, relative_y/(360*env.n_levels), distance_to_platform))
	# Pad out the surrounding_platforms list with Max_platform_levels - len(surrounding_platforms) values
	
	surrounding_platforms += [(-1, -1, -1, -1)] * (MAX_PLATFORM_LEVELS - len(surrounding_platforms))

	return surrounding_platforms

def generate_ml_move(env, king, nets):
	surrounding_platforms = [item for sublist in get_surrounding_platforms(env, king) for item in sublist]
	inputs = surrounding_platforms#	 + [king.y/(360*env.n_levels)]
	output = nets[env.kings.index(king)].activate(inputs)
	length = int(round(output[4] * 31))
	if length < 4:
		alist = generate_random_move()
		return alist
	number = output.index(max(output[0:4]))

	alist = [number] * length
	if number == 2:
		alist.append(0)
	elif number == 3:
		alist.append(1)
	else:
		alist.append(4)
	return alist


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
	

def eval_genomes(genomes, config):

	global generation
	global n_moves
	# Environment Preparation
	action_dict = {
		0: 'right',
		1: 'left',
		2: 'right+space',
		3: 'left+space',
		4: 'idle',
		#5: 'space',
	}                

	env = JKGame(max_step=100000, n_kings=len(genomes), n_levels=3)
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

	# Actually doing some training
	if generation > 5:
		n_moves = int(round(generation*1.6))
	running = True
	toquit = False
	print("Generation: " + str(generation))
	print("Number of Moves: " + str(n_moves))
	generation += 1

	while True:
		for index, king in enumerate(env.kings):
			if kings_finished_list[index] == 1:
				actions[index] = 4
			else:
				if len(actions_queue[index]) > 0:
					actions[index] = actions_queue[index].pop(0)

				elif len(actions_queue[index]) == 0 and kings_move_count[index] < n_moves:
					if env.move_available(king):
						#actions_queue[index] = generate_random_move()
						actions_queue[index] = generate_ml_move(env, king, nets)
						actions[index] = actions_queue[index].pop(0)
						kings_move_count[index] += 1
					else:
						#print("THIS TRIGERED")
						actions[index] = 4

				elif (len(actions_queue[index]) == 0) and kings_move_count[index] >= n_moves:
					kings_finished_list[index] = 1
					actions[index] = 4

		
		if sum(kings_finished_list) >= len(env.kings)-1:
			if all(kings_move_count >= n_moves for kings_move_count in kings_move_count) and all(env.move_available(k) for k in env.kings):
				toquit = True
		
		#print(actions)
		env.step(actions)
		for index, genome in enumerate(genomes):
			genome[1].fitness = env.kings[index].reward

		if toquit:
			for index, genome in enumerate(genomes):
				print(f"King {index+1} Fitness: {genome[1].fitness}")
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


def run(config_file):
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
	p = neat.Population(config)

	# Add a stdout reporter to show progress in the terminal.
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	
	winner = p.run(eval_genomes, 10000)

	# print('\nBest genome:\n{!s}'.format(winner))
	# print('\nTraining completed. Reason: {!s}'.format(p.stop_reason))

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