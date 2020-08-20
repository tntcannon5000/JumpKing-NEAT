#!/usr/bin/env python
#
#
#
#

import pygame
import os
import math
import collections
from spritesheet import SpriteSheet

class Flyers:

	def __init__(self):

		self.flyers = {}

		self._load_flyers()

	def _load_flyers(self):

		self.flyers[2] = Crow(19, 72, 1, "gold")

		self.flyers[4] = Crow(42, 64, 1, "gold")

		self.flyers[8] = Crow(394, 184, -1, "gold", True)

		self.flyers[10] = Crow(15, 184, 1, "gold")

		self.flyers[42] = Angels()

class Flyer:

	def __init__(self):

		self.scale = int(os.environ.get("resolution"))

		self.images = None

		self.interval = 6

		self.blit_counter = 0

class Crow(Flyer):

	def __init__(self, x, y, direction, gold = "none", reverse = False):

		super().__init__()

		self.rect = pygame.Rect(x * self.scale, y * self.scale, 32 * self.scale, 32 * self.scale)

		self.directory = "Cro"

		self.filename1, self.filename2 = "raven_fly.png", "raven_idle.png"

		self.direction = direction

		self.channel = pygame.mixer.Channel(16)

		self.audio = pygame.mixer.Sound("Audio\\Misc\\bird_fly.wav")

		self.reverse = reverse

		self.interval = 60

		self.gold = gold

		self.state = "idle"

		self.active = False

		self.flyCount = 0

		self.images = {}

		self._load_images()

	def _load_images(self):

		start_rect1, start_rect2 = (0, 0, 48, 32), (0, 0, 32, 32) 

		fly_images = SpriteSheet(f"{self.directory}\\{self.filename1}").load_grid(start_rect1, 3, 2, -1)

		idle_images = SpriteSheet(f"{self.directory}\\{self.filename2}").load_grid(start_rect2, 4, 2, -1)

		if self.reverse:

			fly_images = [pygame.transform.flip(image, True, False) for image in fly_images]

			idle_images = [pygame.transform.flip(image, True, False) for image in idle_images] 

		self.images["none"] = {"idle": idle_images[0:4], "flying" : fly_images[0:3]}

		self.images["gold"] = {"idle" : idle_images[4:], "flying" : fly_images[3:]}

	def blitme(self, screen):

		if self.blit_counter >= len(self.images[self.gold][self.state]) * self.interval:

			self.blit_counter = 0

		screen.blit(self.images[self.gold][self.state][self.blit_counter // self.interval], self.rect)

		self.blit_counter += 1

	def update(self, king):

		if king.rect.colliderect(self.rect):

			if not self.active:

				self.channel.play(self.audio)

			self.active = True

			self.state = "flying"

			self.interval = 6

		if self.active == True:

			self.rect.x += 1 * self.direction

			self.rect.y -= self.flyCount ** 1.001

			self.flyCount += 1

			if self.flyCount > 10:

				self.flyCount = 10

class Angels(Flyer):

	def __init__(self):

		super().__init__()

		self.filename = "ending_animations.png"

		self.start_rect1, self.start_rect2 = (224, 96, 32, 32), (224, 160, 32, 32)

		self.rect = pygame.Rect(int(os.environ.get("screen_width")), 0, 32, 32)

		self.spritesheet = SpriteSheet(self.filename)

		self.images = {}

		self.images["Crown"] = self.spritesheet.load_grid(self.start_rect1, 2, 1, -1)

		self.images["NoCrown"] = self.spritesheet.load_grid(self.start_rect1, 2, 1, -1)

		self.channel = pygame.mixer.Channel(15)

		self.audio = pygame.mixer.Sound("Audio\\Misc\\plink.wav")

		self.active = False

		self.crown = "Crown"

	def blitme(self, screen):

		if self.blit_counter >= len(self.images[self.crown]) * self.interval:

			self.blit_counter = 0

		screen.blit(self.images[self.crown][self.blit_counter // self.interval], self.rect)

		self.blit_counter += 1

	def update(self, king):

		if self.active:
			try:
				self.rect.move_ip(-math.sqrt(self.rect.x - king.x + 2 * self.scale) / 2, math.sqrt(king.y - self.rect.y - king.y / 3) / 2)
			except:
				pass

		else:
			try:
				self.rect.move_ip((math.sqrt(int(os.environ.get("screen_width")) - self.rect.x)), -math.sqrt(self.rect.y - 0))
			except:
				pass