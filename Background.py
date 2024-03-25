#!/usr/bin/env python
#
#
#
#

import pygame
import re
import os
import collections

class Background():

	def __init__(self, filename, current_level, n_levels):

		self.image = self._load_image(filename)

		self.x, self.y = 0, 0
		self.width, self.height = self.image.get_size()
		self.current_level = current_level
		self.n_levels = n_levels

	@property
	def rect(self):
		return pygame.Rect(self.x, self.y+360*((self.n_levels-1)-self.current_level), self.width, self.height)
	
	def _load_image(self, filename, colorkey = None):

		""" Load a specific image from a file """

		try:

			image = pygame.image.load(filename).convert_alpha()

		except pygame.error as e:
			print(f'Unable To Load Image: {filename}')
			#raise SystemExit(e)

		return image

	def blitme(self, screen):

		screen.blit(self.image, self.rect)

class Backgrounds():

	def __init__(self, directory, n_screen, n_levels):

		pygame.init()

		self.directory = directory
		self.n_screen = n_screen
		self.n_levels = n_levels
		self.backgrounds = collections.defaultdict()

		self._load_background_sprites()
		

	def _load_background_sprites(self):

		for filename in sorted(os.listdir(self.directory), key = lambda filename: int(re.search(r'\d+', filename).group())):
			
			bg = Background(os.path.join(self.directory, filename), self.n_screen, self.n_levels)

			level = int(re.search(r'\d+', filename).group()) - 1

			self.backgrounds[level] = bg

if __name__ == "__main__":

	background = Backgrounds(pygame.display.set_mode((480, 360)), 's', "BG")
