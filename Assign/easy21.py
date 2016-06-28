import numpy as np
import random

class Easy21(object) :
	# class of easy21 game

	def __init__(self) :
		self.dealer_sum = 0
		self.player_sum = 0
		self.game_in_progress = False

	def draw_card(self) :
		number = random.randint(1, 10)
		if random.random() <= 0.333333 : number = -number
		return number

	def state(self) :
		# (dealer card, player sum)
		return (self.dealer_sum, self.player_sum)

	def init_game(self) : 
		self.dealer_sum = random.randint(1, 10)
		self.player_sum = random.randint(1, 10)
		self.game_in_progress = True
		return self.state() 

	def step(self, action) :
		# if action is "hit", draw card
		# if action is "stick", stop drawing cards and dealer draws cards, and show the result
		# return value is (state, reward). if state is None, it means that game is terminated.

		if not self.game_in_progress :
			print "Please start game before having a step."
			return

		if action == 0 : # hit :
			self.player_sum += self.draw_card()
			if self.player_sum > 21 or self.player_sum < 1 :
				self.game_in_progress = False
				return (None, -1.0)
			else : return (self.state(), 0.0)

		if action == 1 : # "stick"
			self.game_in_progress = False
			while self.dealer_sum < 17 :
				self.dealer_sum += self.draw_card()
				if self.dealer_sum > 21 or self.dealer_sum < 1 :
					return (None, 1.0)

			diff = self.dealer_sum - self.player_sum
			if diff > 0 : return (None, -1.0)
			elif diff < 0 : return (None, 1.0)
			else : return (None, 0.0) 
