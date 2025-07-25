# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train an agent to solve the SlimeVolley task.

Slime Volleyball is a game created in the early 2000s by unknown author.

The game is very simple: the agent's goal is to get the ball to land on
the ground of its opponent's side, causing its opponent to lose a life.

Each agent starts off with five lives. The episode ends when either agent
loses all five lives, or after 3000 timesteps has passed. An agent receives
a reward of +1 when its opponent loses or -1 when it loses a life.

An agent loses when it loses 5 times in the Test environment, or if it
loses based on score count after 3000 time steps.

During Training, the game is simply played for 3000 time steps, not
terminating even when one player loses 5 times.

This task is based on:
https://otoro.net/slimevolley/
https://github.com/hardmaru/slimevolleygym

Example command to run this script: `python train_slimevolley.py --gpu-id=0`
"""

import argparse
import os
import shutil
import jax

from evojax.task.slimevolley import SlimeVolley
from evojax.policy.mlp import MLPPolicy
from evojax.algo import CMA
from evojax import Trainer
from evojax import util

import copy
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def visualize_network(npz_file, save_pth=None):
	"""
	Load parameters from the saved npz file and visualize the MLP network structure.
	The assumed architecture is: input layer (12 nodes), hidden layer (20 nodes), output layer (3 nodes).
	We display each bias as a separate node.
	
	Args:
		npz_file (str): Path to the npz file.
		save_pth (str): Path to save the visualization image. If omitted, only display.
	"""
	# Load npz file
	data = np.load(npz_file)
	params = data["params"]
	print(f"Loaded params shape: {params.shape}")
	
	# Define the MLPPolicy architecture
	input_size = 12
	hidden_size = 20
	output_size = 3
	
	# Split up the parameters
	offset = 0
	# Weights from input to hidden layer (12*20 = 240)
	breakpoint()
	w1 = params[offset:offset + input_size * hidden_size].reshape(input_size, hidden_size)
	offset += input_size * hidden_size
	# Bias for hidden layer (20)
	b1 = params[offset:offset + hidden_size]
	offset += hidden_size
	# Weights from hidden to output layer (20*3 = 60)
	w2 = params[offset:offset + hidden_size * output_size].reshape(hidden_size, output_size)
	offset += hidden_size * output_size
	# Bias for output layer (3)
	b2 = params[offset:offset + output_size]
	offset += output_size
	
	# Create a directed graph with NetworkX
	G = nx.DiGraph()
	
	# Node lists
	input_nodes  = [f"I{i}" for i in range(input_size)]
	hidden_nodes = [f"H{i}" for i in range(hidden_size)]
	output_nodes = [f"O{i}" for i in range(output_size)]
	
	# Add bias nodes for hidden layer and output layer
	bias_hidden = "B_hidden"
	bias_output = "B_output"
	
	# Add nodes to the graph (with colors)
	G.add_nodes_from(input_nodes, color="blue")
	G.add_nodes_from(hidden_nodes, color="green")
	G.add_nodes_from(output_nodes, color="red")
	G.add_node(bias_hidden, color="purple")
	G.add_node(bias_output, color="purple")
	
	# Add edges from input layer to hidden layer
	for i, inp in enumerate(input_nodes):
		for j, hid in enumerate(hidden_nodes):
			weight = w1[i, j]
			if abs(weight) > 0.01:
				G.add_edge(inp, hid, weight=weight)
	
	# Add edges from hidden-layer bias to each hidden node
	for i, hid in enumerate(hidden_nodes):
		weight = b1[i]
		if abs(weight) > 0.01:
			G.add_edge(bias_hidden, hid, weight=weight)
	
	# Add edges from hidden layer to output layer
	for i, hid in enumerate(hidden_nodes):
		for j, out in enumerate(output_nodes):
			weight = w2[i, j]
			if abs(weight) > 0.01:
				G.add_edge(hid, out, weight=weight)
	
	# Add edges from output-layer bias to each output node
	for j, out in enumerate(output_nodes):
		weight = b2[j]
		if abs(weight) > 0.01:
			G.add_edge(bias_output, out, weight=weight)
	
	# Retrieve node colors
	colors = [G.nodes[n]["color"] for n in G.nodes]
	
	# Drawing settings
	plt.figure(figsize=(12, 8))
	pos = nx.spring_layout(G, seed=42)
	nx.draw(G, pos, with_labels=True, node_color=colors, edge_color="gray", node_size=500)
	edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
	nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
	plt.title("Evolved NEAT MLP Network Visualization")
	if save_pth is not None:
		plt.savefig(save_pth)
	plt.show()

# -------------------------------
# Simple NEAT Implementation (Hard-Coded)
# -------------------------------
# Here, we skip the complex node/connection representation of NEAT and treat each individual’s
# genetic representation as a single weight vector (wVec).
# Since the Trainer uses a fixed MLPPolicy (with a parameter count of 323),
# the length of each individual's wVec is 323 (or hyp["ann_num_params"]).

class Ind:
	"""
	Individual (Genome) class.
	This class manages a weight vector (wVec) as the genetic representation of an individual in NEAT.
	It also stores the fitness (fitness) and the generation of birth (birth).
	"""
	def __init__(self, wVec):
		# wVec: a 1D array representing the parameters of the individual
		self.wVec = np.copy(wVec)
		self.fitness = 0.0
		self.birth = 0

	def mutate(self, p):
		"""
		Mutation process for an individual:
		- For each element, add a perturbation based on a normal distribution with probability p['prob_mutConn'].
		
		Args:
			p (dict): Hyperparameters for mutation rate and magnitude, etc.
		"""
		mutated = np.random.rand(self.wVec.shape[0]) < p['prob_mutConn']
		delta = mutated * np.random.randn(self.wVec.shape[0]) * p['ann_mutSigma']
		self.wVec += delta
		return self

	def crossover(self, mate):
		"""
		Crossover process:
		- Randomly select parts of the two parents’ weight vectors to pass on to the child.
		
		Args:
			mate (Ind): The individual to crossover with.
		
		Returns:
			child (Ind): The newly created offspring.
		"""
		child = copy.deepcopy(self)
		bProb = 0.5  # Probability of selecting genes from mate
		mask = np.random.rand(self.wVec.shape[0]) < bProb
		child.wVec[mask] = mate.wVec[mask]
		return child

class NEATJax:
	"""
	Simplified NEAT evolutionary algorithm class.
	Trainer expects solver.ask() to return a JAX array (shape: (pop_size, param_size)),
	so here we implement stacking of each individual's genetic representation (wVec) when returning it.
	"""
	def __init__(self, hyp):
		"""
		Constructor
		
		Args:
			hyp (dict): A dictionary containing the hyperparameters of NEAT.
		"""
		self.p = hyp
		self.pop_size = hyp['popSize']  # Define pop_size so that Trainer can reference it
		self.pop = []         # Population (a list of Ind objects)
		self.gen = 0          # Generation counter
		self.best_ind = None  # The best individual so far

	def initPop(self):
		"""
		Generate the initial population:
		Create a random weight vector for each individual according to the fixed parameter count (ann_num_params).
		"""
		nParams = self.p.get("ann_num_params", 323)
		# Generate initial weights with uniform random distribution (range: -ann_absWCap to ann_absWCap)
		wVec = np.random.uniform(-self.p["ann_absWCap"], self.p["ann_absWCap"], size=nParams)
		self.pop = []
		for i in range(self.p['popSize']):
			ind = Ind(wVec)
			self.pop.append(copy.deepcopy(ind))
		self.best_ind = self.pop[0]

	def ask(self):
		"""
		Generate the next generation of the population and return it as a JAX array for the Trainer.
		For the first time, it creates the initial population;
		afterwards, it generates the new generation via elitism, crossover, and mutation.
		
		Returns:
			A JAX array (shape: (pop_size, param_size)) of stacked weight vectors from all individuals.
		"""
		if len(self.pop) == 0:
			self.initPop()
		else:
			# Sort in descending order of fitness
			self.pop.sort(key=lambda ind: ind.fitness, reverse=True)
			elite = self.pop[:max(1, int(0.1 * len(self.pop)))]
			new_pop = []
			while len(new_pop) < len(self.pop):
				parent1 = np.random.choice(elite)
				parent2 = np.random.choice(elite)
				if np.random.rand() < self.p['prob_crossover']:
					child = parent1.crossover(parent2)
				else:
					child = copy.deepcopy(parent1)
				child.mutate(self.p)
				new_pop.append(child)
			self.pop = new_pop
			self.gen += 1
		# Stack each individual's wVec and return as a JAX array
		params_list = [ind.wVec for ind in self.pop]
		return jnp.array(np.stack(params_list, axis=0))

	def tell(self, fitness):
		"""
		Update the fitness of each individual and track the best individual.
		
		Args:
			fitness (list or array): Fitness for each individual.
		"""
		for i, ind in enumerate(self.pop):
			ind.fitness = fitness[i]
			if ind.fitness > self.best_ind.fitness:
				self.best_ind = ind

	@property
	def best_params(self):
		"""
		Return the weight vector (as a JAX array) of the best individual for the Trainer.
		"""
		return jnp.array(self.best_ind.wVec)

	@best_params.setter
	def best_params(self, params):
		pass

# -------------------------------
# Hyperparameters (hyp)
# -------------------------------
# Various parameters controlling the behavior of the NEAT process
hyp = {
	"task": "slimevolley",
	"popSize": 32,            # Population size
	"ann_num_params": 323,    # Must match the policy parameter count
	"ann_absWCap": 5.0,       # Absolute weight cap
	"prob_initEnable": 1.0,   # Probability that a connection is enabled initially (always enabled here)
	"prob_mutConn": 0.8,      # Probability of mutating each weight
	"ann_mutSigma": 0.1,      # Standard deviation of weight perturbation
	"prob_addNode": 0.03,     # Probability of adding a new node (not implemented here)
	"prob_addConn": 0.05,     # Probability of adding a new connection (not implemented here)
	"prob_crossover": 0.8,    # Probability of crossover
	"prob_enable": 0.01,      # Probability of enabling a connection (unused)
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop-size', type=int, default=128, help='ES population size.')
    parser.add_argument(
        '--hidden-size', type=int, default=20, help='Policy hidden size.')
    parser.add_argument(
        '--num-tests', type=int, default=100, help='Number of test rollouts.')
    parser.add_argument(
        '--n-repeats', type=int, default=16, help='Training repetitions.')
    parser.add_argument(
        '--max-iter', type=int, default=500, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=50, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=10, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=123, help='Random seed for training.')
    parser.add_argument(
        '--init-std', type=float, default=0.5, help='Initial std.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--resume', type=str, default="", help='.')
    parser.add_argument(
        '--log_dir', type=str, default="./log/slimevolley", help='.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


def main(config):
    load_checkpoint = False
    log_dir = config.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    else:
        load_checkpoint = True
    load_checkpoint = False
    

    visualize_network(os.path.join(log_dir, 'init.npz'), save_pth=os.path.join(log_dir,  'init.png'))


    logger = util.create_logger(
        name='SlimeVolley', log_dir=log_dir, debug=config.debug)
    logger.info('EvoJAX SlimeVolley')
    logger.info('=' * 30)

    max_steps = 3000
    train_task = SlimeVolley(test=False, max_steps=max_steps)
    test_task = SlimeVolley(test=True, max_steps=max_steps)
	
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        hidden_dims=[config.hidden_size],
        output_dim=train_task.act_shape[0],
        output_act_fn='tanh',
    )
    # breakpoint()
    policy.get_model() # .save(os.path.join(log_dir, 'init_model.npz'))

    # solver = CMA(
    #     pop_size=config.pop_size,
    #     param_size=policy.num_params,
    #     init_stdev=config.init_std,
    #     seed=config.seed,
    #     logger=logger,
    # )
	
    solver = NEATJax(hyp)

    # Train.
    trainer = Trainer(
        model_dir=log_dir if load_checkpoint else None,
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=config.n_repeats,
        n_evaluations=config.num_tests,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
    )
    if not load_checkpoint:
        # breakpoint()
        trainer.run(demo_mode=False)

        # Test the final model.
        src_file = os.path.join(log_dir, 'best.npz')
        tar_file = os.path.join(log_dir, 'model.npz')
        shutil.copy(src_file, tar_file)
        trainer.model_dir = log_dir
    trainer.run(demo_mode=True)

    # Visualize the policy.
    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    action_fn = jax.jit(policy.get_actions)
    best_params = trainer.solver.best_params[None, :]
    key = jax.random.PRNGKey(0)[None, :]

    task_state = task_reset_fn(key)
    policy_state = policy_reset_fn(task_state)
    screens = []
    for _ in range(max_steps):
        action, policy_state = action_fn(task_state, best_params, policy_state)
        task_state, reward, done = step_fn(task_state, action)
        screens.append(SlimeVolley.render(task_state))

    gif_file = os.path.join(log_dir, 'slimevolley.gif')
    screens[0].save(gif_file, save_all=True, append_images=screens[1:],
                    duration=40, loop=0)
    logger.info('GIF saved to {}.'.format(gif_file))
	
    visualize_network(os.path.join(log_dir, 'init.npz'), save_pth=os.path.join(log_dir,  'init.png'))
    visualize_network(os.path.join(log_dir, 'model.npz'), save_pth=os.path.join(log_dir, 'trained.png'))

if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main(configs)
