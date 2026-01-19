# World Models

## What Are World Models

A world model is a system that learns to represent how an environment evolves over space, time, and under actions.
Typically, it predicts future states, observations, or rewards conditioned on past observations and actions.

World models are widely used in reinforcement learning, simulation, planning, and robotics. More recently, they have attracted renewed attention with the release of Genie-3, which can generate rich, interactive environments from short textual prompts—blurring the line between generative modeling, simulation, and embodied intelligence.

At a high level, world models combine three core ideas:

- Representation learning – compressing high-dimensional observations into useful latent states
- Dynamics modeling – learning how those latent states evolve over tim
- Action conditioning – modeling how an agent’s actions influence future states

## Why Are World Models Exciting

World models enable qualitatively new capabilities that go beyond traditional perception or control:
- Embodied reasoning: agents can reason about consequences of actions by simulating futures internally
- Learning by imagination: policies can be trained inside a learned model without expensive real-world interaction
- Data efficiency: reusing experience via simulation drastically reduces sample complexity
- Generalization: abstract latent dynamics can transfer across tasks and environments

Recent results show world models scaling to hundreds of tasks, handling long horizons, and integrating with language and multimodal models, suggesting they may be a core building block for general-purpose agents.

## Background

Before diving deeper into world models, it is useful to briefly review a few foundational concepts.
The goal here is intuition and context, not a full tutorial—links are provided for deeper study.

### Representation Learning

At the heart of any world model is a latent representation of the environment.

Raw observations—images, video frames, sensor streams—are typically high-dimensional and noisy. World models rely on representation learning to compress these observations into a latent space that is:

- Compact – lower dimensional than the raw input
- Predictive – sufficient to model future evolution
- Action-relevant – captures factors affected by agent actions

In practice, this often involves:

- Convolutional neural networks (CNNs) to extract spatial features from images
- Autoencoders or VAEs to learn stochastic, compressed latent states
- Recurrent or attention-based models to maintain temporal context

Unlike representation learning for classification, the objective here is not discriminative performance but temporal consistency and predictability. A good latent state is one that makes the future easy to model.
This perspective naturally connects representation learning with video prediction and generative modeling.

### Reinforcement Learning

World models are most commonly studied in the context of reinforcement learning (RL).

In RL, an agent interacts with an environment modeled as a Markov Decision Process (MDP), where:
- The environment has a (possibly hidden) state
- The agent takes actions
- The environment transitions to a new state and emits a reward

Traditional RL approaches fall into two broad categories:
- Model-free RL: directly learns a policy or value function from experience
- Model-based RL: learns a model of the environment and uses it for planning or policy learning

World models belong to the second category.

Instead of modeling the environment in observation space, modern approaches learn a latent dynamics model. This model can then be used to:
- Roll out imagined futures
- Evaluate candidate actions
- Train policies entirely inside the learned model

A concise refresher on RL concepts can be found in:
- Lilian Weng’s overview: https://lilianweng.github.io/posts/2018-02-19-rl-overview/
- UC Berkeley’s CS285 lectures on model-based RL
For world models, the key insight is that learning a good dynamics model can dramatically simplify policy learning.

### Dynamics Modeling

Dynamics modeling focuses on learning how latent states evolve over time, typically conditioned on actions.
Common approaches include:
- Recurrent neural networks (e.g., GRUs, LSTMs)
- Stochastic state-space models
- Transformers for sequence modeling
Because real environments are stochastic and partially observable, modern world models usually incorporate uncertainty, predicting distributions over future states rather than single deterministic trajectories.

## Evolution of World Models

(Add diagram here illustrating encoder → latent dynamics → decoder / policy)

The paper “World Models” (Ha & Schmidhuber, 2018) is one of the earliest works that clearly illustrates the concept.
The model consists of three components:
- A Variational Autoencoder (VAE) that encodes images into a compact latent representation
- A recurrent neural network (RNN) that models temporal dynamics in latent space
- A controller that maps latent states to actions

A particularly striking idea in this work is that the agent can be trained inside the learned world, using imagined rollouts rather than real environment interaction.

“Recurrent World Models Facilitate Policy Evolution” extends this idea and provides insights into how compact policies can operate effectively when paired with learned latent dynamics.

“Contrastive Learning of Structured World Models” introduces compositional structure, explicitly modeling objects and relations using contrastive learning. This direction connects world models with object-centric representation learning and relational reasoning.

More recent work such as “Mastering Diverse Control Tasks through World Models” demonstrates that these ideas can scale to many tasks, long horizons, and complex environments, suggesting world models are not just toy examples but practical foundations for general agents.


### Further Resources:
- World Models project page: https://worldmodels.github.io
- NeurIPS tutorial: https://nips.cc/virtual/2023/tutorial/73952
- AAAI 2024 tutorial: https://sites.google.com/view/aaai2024worldmodel/home
- Curated paper list: https://github.com/leofan90/Awesome-World-Models
- Hands-on implementations: https://github.com/google-research/world_models

### Key Papers:
- World Models (2018)
- Recurrent World Models Facilitate Policy Evolution (2018)
- Contrastive Learning of Structured World Models (2019)
- Genie-3 (2024)
