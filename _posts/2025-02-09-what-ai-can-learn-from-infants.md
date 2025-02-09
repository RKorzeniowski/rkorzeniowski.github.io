# What AI Can Learn From Infants

`Date: February 09, 2025 | Author: Renard Korzeniowski `

## Table of Contents
1. [Introduction](#introduction)
2. [Some paragraph](#paragraph1)
    1. [Sub paragraph](#subparagraph1)
3. [Another paragraph](#paragraph2)

AI models are amazing and can do cool stuff already but to unlock thier full potential there are a few problems that we need to overcome. 

First lets expand a bit on all of the above. AI models are amazing because chatbots, image generation, zeroshot task solving via prompting. 
By the full potential I mean 
1. Multimodality: allowing models to interact with a world in a more similar way to humans since sight and hearing will unlock new functionality that was not 
possible with only text reasoning or only image reasoning
2. Agents: instead of humans doing all the planing as well as verificantion and only relying on LLM to peform well defined simpled tasks we will be able to provide 
high level goals
3. Embodiment: allowing AI to interact with phisical world via some form of a body. The effects on blue collar sector will be similar to the impact of automation on manufacturing.
Invididual will become increadbly more productive. If well organized leading to prosperity and ability to focus on even greather achivments. 

Having that said problems like 
1. Lack of data both in terms of it not existing and no real feasability of creating such datasets suitable for our current generation of models,
2. Low computational efficiency needed to process dynamic world around us,
3. Planning far into the future and correcting errors madealong the way, inaccurate word models, hallucinations, social aspect, learning online (no need for backprop). 

## Developmental Psychology

Like we always do the eaiest way to solve those types of problems is to cheat and look at what best student in the class is doing. Same thing we always did. You want to learn to fly: try to do the same thing the bird does, ... And humans, particualarly children had to deal with all of the above and more. We might not be able to directly apply all the tricks evolution encoded into our bodies same as the plane is not just a huge bird that swallows the passangers and spits them out on the other side of the globe but we did learn a lot from birds. That's why I think lessons from developmental psychology are a good source of inspiration. 
To motivate this notion let's name a few examples quantization (VQVAE, human quantization of colors), knowledge distilation (cut smallest weights or match features between big and small model teacher guidance, human brain development stage called consolidation), inference speed up as final stage (deployment and grapth complilation, myelination), critical periods (neural networks can only learn really news stuff when it's in the training data from the start (cite Physics of LLMs, michal paper), critical periods), "selective learning" i.e. picking right challange to maximize learning efficency (visual models that go from low resolution simple task to higher resolution harder tasks mimic child development, [Lynda Smith Nips 2024] babies actively look for simple edges to tune thier vision and as they get older start looking for graudally more complex patterns. Kind of unsupervised active learning based on build-in baiases), ...

Examples directly inspierd by nature: attention, curiosity loss, LSTM

What could we learn from dev psy. 
- [Lynda Smith Nips 2024] ideas about episodic learning, 
- leaning more on selective learning and making it more active, (check what other things were said). 
- feedback loops for adjusting task complexity
- "allowing models to explore the real world" rather than using static data. humans are not passive percivers but active explorers
- Copy study techniques to understand model behavior and compare with human learning biases (examples of hallucinations in humans)
- U-shaped developmental pattern (expert models as training wheels in early stages of training)
- intermodal integrations: build-in from very early on in humans and very important (check more on that)
- infant reason differently about social and non-social agents. infantes interpret actions of other social agents in terms of goals. They also encode events in terms of goals but only when it's performed by social agent
(attachment as basis of children social bonds?)
- Two systems 
  - explicit cognition: involves having awarness of knowledge or of thought process. Usually being able to describe it in words
  - implicity cognition: works outside of awarness and is hard to describe in words e.g. knowing how to ride a bike or feeling that animal is sick
- (learning as a group with hidden information rather than individuals?)


What is not applicable from humans to AI. 
Disadvantages we don't have to worry about: having a valid system at any moment of development. 
Advandates we can't use amazing body, mental tools, existing experts


What are unique aspect of how machines learn that what one machine learns can be instantly shared with all of them. That's why it's worth investing crazy ammount of money
into elite model while investing all the resources into a single person would not make sense. 


```math
F(y) = C(y, \hat{y}) + \beta D_{KL}[\mathcal{N}(\mu,\,\sigma^{2}) \vert| \mathcal{N}(0, 1)] = C(y, \hat{y}) \dfrac{\beta}{2}\sum_{i=1}^d \sigma^{2}_i - log(\sigma^{2}_i) - 1 + \mu_i^2
```

```math
\underset{\lVert\vec{v}\rVert=1}\max \nabla f(a_1, b_1) \cdot \vec{v}
```


## This is the introduction <a name="introduction"></a>
Some introduction text, formatted in heading 2 style

## Some paragraph <a name="paragraph1"></a>
The first paragraph text

### Sub paragraph <a name="subparagraph1"></a>
This is a sub paragraph, formatted in heading 3 style

## Another paragraph <a name="paragraph2"></a>
The second paragraph text
