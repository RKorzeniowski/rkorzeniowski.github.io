# What AI Can Learn From Infants

## Table of Contents
1. [Neuroscience Perspective](#sec_neuro_per)
2. [Developmental Psychology Perspective](#sec_dev_per)
3. [Multiple Discoveries](#sec_mul_disc)
4. [Biologicaly Inspiered](#sec_biological_inspiration)
5. [Trash or Treasure](#sec_uniq_human_int)
6. [Unique Aspects of Machine Intelligence](#sec_uniq_machine_int)
7. [Glimpses of the Future](#sec_future)

The pace of artificial intelligence innovation in 2025 isn’t just accelerating—it’s redefining what’s possible. Imagine AI agents that don’t just follow instructions but orchestrate entire workflows autonomously, like resolving customer service issues end-to-end, processing payments, and even detecting fraud in real time—all while mimicking human reasoning 17. Healthcare is witnessing a revolution, too: passive monitoring devices now track your health metrics 24/7, flagging anomalies like cognitive decline through voice analysis or detecting early-stage diseases with 99% accuracy—sometimes before symptoms even appear 10. Meanwhile, multimodal AI is breaking creative barriers, transforming text prompts into cinematic-quality videos or crafting interactive virtual worlds that adapt to user inputs in real time 46. And for everyday magic, tools like Microsoft Copilot Vision analyze your screen’s content to summarize meetings, prioritize tasks, or even explain complex diagrams—making productivity feel almost effortless 7. This isn’t just progress; it’s a leap into a future where AI isn’t a tool but a collaborator, reshaping work, health, and creativity in ways we’re only beginning to grasp. But to get there we need to solve a few critical problems.

But bridging this gap requires tackling quite a few thorny challenges head-on, to mention some of them. First, the data drought plagues niche domains like rare disease diagnosis and ultra-personalized robotics—where datasets either don’t exist or can’t feasibly scale to the trillion-token appetites of modern models. Second, today’s computational engines struggle to keep pace with the real world’s chaos: autonomous drones still stutter when processing dynamic environments, burning unsustainable energy to analyze shifting variables like weather or crowds. Third, while GPT-4 can write a sonnet, it falters at long-horizon planning—imagine an AI construction manager failing to adjust timelines when a material shortage arises six months into a project. Fourth, even state-of-the-art models harbor inaccurate world models, like believing "turning up a thermostat cools a room" (a real GPT-4 error). Fifth, hallucinations persist as silent saboteurs—medical AIs inventing plausible-but-fake drug interactions or legal tools citing non-existent precedents. Finally, the elephant in the room: backpropagation. This 50-year-old algorithm remains the backbone of modern AI, but its inefficiency and biological implausibility limit systems like Tesla’s Optimus robot, which can’t afford days of training to learn a single new task. Solving these isn’t just academic—it’s the price of admission for an AI future that’s truly robust, adaptive, and aligned with our messy reality.

## Neuroscience Perspective<a name="sec_neuro_per"></a>

While the quest to mimic biological neural systems has driven AI since its inception—from McCulloch and Pitts’ foundational neuron models to today’s deep learning architectures—modern neuroscience is pushing the boundaries of how and why we replicate the brain. Traditional neural networks, including transformers, abstractly mirror synaptic connections, but the field is now diving deeper into biophysical realism and embodied cognition to bridge gaps in adaptability and efficiency. For instance, neuromorphic computing leverages silicon neurons and memristive devices to emulate the brain’s event-driven, energy-efficient computation, enabling systems like Spiking Neural Networks (SNNs) that process temporal data with 100x lower power consumption than conventional GPUs 26. These architectures replicate not just connectivity but also synaptic plasticity, mimicking the brain’s ability to strengthen or weaken connections based on experience—a feature critical for lifelong learning without catastrophic forgetting.

Critically, neuroscience is challenging AI’s reliance on brute-force scaling. The brain’s ability to learn with minimal data and correct errors without full backpropagation—a process mirrored in emerging algorithms like Bayesian synaptic plasticity—is driving research into alternatives to backpropagation, which remains biologically implausible and computationally costly 213. For instance, Tesla’s Optimus robot leverages neuromorphic principles to adapt locomotion in real time, sidestepping days of retraining. The future lies not in slavishly replicating the brain but in distilling its principles: efficiency, adaptability, and embodied intelligence.

## Developmental Psychology Perspective <a name="sec_dev_per"></a>

Compared to neuroscience which studies (definition) developmental psychology studies (definition). While with neuroscience we tried to more or less (maybe I need to introduce some terminiology) replicate from more physical but in case of developmental psychology we could look for replicating biases or mental tools present in human development. 

(Like we always do the eaiest way to solve those types of problems is to cheat and look at what best student in the class is doing. Same thing we always did. You want to learn to fly: try to do the same thing the bird does, ... And humans, particualarly children had to deal with all of the above and more. We might not be able to directly apply all the tricks evolution encoded into our bodies same as the plane is not just a huge bird that swallows the passangers and spits them out on the other side of the globe but we did learn a lot from birds. That's why I think lessons from developmental psychology are a good source of inspiration.)

## Multiple Discoveries <a name="sec_mul_disc"></a>

Discovered first by naure and independently via AI reserachers

To motivate this notion let's name a few examples 
- quantization (VQVAE, human quantization of colors)
- knowledge distilation (cut smallest weights or match features between big and small model teacher guidance, human brain development stage called consolidation)
- inference speed up as final stage (deployment and grapth complilation, myelination)
- need to learn from the very beginning (neural networks can only learn really news stuff when it's in the training data from the start (cite Physics of LLMs, michal paper), critical periods)
- "selective learning" i.e. picking right challange to maximize learning efficency (visual models that go from low resolution simple task to higher resolution harder tasks mimic child development, [Lynda Smith Nips 2024] babies actively look for simple edges to tune thier vision and as they get older start looking for graudally more complex patterns. Kind of unsupervised active learning based on build-in baiases)

## Biologicaly Inspiered <a name="sec_biological_inspiration"></a>

A few examples directly inspierd by nature: 
- neural networks
- attention
- curiosity loss
- LSTM

## Trash or Treasure <a name="sec_uniq_human_int"></a>

(Think if limitations are not a source of development. Like civilizations in cold regions had to be more resourceful and learn to plan while civilizations in hot regions did not have to focus on this aspects and fell behind. Initial disadvantage turning into advantage in the long run)
What Is Not Applicable From Humans to AI.
Disadvantages we don't have to worry about: having a valid system at any moment of development. 
Advandates we can't use amazing body, mental tools, existing experts

## Unique Aspects of Machine Intelligence <a name="sec_uniq_machine_int"></a>

What are unique aspect of how machines learn that what one machine learns can be instantly shared with all of them. That's why it's worth investing crazy ammount of money
into elite model while investing all the resources into a single person would not make sense. 

## Glimpses of the Future <a name="sec_future"></a>

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


## References
- [1] XYZ

