
# Bayes and Bach: An RNN-powered affair #

This is the repository for the project hosted in  [BrainHack Donostia 2024](https://brainhack-donostia.github.io).


## Project Aim ##

We want to do a proof-of-concept: test whether artificial recurrent neural networks (RNN) learn to predict the next tone when trying to perceive music in a noisy environment.

### The Bayesian bit ###

Perceiving in noisy environments is tricky: noise occludes part of the information we need to create a truthful percept of what we are hearing. The optimal way to *fill in the missing information* is to combine what we are hearing with everything we have learnt in the past. This is, informally, the principle behind Bayesian Inference: our posterior (after we heard the sound) believes on what just happened are a combination of what we have heard and our prior (before we hear the sound) beliefs of what will happen. The Bayesian Brain Hypothesis proposes that this is actually what the brain does to perceive and interact with the world: constantly compute the priors on what will happen next to power optimal inference on our surroundings. But how can we test this is actually what's happening?

### The Bachian bit ###

Enters Bach! We will use Bach's compositions as a synechdote for auditory perception. Although our daily audition requires doing inference on structures that can be much more complex than that (e.g., speech), Bach's music has the essential basic elements: a hierarchical structure that can be exploited to generate priors on what will happen next. Since this is a proof-of-concept, we will use a single composer to make sure predictable patterns are relatively easy to detect.

### The power of the RNN ###

We could recruit a bunch of newborns, force them to listen to Bach non-stop for their first 3 years of live, and then check whether their brains learned to make predictions on what will happen next; however, that would take too long. Instead, we will train recurrent neural networks: an oversimplistic but nevertheless useful metaphore of a (tiny) brain. RNNs have a neural-inspired architecture, states that can be used to inspect their internal working strategies, and are extremely flexible in the computational strategies they can learn. Despite the obvious differences with the biological brain, studying how RNNs learn to solve problems is a reasonable way to generate hypotheses on how the brain does it.

### Overview of the project ###

We will start by collecting a lot of Bach's compositions. We will transform each composition into a sequence of tokens: each token will be one note or several notes (i.e., a chord) that were meant to be played together in the original composition. For this proof-of-concept we will ignore information about the duration or loudness of each note/chord. For each sequence, we will generate two items: a ground-truth (the sequence itself), and a set of observations (the sequence plus some noise that will corrupt part of the information).

We will then present the observations to the RNN and train its weights so that, for each of the observation sequences, the RNN is able to extract the ground-truth sequence. In other words: we will train the RNNs to perceive the original musical sequence usign only the noisy observations.

After the training we will test whether the RNN is encoding, in its internal states, a set of predictions on the next token. If that is the case, we will conclude that the RNN is actively attempting to predict the next item to implement perception.

The results will help us generating hypotheses on how the brain decodes music. The neuroscientific community will be able to use these hypotheses to inform the design of experiments IRL brains.


## Work Plan ##

We will work in three parallel lines of work: data retrieval, data preprocessing, and RNN training. 

### Data retrieval ###

Data Retrieval will be responsible for the generation of a database of midi files encoding Bach's compositions. Depending on the skills of the retriever(s), this can be fully automatised using web scrappers (preferred), or done by hand (not preferred). Web scrapping would be very easy to learn for anyone with basic notions of python, and a very valuable skill for future projects!

**Resources on MIDI files:** [This website](http://www.jsbach.net/midi/index.html) could be a good place to start. [This other website](http://www.jsbach.net/bcs/link-midi.html) has a lot of links with potentially better quality MIDI files. The better the quality of the MIDI, the easier the work of the preprocesser(s)!

**Resources on data scrappers:** I used [this tutorial](https://www.geeksforgeeks.org/python-web-scraping-tutorial/) when I needed to implemet a scraper in the past. This [other tutorial](https://realpython.com/python-web-scraping-practical-introduction/) also looks informative and easy to follow.

### Data preprocessing ###

Data Preprocessing will use the library of MIDI files to generate a dataset of the RNN training. The resulting code should list all the MIDI files and, for each file, generate a sequence of ground-truth note/chords and a sequence of observations. Observations should be generated with different levels of noise. Pairs of ground-truth and observation sequences should be accessible via a function call.

The ground-truth and observation output sequences should be np-arrays of dimension `(N_notes x N_tokens)` where `N_notes = 128` is the number of possible notes in a MIDI file and `N_tokens` is the number of notes/chords in the composition (and thus depends on the composition).

Data Preprocessing can start developing their code before Data Retrival has finished generating the database using [this MIDI file](http://www.jsbach.net/midi/bwv988/988-v01.mid).

**Resources on how to read MIDI files with python:** I found [this tutorial](https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c) very easy to follow.

### RNN training ###

RNN Training will code a trainable RNN architecture with GRU units and one using pytorch. The network will have a linear input `N_tones -> N_units` and a linear output `N_units -> N_tones`. The RNN Trainer(s) will code a loss function that measures the difference between the network's output and the ground-truth target, and adjust the hyperparameters of the network and the training.

RNN Training can start developing their code before Data Retrival has finished generating their code by randomly generated a sample with the same data structure as the Data Preprocessing output.

Once the code is ready, training should be run over 80% the available dataset; validation of the training (as measured by perception accuracy) should be run over the remaining 20% of the dataset.

**Resources:** If you are unfamiliar with pytorch, start [here](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html). Pytorch includes an [implementation of an RNN with GRU units](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html). 

### Testing ###

After training, all participants can be invoved in the testing of the main hypothesis. The first step is to add a second linear readout to the RNN. This linear readout will be trained to predict the next token rather than on the perception of the current one. During the training, the internal weights of the RNN should be frozen to ensure we are only training the linear readout.

Data will be again split into training (80%) and testing (20%) partitions. We will then train the linear readout on the training set, and measure the prediction accuracy on the testing set. We will determine that the network encodes predictions if the performance of the trained model is higher than baseline.

Baseline performance will be computed in parallel as the performance of a model that always predicts the average global distribution of tones.

### Milestones ###

- A web scrapper that collects MIDI files of Bach's compositions
- A collection of MIDI files of Bach's compositions
- A MIDI-to-data pipeline that transforms MIDI files in pairs of (ground-truth, observation) that can be used for training
- Several RNN architectures designed to retrieve the ground truth from noisy observations of sequences of tones/chords
- The trained RNN(s)
- Validation/refutation of the starting hypothesis

### Extensions ###

This project should keep us busy for the full brainhack; however, if we manage to have some time left, there are a few cool things we could look into:

**Generalisation to other composers:** How does the network perform on inferring sequences from other Baroque composers? And composers from earlier (Renaissance) or posterior (Classical, Romantic) periods? How does it perform on inferring Jazz? And pop music?

**Encoding of context:** Does the network develop an internal representation of the tonal key of the composition to aid inference? If that was the case, we should be able to decode the key of the composition from the internal states of the network. 


## Teams and participants ##

This is an accessible project that does not require an extensive coding expertise. Participants that want to contribute with coding will be recommended to do this in python, although data retrieval can be done in any other language. Participants who do not wish to contribute with code can contribute either with a) their musical expertise in Data Preprocessing and during the interpretation of the Testing, or b) their patience and web-browsing skills to manual data retrieval.

### Data retrieval ###

Skills level: python beginners / non-coders.

Manual retrieval welcomes all participants, although automatising this with a web scrapper will be much more fun and will grant the participants a new skill. This task is perfect for beginners who want to train their freshly acquired python abilities. Once data retrieval is finished participants will be encourage to help the Data Preprocessing team. Non-coding participants can help coding participants by identifying the fields to scrap in the websites, curate the download database, etc. 

### Data preprocessing ###

Skills level: python and numpy beginners; non-coding musicians.

Data preprocessing will be more challenging that retrieval, but should be accessible to participants that have recently started their python jorney. The main challenge will be robustly mapping the ecclectic MIDI formats into robust sequences. Understanding the musical structure of the tones would be helpful but it is not strictly necessary. Usage of numpy will be fairly basic and could be learnt during the development of the project.

### RNN training ###

Skills level: mid-experienced numpy users / pytorch beginners.

Pytorch provides for objects coding GRU-RNNs, computing the optimisation steps, etc: the only challenge of this task is to use pytorch itself. A fairly-experienced numpy user should be able to catch up with the basics in a few hours. Participants with previous pytorch experience will enjoy practicing partial training of the models. Non-coding participants with theoretical knowledge of machine learning could be helpful on the tuning of the model and optimisation hyperparameters. 


## Project communication Mattermost channel ##

https://mattermost.brainhack.org/brainhack/channels/bh-donostia-2024--bayes-3-bach
