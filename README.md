# Wattsborg

Experiments in computer-generated poetry and music. 
Currently (and tentatively) named "wattsborg" after Isaac Watts, one of the early writers of original hymns in English.


## References
There's a lot of publications and projects out there relevant to computer-generated poetry. Here's just a few. 


### Using Deep Learning

As a general overview (as of late 2018): 
Hobby text generation projects using neural networks got a big boost in 2015 with Andrej Karpathy's 
[blog post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) (and [code](https://github.com/karpathy/char-rnn)) 
about using character-level RNNs to generate Shakespeare. A bunch of projects took that and ran with it. One notable 
recent example is [Max Woolf's textgenrnn](https://github.com/minimaxir/textgenrnn) ([blog post](https://minimaxir.com/2018/05/text-neural-networks/))
which allows you to build a customized and somewhat modernized version of Karpathy's model. It can also do word-level 
embeddings, rather than character-level, which is interesting.  

Some other examples and links in that vein: 
 * http://aiweirdness.com/post/180654319147/how-to-begin-a-song  first lines of songs, using textgenrnn
 * https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms
 * http://warmspringwinds.github.io/pytorch/rnns/2018/01/27/learning-to-generate-lyrics-and-music-with-recurrent-neural-networks/


More recently, a number of people have been exploring GANs (Generative Adversarial Networks) or reinforcement learning techniques for text generation. For example: 

 * 2018: [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/abs/1709.08624) ([code on Github](https://github.com/CR-Gjx/LeakGAN))
 * 2018: [MaskGAN: Better Text Generation via Filling in the ____](https://arxiv.org/abs/1801.07736)
 * 2017: [Language Generation with Recurrent Generative Adversarial Networks without Pre-training](https://arxiv.org/abs/1706.01399) ([code on Github](https://github.com/amirbar/rnn.wgan))
 * https://akshaybudhkar.com/2018/03/26/generative-adversarial-networks-gans-for-text-using-word2vec/


Variational Autoencoders (VAEs) have also been a popular research topic. For example: 
 * 2018: Spherical Latent Spaces for Stable Variational Autoencoders. [[pdf]](https://arxiv.org/abs/1808.10805)[[code]](https://github.com/jiacheng-xu/vmf_vae_nlp)
 * 2018: Variational Attention for Sequence-to-Sequence Models [[pdf]](https://arxiv.org/abs/1712.08207)[[code]](https://github.com/variational-attention/tf-var-attention)
 * 2018: Generating Sentences by Editing Prototypes [[pdf]](https://arxiv.org/pdf/1709.08878.pdf)[[code]](https://github.com/kelvinguu/neural-editor)
 * 2017: A Hybrid Convolutional Variational Autoencoder for Text Generation [[pdf]](https://arxiv.org/abs/1702.02390)[[code]](https://github.com/ryokamoi/hybrid_textvae)
 * 2017: Piecewise Latent Variables for Neural Variational Text Processing [[pdf]](https://www.aclweb.org/anthology/D17-1043)
 * 2017: Toward Controlled Generation of Text [[pdf]](https://arxiv.org/abs/1703.00955)[[code]](https://github.com/wiseodd/controlled-text-generation)
 * 2017: Improved Variational Autoencoders for Text Modeling using Dilated Convolutions [[pdf]](https://arxiv.org/abs/1702.08139)[[code]](https://github.com/ryokamoi/dcnn_textvae)
 * 2016: Generating Sentences from a Continuous Space [[pdf]](https://arxiv.org/abs/1511.06349)[[code]](https://github.com/timbmg/Sentence-VAE)


### Not Using Deep Learning

 * Wattsbot [[code](https://github.com/leahvelleman/wattsbot)]  Notable particularly because she also named her project after Isaac Watts :)
Uses Markov chains. Nicely done, and the output is great.

( a lot more links to come here; this space, especially using Markov chains, has been active for many years)


### Libraries and Utilities

 * Kadot [[code]](https://github.com/the-new-sky/Kadot) NLP library primarily focusing on unsupervised solutions.
 * Texar [[website](https://texar.io/)] [[code](https://github.com/asyml/texar)]
 * pronouncingpy [[code](https://github.com/aparrish/pronouncingpy)] Python interface to CMU's pronouncing dictionary by Allison Parrish. Great for finding rhymes (among other things)
   * also available in a [javascript version](https://github.com/aparrish/pronouncingjs)

### Corpora / Datasets

 * A Gutenberg Poetry Corpus [[code](https://github.com/aparrish/gutenberg-poetry-corpus)] [[corpus](http://static.decontextualize.com/gutenberg-poetry-v001.ndjson.gz)] 
Allison Parrish has done the hard work of extracting and preprocessing all of the public domain English-language poetry in Project Gutenberg. Amazing.
 * pycorpera [[code](https://github.com/aparrish/pycorpora)], a Python interface for Darius Kazemi's Corpora Project, "a collection of static corpora (plural of 'corpus') that are potentially useful in the creation of weird internet stuff."
 * gutenberg-dammit [[code](https://github.com/aparrish/gutenberg-dammit)] Need text? We got your text right here.
 * MetroLyrics 380,000+ dataset: https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics/home
 * LyricsFreak 55,000+ lyrics dataset: https://www.kaggle.com/mousehead/songlyrics/version/1


### Word Embeddings
For this purpose, I prefer (theoretically, at least) word embeddings generated from a source like Project Gutenberg or
perhaps generic web crawl data, rather than those generated from Wikipedia or a news dataset. Why? Because we want our
generated text to sound like fiction, generally, or poetry specifically, not a wiki page or news article. In particular, 
poetry can contain language with significantly more color than the generally more restricted vocabulary of descriptive 
writing.
Some references that may be useful: 

 * [Project Gutenberg and Word2Vec](https://jayantj.github.io/posts/project-gutenberg-word2vec) some experiments with modelling Gutenberg texts and clustering them, with gensim. 
 * [How to Develop a Word-Level Neural Language Model and Use it to Generate Text](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/): Quick overview of how to prepare your own word embeddings, in case we want to generate our own from a specific corpus.
 * Learning Word Vectors from Project Gutenberg Texts  [[code](https://github.com/pat-coady/word2vec)] [[project site](https://pat-coady.github.io/word2vec/)] Another little project to generate word vectors from arbitrary corpora (with Gutenberg texts as an example)
 * [Various Optimisation Techniques and their Impact on Generation of Word Embeddings](https://hackernoon.com/various-optimisation-techniques-and-their-impact-on-generation-of-word-embeddings-3480bd7ed54f)
 * Phonetic Similarity Vectors [[paper](https://aaai.org/ocs/index.php/AIIDE/AIIDE17/paper/view/15879/15227)] [[summary](http://www.exag.org/paper-summaries/#paper07)] [[code](https://github.com/aparrish/phonetic-similarity-vectors/)]] [[talk](https://www.youtube.com/watch?v=L3D0JEA1Jdc)]
Yet another Allison Parrish project. I'm curious how effectively a DL model can learn rhyming words on its own, but if 
it needs help, this is one way to provide phonentic information. Interesting to consider providing phonetic information 
along with semantic information (in the form of more conventionally-generated word vectors) for words...although that 
does mean *very* long input vectors.



### Inspiration 
http://gutenberg-poetry.decontextualize.com/
http://static.decontextualize.com/plot-to-poem.html [[code](https://github.com/aparrish/plot-to-poem/)]
(more to come here... I have dozens of tabs open :) )
