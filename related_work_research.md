# Music Genre Classification Using Acoustic Features and Autoencoders (2021)
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9598979

* using digital signal processing techniques and autoencoders
* GTZAN dataset has been used.
* Aim is to compare the digital signal processing technique to using autoencoders.
* To train the autoencoder - Mel Frequency Cepstral Coefficients (MFCC) are used
* * https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd
* The autoencoder is a CNN (which they describe in Figure 1)
* They use the latent space representation (ie the bit between the encoder and decoder) to classify and cluster
* "Therefore it is seen that using just autoencoder has not made any improvement rather it gives less accuracy"

* Essentially they're trying to see if you can use the features picked out by an autoencoder to classify the music, and they find that no this is not possible even with latent spaces of varying sizes.
* They did reference the 2020 Elbir paper for a similar technique

# Music genre classification and music recommendation by using deep learning (2020 - Elbir)
https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/el.2019.4202

* This paper has lots of references to other research done with the GTZAN dataset
* Came up with MusicRecNet
* This study wanted to focus on both music genre classification AND music recommendation (the latter isn't focused on so much in other papers)
* MusicRecNet can detect plagiarism 🤨
* In their MusicRecNet they use different types of classifiers and then evaluate the performance of each. (given in Table 2)
* * We can see that some perform a lot better than others
* "When the performance results are examined, some similar music genres can lead to mis-classification and mis-recommendation such as Jazz and Classic"
* MusicRecNet has 3 layers and has considerably better accuracy scores


# Music Genre Classification using Machine Learning Techniques (Bahuleyan 2018)
https://arxiv.org/pdf/1804.01149.pdf

* Compares two models - deep CNN and one feeding features in the time domain and frequency domain to ML models such as Random Forest, SVM etc.
* Uses 'Audio Set' dataset
* Again this paper has a good literature review
* (This paper is very good at explaining the CNN architecture and we can use this as inspiration for describing ours)
* Uses transfer learning and fine tuning
* * Found that fine tuning setting gave v slightly better results than transfer learning (0.63 to 0.64)
* * need to research this more if this is a paper we choose to use later
* So for CNN it is pretty much as we have used it
* For the hand-crafted features they pick out quite a few features, and then report which contributes the most, having ran it through 4 ML models.

* Conclusion was that CNN was better and fine tuning CNN didn't give much better results than transfer learning CNN 🤯


# Music Genre Classification using Transfer Learning on log-based MEL Spectrogram 
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9418035
* Mentions a bunch of different methods from different papers. One such paper SVMs were the best classifiers for Content Based Music Classification. 
* Proposing a transfer learning approach : "The transfer learning option helps speed up the learning process since the core features of the image have already been properly learned by the model and so the generalization during the fine tuning can focus more on the spectrogram for better classification to take place"
* Transfer learning speeds up the training process and makes performance of deep learning models better. 
* Some of the training & metrics approaches could be useful such as a discriminitive learning rate 

# Exploring Data Augmentation to Improve Music Genre Classification with ConvNets
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8489166
* They cover: noise addition, pitch shifting, time stretching and loudness variance. Pitch shifting and time stretching would fall under extension 1 but we could consider doing noise addition or loudness variance
* They use the Latin music database
* One paper mentioned achieved their highest accuracy by applying data augmentation techniques in both the training and test sets
* Artist filter concept could be something we mention in the report as a limitation. This basically is a method of ensuring that songs from the same artist are not found in the training set and the test set so that the classifier doesn't become a classifier of author recognition rather than genre recognition. We do not have access to author information to be able to do this.
* This paper does 3-fold cross val and present the average accross the folds and standard deviation 
* Out of the two options: noise addition and loudness variance, loudness variance produces better results.
* Mentions sum rules and product rules and these seems to affect accuracy but idk what they are rly

# Extension ideas:
* Autoencoder
* Using different classifiers - currently we're using ?MLP? but trying out SVM or LDA
* Data augmentation that is not pitch shifting or time stretching (sadface because pitch shifting is best)
