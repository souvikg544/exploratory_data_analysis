This project uses web scraping to collect the recent news on a topic, and renders a confidence score to the readers about the correctness of the news. Here, concepts of NLP have been used for preprocessing the data, that include removing stopwords, lemmatizing and stemming. The vocabulary consists of vectors of all words that have been used in the dataset, also known as bag of words model.

This project is divided into two parts
1. Loading the dataset, and training the convolutional neural network model, and storing the weights in the form of pickle file.
2. Scraping the news and testing the trained model.

Both the tasks mentioned have been implemented above, in different files. 

