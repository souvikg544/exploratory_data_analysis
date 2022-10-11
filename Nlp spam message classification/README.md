# NLP-spam-detector
## Click Here https://spam-detector-ankur.herokuapp.com/ to redirect to website where it is hosted 

### Natural Language Processimg 
**Approach** - I have data of some text msg . Some are spam and some are not . After cleaning and other preprocessing , i saparate out spam and non - spam msg . 
Then we will extract words that are common . Actually probability of being  spam and non spam is calculated over this 

**Concept** - As stated concept used is _**naive bayes algorithm **_ in Simple way <br/>

![image](https://user-images.githubusercontent.com/80956033/152722731-5d475a31-c586-4b8e-bc97-7eeb8c5bc497.png)
<br/>
P(spam) = no. of spam / total  <br/>
        = 0.13 
This mean 13% of total mail generally found as spam <br/>

P(text| spam)- all logic lies here <br/>
dictionary with    key = word     and   value = probability  of occurance  is created  for both spamer and non- spammer word bag <br/>


we will check each work of text entered by user and multiply all to determine spam-probability and non spam probability . Then compare and provide result . <br/>
Example = " he  is good  spammer "<br/>
we will multiply probability value ; Both spammer and non -spamer 
then comapre 
