# Email-Spam-Classification-using-SVM

The codes above help to classify a given mail as a spam or non spam.

Directions of use:

1. Download the emails dataset from https://tinyurl.com/y93s2kcm and copy all the emails from into a folder named "emails" which should be presented at the same place where these codes are downloaded. The dataset has been formed using emails downloaded from http://www2.aueb.gr/users/ion/data/enron-spam/ website. The data consists of around 10k emails.
2. First run the code extracting_unique_words_from_all_emails.py which helps to make file wordslist.csv which contains the unique words found in emails which occur more than or equal to 100 times in all emails.
3. Next run processing_emails_to_find_occurance_of_words.py to find the occurance of each word in the emails dataset and make frequency.csv file which stores the frequency of each important word present in wordlist.csv in every email.
4. Finally run the svm_implementation.py code to study the predicted results of Support Vector Machine Algorithm on the dataset.
