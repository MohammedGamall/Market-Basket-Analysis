# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:04:36 2023

@author: Fahmy
"""

class ToolBox:
    
    import re
    import numpy as np
    import pandas as pd
    
    # Text Cleaning
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    #Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    #Text Preprocessing, splitting and cross testing
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split , cross_val_score

    #ML model for predictions
    from sklearn.linear_model import LogisticRegression

    # Model Evaluation Matrices
    from sklearn.metrics import classification_report, confusion_matrix, jaccard_score
    
    def explore_df(df):
        
        """"input  >dataframe, required column (items)
                output > dataframe shape, columns with null values, describtion for all columns"""
    
        missing_val_count_by_column = (df.isnull().sum())
        
        print("DataFrame Structure Consists of ({}) rows and ({}) Columns \n".format(df.shape[0], df.shape[1]))
        print("Missing values are in columns \n {} \n".format(missing_val_count_by_column[missing_val_count_by_column > 0]))
        print("\t\t\t\t Quick Stats for the DataFrame")
        print(display(df.describe(include='all')))
        print("\n  \t \t \t\t Sample from The DataFrame")
        print(display(df))

    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        num_outliers = sum((df[column] < lower_bound) | (df[column] > upper_bound))
        print(f"Number of outliers for '{column}': {num_outliers}")
        print(f"Lower bound for '{column}': {lower_bound}")
        print(f"Upper bound for '{column}': {upper_bound}")
        
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    def Arabic_txt_Cleaning(column):
        """ input arabic column name 
            output Cleaned list of rows
            process 
            1-extract text into list
            2- iterate through cleaning steps
            3- remove punctuation
            4- remove white spaces
            5-Lemmatization
            """
        import re
        import string
        from nltk.corpus import stopwords
        from farasa.stemmer import FarasaStemmer
        import time,sys
        
        start_time = time.time()
        stemmer = FarasaStemmer(interactive=True)
        corpus = []
        hashes = ["كجم","جم","جرام","مل","ملل","مللى","لتر","ليتر","عبوة","قطعة","علبة","كرتونة","كيلو","ك","قطع","جامبو", "قطعتين","متر","سم","–",",","ابيض","بكرات","كيس","زجاجة","عبوات","عبوتين","وسط","صغير","كبير","لارج","×"]
        text = list(column)
        
        total = len(text)
        point = total / 100
        increment = total *0.01
        
        for i in range(len(text)):
            #Excluding numeric digits &  English alphabets
            r = re.sub("[0-9]", '', str(text[i]))
            r = re.sub("[a-zA-Z]",'',r)
            r = re.sub("([×|–|،])", "", r)
            #Excluding punctuations
            r = r.translate(str.maketrans(string.punctuation, ' '*32))
            r = r.split(' ')
            #Excluding white extra spaces
            r = [i.strip() for i in r if i != '']
            #Removing stopwords
            r = [word for word in r if word not in stopwords.words('arabic')]
            #Removing out of context words (Confounds variables)
            r = [word for word in r if word not in hashes ]
            #Stemming words accoring to Farasa engine
            r = [stemmer.stem(word) for word in r]
            r = ' '.join(r)
            corpus.append(r)
            if(i % (5 * point) == 0):
                sys.stdout.write("\r[" + "█" * int((i / increment)) +  " " * int(((total - i)/ increment)) + "]" +  str((i / point)+5) + "%")
                sys.stdout.flush()
        #Terminate stemming engine for resources optimization
        stemmer.terminate()
        print("--- %s seconds ---" % (time.time() - start_time))
        return corpus
    
    def confusion_Matrix_map(y_test, pred):
        
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        
        label = np.unique(y_test)
        df = pd.DataFrame(confusion_matrix(y_test,pred,labels=label),index= label, columns=label)
        
        class_names=label # name  of classes
        
        fig, ax = plt.subplots()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        
        # create heatmap
        sns.heatmap(df, annot=True, cmap="YlGnBu" ,fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')    
        plt.Text(0.5,257.44,'Predicted label');
        plt.show()
        
    def clean_json_dict(column):
        """input dic {'ar':text,'en':text} 
        output english text"""
        cleaned = column.apply(lambda st : st[st.find("\"en\":")+6:st.find("\"}")])
        return cleaned

        
