# Import necessary packages
import pandas as pd
import xml.etree.ElementTree as ET
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import spacy
from matplotlib import pyplot as plt
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.simplefilter("ignore")
nlp = spacy.load('en_core_web_lg')

def xml_to_df(filename):
    # Parse xml file into pandas data frame to work with at the sentence level
    tree = ET.parse(filename)
    root = tree.getroot()
    reviews = root.findall('Review')
    df_columns = ['rid','id','text','category','predicted_category','polarity','predicted_polarity']
    reviews_df = pd.DataFrame(columns=df_columns)

    for review in reviews:
        rid = review.get('rid')
        sentences = review.findall('sentences/sentence')
        for sentence in sentences:
            id = sentence.get('id')
            text = sentence.find('text').text
            opinions = sentence.findall('Opinions/Opinion')
            for opinion in opinions:
                category = opinion.get('category')
                polarity = opinion.get('polarity')
                predicted_category = ''
                predicted_polarity = ''
                reviews_df = pd.concat([reviews_df, 
                            pd.DataFrame([[rid, id, text, category, 
                            predicted_category, polarity,predicted_polarity]],
                            columns=df_columns)], ignore_index=True)
    
    return reviews_df

train_reviews_df = xml_to_df(filename='Laptops_Train_p1.xml')
train_reviews_df.head()

# Creating a copy of original data frame to populate with processed text
processed_df = train_reviews_df.copy()

def tokenise(text):
    # Tokenising sentences
    tokenised_text = [word_tokenize(sentence.lower()) for sentence in text]
    return tokenised_text

def remove_stopwords(tokenised_text):
    # Remove stop words
    tokens = []
    for token in tokenised_text:
        if token not in stopwords.words('english'):
            tokens.append(token)
    return tokens

def remove_non_alpha(tokenised_text):
    # Remove punctuation
    alpha_tokens = []
    for token in tokenised_text:
        if token.isalpha():
            alpha_tokens.append(token)
    return alpha_tokens

def stem(tokenised_text):
    # Stem tokenised text
    snow_stemmer = SnowballStemmer(language='english')
    stem_tokens = []
    for token in tokenised_text:
        stem_tokens.append(snow_stemmer.stem(token))
  
    stemmed_text = " ".join(stem_tokens)
    return stemmed_text

def preprocess(tokenised_text):
  # output processed text
  pp_data = []
  for sentence in tokenised_text:
    pp_text = remove_stopwords(sentence)
    pp_text = remove_non_alpha(pp_text)
    pp_text = stem(pp_text)
    pp_data.append(pp_text)
  return pp_data

processed_df['text'] = preprocess(tokenise(train_reviews_df['text']))
processed_df = processed_df.rename(columns={"text": "processed_text"})
processed_df.head()

def e_a_predict(features, X_train_counts, model, threshold):
    # Returns the next most likely topic if a given sentence has multiple categories
    # if probability for next most likely topic is > threshold, else return most likely topic.

    N = len(features)
    tmp = 0
    repeat = 0
    predictions = np.zeros(N)
    for i in range(N):
        if features[i] != tmp:
            predictions[i] = model.predict(X_train_counts[i,:])
            repeat = 0
        else:
            repeat += 1
            arr = model.predict_proba(X_train_counts[i,:])
            sorted_index = np.argsort(arr)[0]
            if arr[0][sorted_index[-repeat-1]] > threshold/repeat:
                predictions[i] = float(sorted_index[-repeat-1])
            else:
                predictions[i] = model.predict(X_train_counts[i,:])
            # handle part 2 error where >8 opinions for single review gives index error
            if repeat > 7:
                repeat -= 1
        tmp = features[i]
    
    return predictions

def sorted_predictions(df):
    # Aligns predictions with matching labels for sentences that have multiple opinions.
    # e.g., ground truth for sentence id=1: LAPTOP#GENERAL, LAPTOP#BATTERY_PERFORMANCE
    # predictions for sentence id=1 pre-alignment: LAPTOP#BATTERY_PERFORMANCE, LAPTOP#GENERAL
    # predictions post-alignment: LAPTOP#GENERAL, LAPTOP#BATTERY_PERFORMANCE

    N = len(df['category'])
    labels_dict = {}
    predictions_dict = {}
    sorted_predictions = []
    i = 0
    tmp = 0

    for id in df['id'].unique():
        labels_dict[id] = list(df.query(f"id == '{id}'")['category'])
        predictions_dict[id] = list(df.query(f"id == '{id}'")['predicted_category'])

    for key in labels_dict.keys():
        for value in labels_dict[key]:
            if value in predictions_dict[key]:

                idx = labels_dict[key].index(value) # obtain label index for matching label & prediction for given sentence id
                idx2 = predictions_dict[key].index(value) # obtain prediction index for matching label & prediction for given sentence id

                tmp = predictions_dict[key][idx]

                predictions_dict[key][idx] = value # re-order predictions so that they align with ground truth for given id
                predictions_dict[key][idx2] = tmp # swap changed values in predictions list for sentence id
    
    for values in predictions_dict.values():
        for value in values:
            sorted_predictions.append(value)

    return list(sorted_predictions)

def numerical_entity_attributes(df):
    # Numerical representation of predicted categories is required for accuracy
    # and classification report. Function converts category predictions to
    # numerical representation

    category, predicted_category = df['category'], df['predicted_category']

    e_a_list = category.unique().tolist()
    i = 0
    e_a_label_dict = {}
    for e_a in e_a_list:
        e_a_label_dict[e_a] = i
        i += 1

    e_a_labels = []
    for i in range(len(category)):
        e_a_labels.append(e_a_label_dict[category[i]])

    e_a_predictions = []
    for i in range(len(category)):
        if predicted_category[i] in e_a_list:
            e_a_predictions.append(e_a_label_dict[predicted_category[i]])
        else:
            e_a_predictions.append(99) # predicted E#A pair not in ground truth set of E#A pairs

    e_a_labels = np.array(e_a_labels, dtype=int)
    e_a_predictions = np.array(e_a_predictions, dtype=int)
    e_a_list.append('N/A')

    return e_a_labels, e_a_predictions, e_a_list, e_a_label_dict

def reverse_dict(my_dict):
    # Reverses the keys and values in a dictionary, useful for several later steps
    reversed_dict = {}
    for key, value in my_dict.items(): 
        reversed_dict[value] = key 
    return reversed_dict

def numerical_labels(df, label_name, new_column_name=str()):
    # Convert ground truth word labels for entities or attributes to numerical representation
    labels_list = df[label_name].unique().tolist()
    df[new_column_name] = ''
    i = 0
    labels_dict = {} # Will use this to convert numerical class predictions back to attributes
    for label in labels_list:
        df.loc[df[label_name] == label, new_column_name] = i
        labels_dict[i] = label
        i += 1
    
    return labels_list, labels_dict

def convert_numerical_predictions(num_predictions, label_dict):
    # Convert numerical predictions back to words
    num_predictions = num_predictions.tolist()
    word_predictions = []

    for pred in num_predictions:
        word_predictions.append(label_dict[pred])
    
    return word_predictions

def combine_entity_attributes(df, entity_pred, attrib_pred):
    # Combine Entity and Attribute predictions to create E#A pair predictions
    # Sort E#A pair category predictions
    combined_list = []
    for i in range(len(df['category'])):
        e_a_pair = entity_pred[i] + "#" + attrib_pred[i]
        combined_list.append(e_a_pair)
    
    return combined_list

def overall_accuracy_absa(df):
    # Returns an overall accuracy score as a % of the instances where both the
    # category predictions AND the sentiment predicitons were correct.
    N = len(df['category'])
    correct_count = 0
    for i in range(N):
        if df['category'][i] == df['predicted_category'][i] and df['polarity'][i] == df['predicted_polarity'][i]:
            correct_count += 1
    
    accuracy = (correct_count/N)*100
    output = f"Overall accuracy for correct category and sentiment predictions: {accuracy:.0f}%"

    return output

X_train = processed_df['processed_text']
# Using counts to extract features from text
count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)

# Separate entities from categories and add to new column
processed_df['entity'] = [entity.split('#')[0] for entity in processed_df['category']]

# label entities
entities_list, entity_label_dict = numerical_labels(processed_df, 'entity', 'entity_label')

# Training set for entity classification
Y_entity_train = processed_df['entity_label']
Y_entity_train = np.array(Y_entity_train, dtype=int)

# Train the entity classification Logistic Regression model
entity_count_lr = LogisticRegression()
entity_count_lr.fit(X_train_counts, Y_entity_train)


# Testing entity prediction on training data... overfitting likely
Y_entity_train_pred = e_a_predict(X_train,X_train_counts,model=entity_count_lr,threshold=0.1) # threshold hyperparameter tuned on training data
entity_train_accuracy = accuracy_score(Y_entity_train, Y_entity_train_pred)

print(f"Training set entity predicitons accuracy score: {entity_train_accuracy*100:.0f}%")

# Convert entity predictions from numbers back to words
entity_predictions = convert_numerical_predictions(Y_entity_train_pred, entity_label_dict)

# Separate attributes from categories and add to new column
processed_df['attribute'] = [attribute.split('#')[1] for attribute in processed_df['category']]

# label attributes
attributes_list, attribute_label_dict = numerical_labels(processed_df, 'attribute', 'attribute_label')

# Training set for attribute classification
Y_attrib_train = processed_df['attribute_label']
Y_attrib_train = np.array(Y_attrib_train, dtype=int)

# Train the attribute classification Logistic Regression model
attrib_count_lr = LogisticRegression()
attrib_count_lr.fit(X_train_counts, Y_attrib_train)

# Testing attribute prediction on training data... overfitting likely
Y_attrib_train_pred = e_a_predict(X_train,X_train_counts,model=attrib_count_lr,threshold=0.1) # threshold hyperparameter tuned on training data
attrib_train_accuracy = accuracy_score(Y_attrib_train, Y_attrib_train_pred)

print(f"Training set attribute predictions accuracy score: {attrib_train_accuracy*100:.0f}%")

# Convert attribute predictions from numbers back to words
attribute_predictions = convert_numerical_predictions(Y_attrib_train_pred, attribute_label_dict)

# Combine Entity and Attribute predictions to create E#A pair predictions
processed_df['predicted_category'] = combine_entity_attributes(processed_df, entity_predictions, attribute_predictions)

# Sort Entity and Attribute predictions for more representative accuracy score (see example in function comments)
processed_df['predicted_category'] = sorted_predictions(processed_df)
train_reviews_df['predicted_category'] = sorted_predictions(processed_df)

e_a_labels, e_a_predictions, e_a_list, e_a_label_dict  = numerical_entity_attributes(processed_df)
category_train_accuracy = accuracy_score(e_a_labels, e_a_predictions)

print(f"Training set category predictions accuracy score: {category_train_accuracy*100:.0f}%")

# Reorder columns
reorder_columns = ['rid','id','processed_text','entity','entity_label',
                  'attribute','attribute_label','category','predicted_category',
                  'polarity','predicted_polarity']
processed_df = processed_df.reindex(columns=reorder_columns)
processed_df.head()

train_reviews_df.head()

def split_input_text(text):
    # Split input sentence/review by by coordinating conjunctions e.g., ['and','but','because']
    # If no CC in sentence, then split by punctuation
    # Each element (part of sentence) can then be assigned to different categories for the same sentence id, 
    # based on cosine similarity measure

    split_input = []
    for sentence in text:
        sentence = sentence.lower()
        sent_tag = nltk.pos_tag(nltk.word_tokenize(sentence))

        split_words = []
        pos_tags = ['CC']
        for elem in sent_tag:
            if elem[1] in pos_tags:
                split_words.append(elem[0])
        
        punctuation = [',',';','-']
        cc_in_flag = False
        for elem in sent_tag:
            if elem[1] in pos_tags:
                cc_in_flag = True
        
        if not cc_in_flag:
            for elem in sent_tag:
                if elem[1] in punctuation:
                    split_words.append(elem[0])

        result = []
        if split_words:
            for index, word in enumerate(split_words):
                result.append(sentence.split(split_words[index])[0])
                if len(sentence.split(split_words[index]))>1:
                    sentence = sentence.split(split_words[index])[1]
            result.append(sentence)
        else:
            result.append(sentence)

        result = [item.strip() for item in result]
        result = [item for item in result if item]
        split_input.append(result)
        
    return split_input

def filter_sentiment_input(text, cat):
    # Keep only most relevant element in sentence list to determine polarity for category
    # Measuring similarity between nouns in element of sentence and category (E#A) pair
    # Less expensive (more efficient) similarity calculation using only nouns rather than all words in part of sentence
    reduced_text = []
    for index, row in enumerate(text):
        n = len(row)
        tmp_dict = {}
        for i in range(n):
            cat_list = re.split('_|#', cat.loc[index])
            cat_list = [item.title() if item != 'OS' else item for item in cat_list]
            category = nlp(' '.join(cat_list))

            sent_tag = nltk.pos_tag(nltk.word_tokenize(row[i].lower()))
            noun_pos = ['NN','NNS','NNP']
            noun_list = [elem[0] for elem in sent_tag if elem[1] in noun_pos]
            noun_text = nlp(' '.join(noun_list))

            tmp_dict[row[i]] = category.similarity(noun_text)

        reduced_text.append(max(tmp_dict, key = tmp_dict.get))

    return reduced_text

# label sentiments
sentiments_list, polarity_label_dict = numerical_labels(processed_df, 'polarity', 'polarity_label')

# Training set for sentiment classification
Y_sentiment_train = processed_df['polarity_label']
Y_sentiment_train = np.array(Y_sentiment_train, dtype=int)

# Train the attribute classification Naive Bayes model
sentiment_count_lr = LogisticRegression()
sentiment_count_lr.fit(X_train_counts, Y_sentiment_train)

sentiment_processed_df = train_reviews_df.copy()
sentiment_processed_df['text'] = split_input_text(train_reviews_df['text'])

# N.B. cell takes ~30secs to run
sentiment_processed_df['text'] = filter_sentiment_input(sentiment_processed_df['text'], sentiment_processed_df['predicted_category'])

sentiment_processed_df.query("id == '273:9'")

# input for sentiment prediction on training set
sentiment_processed_df['processed_text'] = preprocess(tokenise(sentiment_processed_df['text']))

X_sentiment_input = sentiment_processed_df['processed_text']
X_sentiment_input = count_vectorizer.transform(X_sentiment_input)

# Testing sentiment prediction on training data... overfitting likely
Y_sentiment_train_pred = sentiment_count_lr.predict(X_sentiment_input)

sentiment_train_accuracy = accuracy_score(Y_sentiment_train, Y_sentiment_train_pred)
print(f"Training set sentiment prediction accuracy score: {sentiment_train_accuracy*100:.0f}%")

cr = classification_report(Y_sentiment_train, Y_sentiment_train_pred, target_names=sentiments_list)
# print(cr)
# Note that only 188 neutral training examples out of 2909... will be difficult to learn properties of neutral input features

# Convert sentiment predictions from numbers back to words
sentiment_predictions = convert_numerical_predictions(Y_sentiment_train_pred, polarity_label_dict)

processed_df['predicted_polarity'] = sentiment_predictions
reorder_columns = ['rid','id','processed_text','entity','entity_label',
                  'attribute','attribute_label','category','predicted_category',
                  'polarity','polarity_label','predicted_polarity']
processed_df = processed_df.reindex(columns=reorder_columns)
processed_df.head()

train_reviews_df['predicted_polarity'] = sentiment_predictions
train_reviews_df.head()

test_reviews_df = xml_to_df(filename='Laptops_Test_p1_gold.xml')
test_processed_df = test_reviews_df.copy()
test_reviews_df.head(3)

# Pre-process test set text
test_processed_df['text'] = preprocess(tokenise(test_reviews_df['text']))

# Obtain test set input features
X_test = test_processed_df['text']
X_test_counts = count_vectorizer.transform(X_test)

# Predict test set entities
Y_entity_test_pred = e_a_predict(X_test,X_test_counts,model=entity_count_lr,threshold=0.1) # threshold hyperparameter tuned on training data

# Convert entity predictions back to words
test_entity_predictions = convert_numerical_predictions(Y_entity_test_pred, entity_label_dict)

# Predict test set attributes
Y_attrib_test_pred = e_a_predict(X_test,X_test_counts,model=attrib_count_lr,threshold=0.1) # threshold hyperparameter tuned on training data

# Convert attribute predictions back to words
test_attribute_predictions = convert_numerical_predictions(Y_attrib_test_pred, attribute_label_dict)

# Combine Entity and Attribute predictions to create E#A pair predictions
test_processed_df['predicted_category'] = combine_entity_attributes(test_processed_df, test_entity_predictions, test_attribute_predictions)

# Sort Entity and Attribute predictions for more representative accuracy score (see example in function comments)
test_processed_df['predicted_category'] = sorted_predictions(test_processed_df)
test_reviews_df['predicted_category'] = sorted_predictions(test_processed_df)

e_a_labels, e_a_predictions, e_a_list, e_a_label_dict = numerical_entity_attributes(test_processed_df)

# Note target_names=e_a_list displays E#A categories as text form rather than numerical representation
cr = classification_report(e_a_labels, e_a_predictions, target_names=e_a_list)
print("Predicted E#A Category Classification Report (Test Set):\n")
print(cr)

# Reversing the polarity label dict to faciliatate test set polarity conversion to numerical labels.
reversed_polarity_label_dict = reverse_dict(polarity_label_dict)

Y_sentiment_test_labels = [reversed_polarity_label_dict[test_processed_df['polarity'][i]] for i in range(len(test_processed_df['polarity']))]

# Preparing filtered input for aspect based sentiment predictions on test set
test_sentiment_prep_df = test_reviews_df.copy()
test_sentiment_prep_df['text'] = split_input_text(test_reviews_df['text'])

test_sentiment_prep_df['processed_text'] = preprocess(tokenise(filter_sentiment_input(test_sentiment_prep_df['text'], test_sentiment_prep_df['predicted_category'])))
X_test_sentiment_input = count_vectorizer.transform(test_sentiment_prep_df['processed_text'])

# classifcation report for sentiment precitions on test data
Y_sentiment_test_pred = sentiment_count_lr.predict(X_test_sentiment_input)
Y_sentiment_test_labels = np.array([Y_sentiment_test_labels]).reshape(801,)

cr = classification_report(Y_sentiment_test_labels, Y_sentiment_test_pred, target_names=sentiments_list)
print("Predicted Sentiment Classification Report (Test Set):\n")
print(cr)

# Convert sentiment predictions to words and add to test test reviews data frame
test_processed_df['predicted_polarity'] = convert_numerical_predictions(Y_sentiment_test_pred, polarity_label_dict)
test_reviews_df['predicted_polarity'] = convert_numerical_predictions(Y_sentiment_test_pred, polarity_label_dict)

test_reviews_df[['text','category','predicted_category','polarity','predicted_polarity']].head()

# View the % of predictions where both the category AND polarity were predicted correct
combined_accuracy = overall_accuracy_absa(test_reviews_df)
print("For Part 1 test data:")
print(combined_accuracy)

# Parse xml file into pandas data frame to work with at the review level
def part2_xml_to_df(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    reviews = root.findall('Review')
    df_columns = ['rid','text','category','predicted_category','polarity','predicted_polarity']
    reviews_df = pd.DataFrame(columns=df_columns)

    for review in reviews:
        rid = review.get('rid')
        sentences = review.findall('sentences/sentence')
        text = ''
        for sentence in sentences:
            text += sentence.find('text').text + ' '
        opinions = review.findall('Opinions/Opinion')
        for opinion in opinions:
            category = opinion.get('category')
            polarity = opinion.get('polarity')
            predicted_category = ''
            predicted_polarity = ''
            reviews_df = pd.concat([reviews_df, 
                        pd.DataFrame([[rid, text, category, predicted_category, polarity, predicted_polarity]], 
                        columns=df_columns)], ignore_index=True)
    
    return reviews_df

train_reviews_df_p2 = part2_xml_to_df(filename='Laptops_Train_p2.xml')
train_reviews_df_p2.head()

processed_df_p2 = train_reviews_df_p2.copy()
processed_df_p2['text'] = preprocess(tokenise(train_reviews_df_p2['text']))
processed_df_p2 = processed_df_p2.rename(columns={"text": "processed_text"})
processed_df_p2.head()

def sorted_predictions_p2(df):
    # Updated for Part 2 to query unique review id isntead of sentence id
    # Aligns predictions with matching labels for sentences that have multiple opinions.
    # e.g., ground truth for sentence id=1: LAPTOP#GENERAL, LAPTOP#BATTERY_PERFORMANCE
    # predictions for sentence id=1 pre-alignment: LAPTOP#BATTERY_PERFORMANCE, LAPTOP#GENERAL
    # predictions post-alignment: LAPTOP#GENERAL, LAPTOP#BATTERY_PERFORMANCE

    N = len(df['category'])
    labels_dict = {}
    predictions_dict = {}
    sorted_predictions = []
    i = 0
    tmp = 0

    for rid in df['rid'].unique():
        labels_dict[rid] = list(df.query(f"rid == '{rid}'")['category'])
        predictions_dict[rid] = list(df.query(f"rid == '{rid}'")['predicted_category'])

    for key in labels_dict.keys():
        for value in labels_dict[key]:
            if value in predictions_dict[key]:

                idx = labels_dict[key].index(value) # obtain label index for matching label & prediction for given sentence id
                idx2 = predictions_dict[key].index(value) # obtain prediction index for matching label & prediction for given sentence id

                tmp = predictions_dict[key][idx]

                predictions_dict[key][idx] = value # re-order predictions so that they align with ground truth for given id
                predictions_dict[key][idx2] = tmp # swap changed values in predictions list for sentence id
    
    for values in predictions_dict.values():
        for value in values:
            sorted_predictions.append(value)

    return list(sorted_predictions)

def identify_p2_categories(df_p1, df_p2):
    # Identify p2 categories given category predictions for sentences with 
    # matching review id from part 1
    # Additional functionality: if LAPTOP#GENERAL not predicted, remove least
    # similar predicted category and include LAPTOP#GENERAL as predicted category
    # If same category predicted more than once, remove duplicate and predict new

    categories = []
    pred_cat = 'predicted_category'
    for rid in df_p2['rid'].unique():
        len_pred_cats_p1 = len(df_p1.query(f'rid == "{rid}"')[pred_cat])
        len_pred_cats_p2 = len(df_p2.query(f'rid == "{rid}"')[pred_cat])
        if 'LAPTOP#GENERAL' not in set(df_p1.query(f'rid == "{rid}"')[pred_cat]):
            if len_pred_cats_p2 < len_pred_cats_p1:
                categories += ['LAPTOP#GENERAL']
                categories += list(df_p1.query(f'rid == "{rid}"')[pred_cat])[:len_pred_cats_p2-1]
            elif len_pred_cats_p1 == len_pred_cats_p2:
                categories += ['LAPTOP#GENERAL']
                categories += list(df_p1.query(f'rid == "{rid}"')[pred_cat])[:len_pred_cats_p2-1]
            else:  
                delta = len_pred_cats_p2 - len_pred_cats_p1
                categories += ['LAPTOP#GENERAL']
                categories += list(df_p1.query(f'rid == "{rid}"')[pred_cat])
                for i in range(delta-1):
                    categories += ['LAPTOP#GENERAL'] # duplicate predictions will be handled later

        else:
            if len_pred_cats_p2 < len_pred_cats_p1:
                if 'LAPTOP#GENERAL' in list(df_p1.query(f'rid == "{rid}"')[pred_cat])[:len_pred_cats_p2]:
                    categories += list(df_p1.query(f'rid == "{rid}"')[pred_cat])[:len_pred_cats_p2]
                else:
                    categories += ['LAPTOP#GENERAL']
                    categories += list(df_p1.query(f'rid == "{rid}"')[pred_cat])[:len_pred_cats_p2-1]
            elif len_pred_cats_p1 == len_pred_cats_p2:
                categories += list(df_p1.query(f'rid == "{rid}"')[pred_cat])
            else:
                delta = len_pred_cats_p2 - len_pred_cats_p1
                categories += list(df_p1.query(f'rid == "{rid}"')['predicted_category'])
                for i in range(delta):
                    categories += ['LAPTOP#GENERAL'] # duplicate predictions will be handled later

    return categories

def repredict_duplicate_cats(process_df, clean_df, attrib_list=attributes_list, ent_list=entities_list):
    # Identify duplicate predicted categories for each given review
    # Use spacy cosine similarity measure to re-predict new categories
    categories_new = []
    for rid in process_df['rid'].unique():
        if len(set(process_df.query(f'rid == "{rid}"')['predicted_category'])) < len(list(process_df.query(f'rid == "{rid}"')['predicted_category'])):
            categories_new += set(process_df.query(f'rid == "{rid}"')['predicted_category'])

            delta = len(list(process_df.query(f'rid == "{rid}"')['predicted_category'])) - len(set(process_df.query(f'rid == "{rid}"')['predicted_category']))
            unique_cats = list(set(process_df.query(f'rid == "{rid}"')['predicted_category']))
            duplicates = list(process_df.query(f'rid == "{rid}"')['predicted_category'])
            [duplicates.remove(item) for item in unique_cats if item in duplicates]

            # predict new category from ranked list of 20 most similar categories
            review_text = clean_df.query(f'rid == "{rid}"')['text'].unique()[0]

            entity_sim_dict = {}
            for entity in ent_list[:5]:
                entity_sim_dict[entity] = nlp(entity).similarity(nlp(review_text))
            sorted_entities = sorted(((value,key) for key,value in entity_sim_dict.items()), reverse=True)

            attrib_sim_dict = {}
            for attrib in attrib_list[:4]:
                attrib_sim_dict[attrib] = nlp(attrib).similarity(nlp(review_text))
            sorted_attribs = sorted(((value,key) for key,value in attrib_sim_dict.items()), reverse=True)

            unsorted_categories = {}
            for i in sorted_entities:
                for j in sorted_attribs:
                    unsorted_categories[i[1]+'#'+j[1]] = i[0]*j[0]

            sorted_categories = sorted(((value,key) for key,value in unsorted_categories.items()), reverse=True)
            ranked_categories = [i[1] for i in sorted_categories]

            # Only keep ranked_categories that are not already in unique_cats
            [ranked_categories.remove(item) for item in ranked_categories if item in unique_cats]

            for i in range(delta):
                categories_new += [ranked_categories[i]]
    
        else:
            categories_new += set(process_df.query(f'rid == "{rid}"')['predicted_category'])
        
    return categories_new

# populating E#A category predictions given that the part 2 data contains the same sentences as part 1
processed_df_p2['predicted_category'] = identify_p2_categories(train_reviews_df, train_reviews_df_p2)

# Re-predict duplicate predicted categories for given review
# Note: cell takes ~45secs to run
processed_df_p2['predicted_category'] = repredict_duplicate_cats(processed_df_p2, train_reviews_df_p2)

# Sort category predictions for more representative accuracy score (see example in function comments)
processed_df_p2['predicted_category'] = sorted_predictions_p2(processed_df_p2)
train_reviews_df_p2['predicted_category'] = sorted_predictions_p2(processed_df_p2)

e_a_labels_p2, e_a_predictions_p2, e_a_list_p2, e_a_label_dict_p2  = numerical_entity_attributes(processed_df_p2)
category_train_accuracy_p2 = accuracy_score(e_a_labels_p2, e_a_predictions_p2)

print(f"Training set category predictions accuracy score: {category_train_accuracy_p2*100:.0f}%")

# Reorder columns
reorder_columns = ['rid','processed_text','entity','entity_label',
                  'attribute','attribute_label','category','predicted_category',
                  'polarity','predicted_polarity']

processed_df_p2.head()

def split_input_text_p2(text):
    # Split input review into list of sentences.
    split_input = []
    for review in text:
        split_input.append(sent_tokenize(review))
        
    return split_input

def filter_sentiment_input_p2(text, cat, threshold=1):
    # If predicted LAPTOP#GENERAL, keep entire review text.
    # If one sentence in review is much more similar to predicted category
    # than the next most similar sentence, keep only that sentence for sentiment prediction.
    # Else keep the two most similar sentences for given category from review.

    filtered_input_1 = []
    filtered_input_2 = []
    for idx, review in enumerate(text):
        if cat[idx] == 'LAPTOP#GENERAL':
            filtered_input_1.append(' '.join(review))
            filtered_input_2.append(' '.join(review))
        else:
            cat_list = re.split('_|#', cat[idx])
            cat_list = [item.title() if item != 'OS' else item for item in cat_list]
            category = nlp(' '.join(cat_list))

            sent_sim_dict = {}
            for sentence in review:
                sent_sim_dict[sentence] = category.similarity(nlp(sentence))
            
            sorted_sents = sorted(((value,key) for key,value in sent_sim_dict.items()), reverse=True)
            
            if len(sorted_sents) > 1:
                # threshold hyperparameter determined on training set
                if (sorted_sents[0][0] - sorted_sents[1][0]) < threshold:
                    filtered_input_1.append(sorted_sents[0][1])
                    filtered_input_2.append(sorted_sents[1][1])
                else:
                    filtered_input_1.append(sorted_sents[0][1])
                    filtered_input_2.append(sorted_sents[0][1])
            else:
                filtered_input_1.append(sorted_sents[0][1])
                filtered_input_2.append(sorted_sents[0][1])
                
    return filtered_input_1, filtered_input_2

def p2_sentiment_predictions(input_1, input_2):
    # make sentiment predictions
    # polarity_labels_dict = {0: 'positive', 1: 'negative', 2: 'neutral', 3: 'conflict'} 
    N,D  = input_1.shape
    Y_sentiment_pred = np.zeros(N)

    for i in range(N):
        prediction_1 = sentiment_count_lr.predict(input_1[i,:])
        prediction_2 = sentiment_count_lr.predict(input_2[i,:])
        
        if prediction_1 == prediction_2:
            Y_sentiment_pred[i] = prediction_1

        # if prediction is positive and negative
        elif prediction_1 == 0 and prediction_2 == 1:
            Y_sentiment_pred[i] = 3 # numerical value for conflict

        # if prediction is positive and neutral
        elif prediction_1 == 0 or prediction_2 == 0: 
            Y_sentiment_pred[i] = 0 # numerical value for positive

        # if prediction is negative and neutral
        elif prediction_1 == 1 or prediction_2 == 1: 
            Y_sentiment_pred[i] = 1 # numerical prediction for negative

        # If prediction is neutral for both sentences
        else:
            Y_sentiment_pred[i] = 2 # numerical prediction for neutral
        
    return Y_sentiment_pred

# label part 2 training sentiments
sentiments_list_p2, polarity_label_dict_p2 = numerical_labels(processed_df_p2, 'polarity', 'polarity_label')

# Training labels for part 2 sentiment classification
Y_sentiment_train_p2 = processed_df_p2['polarity_label']
Y_sentiment_train_p2 = np.array(Y_sentiment_train_p2, dtype=int)

# Creating copy of data frame to populate with processed input text
# specifically processed for sentiment prediction
sentiment_processed_df_p2 = train_reviews_df_p2.copy()
sentiment_processed_df_p2['text'] = split_input_text_p2(train_reviews_df_p2['text'])

# N.B. cell takes ~60secs to run
sentiment_processed_df_p2['text_1'],sentiment_processed_df_p2['text_2'] = filter_sentiment_input_p2(sentiment_processed_df_p2['text'],
                                        sentiment_processed_df_p2['predicted_category'],
                                        threshold=0.02)

# Pre process input features for sentiment classification
# two sets of input features
# If predicted sentiments for each set is different, predict conflict
sentiment_processed_df_p2['processed_text_1'] = preprocess(tokenise(sentiment_processed_df_p2['text_1']))
X_train_sentiment_p2_1 = count_vectorizer.transform(sentiment_processed_df_p2['processed_text_1'])

sentiment_processed_df_p2['processed_text_2'] = preprocess(tokenise(sentiment_processed_df_p2['text_2']))
X_train_sentiment_p2_2 = count_vectorizer.transform(sentiment_processed_df_p2['processed_text_2'])

# Make sentiment predictions on part 2 Training data
Y_sentiment_train_pred_p2 = p2_sentiment_predictions(X_train_sentiment_p2_1, X_train_sentiment_p2_2)

sentiment_train_accuracy_p2 = accuracy_score(Y_sentiment_train_p2, Y_sentiment_train_pred_p2)
print(f"Training set sentiment prediction accuracy score: {sentiment_train_accuracy_p2*100:.0f}%")

cr = classification_report(Y_sentiment_train_p2, Y_sentiment_train_pred_p2, target_names=sentiments_list_p2)
print(cr)

test_reviews_df_p2 = part2_xml_to_df(filename='Laptops_Test_p2_gold.xml')
train_reviews_df_p2.head(3)

test_processed_df_p2 = test_reviews_df_p2.copy()
test_processed_df_p2['text'] = preprocess(tokenise(test_reviews_df_p2['text']))
test_processed_df_p2 = test_processed_df_p2.rename(columns={"text": "processed_text"})
test_processed_df_p2.head(3)

# populating E#A category predictions given that the part 2 data contains the same sentences as part 1
test_processed_df_p2['predicted_category'] = identify_p2_categories(test_reviews_df, test_reviews_df_p2)

# Re-predict duplicate predicted categories for given review
# Note: cell takes ~20secs to run
test_processed_df_p2['predicted_category'] = repredict_duplicate_cats(test_processed_df_p2, test_reviews_df_p2)

# Sort category predictions for more representative accuracy score (see example in function comments)
test_processed_df_p2['predicted_category'] = sorted_predictions_p2(test_processed_df_p2)
test_reviews_df_p2['predicted_category'] = sorted_predictions_p2(test_processed_df_p2)

e_a_labels_p2, e_a_predictions_p2, e_a_list_p2, e_a_label_dict_p2  = numerical_entity_attributes(test_processed_df_p2)

cr = classification_report(e_a_labels_p2, e_a_predictions_p2, target_names=e_a_list_p2)
print("(Part 2) Predicted E#A Category Classification Report (Test Set):\n")
print(cr)

# label part 2 test sentiments
test_processed_df_p2['polarity_label'] = ''
i = 0
for label in sentiments_list_p2:
    test_processed_df_p2.loc[test_processed_df_p2['polarity'] == label, 'polarity_label'] = i
    polarity_label_dict_p2[i] = label
    i += 1


# Test labels for part 2 sentiment classification
Y_sentiment_test_p2 = test_processed_df_p2['polarity_label']
Y_sentiment_test_p2 = np.array(Y_sentiment_test_p2, dtype=int)

# Creating copy of data frame to populate with processed input text
# specifically processed for sentiment prediction
test_sentiment_prep_df_p2 = test_reviews_df_p2.copy()
test_sentiment_prep_df_p2['text'] = split_input_text_p2(test_reviews_df_p2['text'])

# N.B. cell takes ~30secs to run
test_sentiment_prep_df_p2['text_1'],test_sentiment_prep_df_p2['text_2'] = filter_sentiment_input_p2(test_sentiment_prep_df_p2['text'],
                                        test_sentiment_prep_df_p2['predicted_category'],
                                        threshold=0.02)

# Pre process input features for sentiment classification
# two sets of input features
# If predicted sentiments for each set is positive & negative, predict conflict
test_sentiment_prep_df_p2['processed_text_1'] = preprocess(tokenise(test_sentiment_prep_df_p2['text_1']))
X_test_sentiment_p2_1 = count_vectorizer.transform(test_sentiment_prep_df_p2['processed_text_1'])

test_sentiment_prep_df_p2['processed_text_2'] = preprocess(tokenise(test_sentiment_prep_df_p2['text_2']))
X_test_sentiment_p2_2 = count_vectorizer.transform(test_sentiment_prep_df_p2['processed_text_2'])

# Make sentiment predictions on part 2 test data
Y_sentiment_test_pred_p2 = p2_sentiment_predictions(X_test_sentiment_p2_1, X_test_sentiment_p2_2)

# Produce classification report
cr = classification_report(Y_sentiment_test_p2, Y_sentiment_test_pred_p2, target_names=sentiments_list_p2)
print("(Part 2) Predicted Sentiment Classification Report (Test Set):\n")
print(cr)

test_processed_df_p2['predicted_polarity'] = convert_numerical_predictions(Y_sentiment_test_pred_p2, polarity_label_dict_p2)
test_reviews_df_p2['predicted_polarity'] = convert_numerical_predictions(Y_sentiment_test_pred_p2, polarity_label_dict_p2)

test_reviews_df_p2[['text','category','predicted_category','polarity','predicted_polarity']].head()

# View the % of predictions where both the category AND polarity were predicted correct
combined_accuracy_p2 = overall_accuracy_absa(test_reviews_df_p2)
print("For Part 2 test data:")
print(combined_accuracy_p2)

# agregate data for plotting
result = test_reviews_df_p2.groupby(['category', 'predicted_polarity']).size().unstack(fill_value=0)

# printing result
print(result)

# Separae data for plotting
result.plot(kind='bar', stacked=True, figsize=(14, 7))
plt.title('Розподіл сентиментів по категоріям')
plt.xlabel('Категорія')
plt.ylabel('Кількість відгуків')
plt.show()