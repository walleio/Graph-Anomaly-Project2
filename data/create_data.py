import os
import csv
import pandas as pd
import json
from itertools import islice

users = []
reviews = []
user_ratings = []
review_number = 0
iters = 0
dataset = 'yelp'

if dataset == 'amazon':
    with open('/projects/p32673/AmazonData/movies.txt', 'r', errors='ignore') as file:
        user_dict = {}
        review_dict = {}
        for line in file:
            line = line.strip()
            if line.startswith('product/productId:'):
                review_dict['productId'] = line.split(': ', 1)[1]
            elif line.startswith('review/userId:'):
                review_dict['userId'] = line.split(': ', 1)[1]
                user_dict['userId'] = line.split(': ', 1)[1]
            elif line.startswith('review/profileName:'):
                # it is possible for there to be no profile name
                try:
                    user_dict['profileName'] = line.split(': ', 1)[1].strip('"')  # Remove quotes
                except:
                    user_dict['profileName'] = ''
            elif line.startswith('review/helpfulness:'):
                try:
                    helpfulness = line.split(': ', 1)[1]
                except:
                    helpfulness = '0/0' 
                
                review_dict['helpfulness numerator'] = int(helpfulness.split('/')[0])
                review_dict['helpfulness denominator'] = int(helpfulness.split('/')[1])
            elif line.startswith('review/score:'):
                review_dict['score'] = float(line.split(': ', 1)[1])
            elif line.startswith('review/time:'):
                review_dict['time'] = int(line.split(': ', 1)[1])
            elif line.startswith('review/summary:'):
                try:
                    review_dict['summary'] = line.split(': ', 1)[1]
                except:
                    review_dict['summary'] = ''
            elif line.startswith('review/text:'):
                review_dict['text'] = line.split(': ', 1)[1]
            
            if line == '':
                users.append(user_dict.copy())
                reviews.append(review_dict.copy())
                if len(user_ratings) == 0 or review_dict['userId'] not in [item['userId'] for item in user_ratings]:
                    try:
                        user_ratings.append({'userId': review_dict['userId'], 
                                                'helpful ratings': review_dict['helpfulness numerator'], 
                                                'total ratings': review_dict['helpfulness denominator']})
                    except:
                        continue
                else:
                    try:
                        for rating in user_ratings:
                            if rating['userId'] == review_dict['userId']:
                                rating['helpful ratings'] = rating['helpful ratings'] + review_dict['helpfulness numerator']
                                rating['total ratings'] = rating['total ratings'] + review_dict['helpfulness denominator']
                    except:
                        continue
                user_dict.clear()
                review_dict.clear()


            filtered_list = list(filter(lambda r: r['total ratings'] >= 20, user_ratings))

            if len(filtered_list) > 1000:
                for l in filtered_list:
                    l['label'] = 2
                    average_rating = l['helpful ratings'] / l['total ratings']
                    if average_rating > 0.7:
                        l['label'] = 1
                    if average_rating < 0.3:
                        l['label'] = 0

                filtered_users = []
                for i in users:
                    label = [j['label'] for j in filter(lambda x: x['userId'] == i['userId'], filtered_list)]
                    if label != [] and label != [2]:
                        if i not in filtered_users:
                            i['label'] = label
                            filtered_users.append(i)

                filtered_reviews = []
                for i in reviews:
                    label = [j['label'] for j in filter(lambda x: x['userId'] == i['userId'], filtered_list)]
                    if label != [] and label != [2]:
                        filtered_reviews.append(i)

                mode = 'a' if os.path.exists('users.csv') else 'w'
                with open('users.csv', mode, newline='', encoding='utf-8') as csvfile:
                    fieldnames = filtered_users[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if mode == 'w':
                        writer.writeheader()

                    for row in filtered_users:
                        row['label'] = row['label'][0]
                        writer.writerow(row)

                with open('reviews.csv', mode, newline='', encoding='utf-8') as csvfile:
                    fieldnames = reviews[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if mode == 'w':
                        writer.writeheader()

                    for row in filtered_reviews:
                        writer.writerow(row)
                
                break
elif dataset ==  'yelp':
    with open('/projects/p32673/Yelp/Yelp JSON/yelp_academic_dataset_review.json', 'r') as f:
        data = [json.loads(line) for line in islice(f, 100)]

    df = pd.DataFrame(data)
    
    
