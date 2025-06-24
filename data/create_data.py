import os
import csv

users = []
reviews = []
review_number = 0
count = 0
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
            if user_dict not in users:
                users.append(user_dict)
            reviews.append(review_dict)
            user_dict = {}
            review_dict = {}

            user_ratings = []
            if len(users) == 100000:
                count += 1
                skip = False
                for movie in reviews:
                    try:
                        if len(user_ratings) == 0 or movie['userId'] not in [item['userId'] for item in user_ratings]:
                            try:
                                user_ratings.append({'userId': movie['userId'], 
                                                        'helpful ratings': movie['helpfulness numerator'], 
                                                        'total ratings': movie['helpfulness denominator']})
                            except:
                                continue
                        else:
                            try:
                                for rating in user_ratings:
                                    if rating['userId'] == movie['userId']:
                                        rating['helpful ratings'] = rating['helpful ratings'] + movie['helpfulness numerator']
                                        rating['total ratings'] = rating['total ratings'] + movie['helpfulness denominator']
                            except:
                                continue
                    except:
                        skip = True
                        
                if not skip:
                    filtered_list = list(filter(lambda r: r['total ratings'] >= 20, user_ratings))

                    for l in filtered_list:
                        l['label'] = 2
                        average_rating = l['helpful ratings'] / l['total ratings']
                        if average_rating > 0.7:
                            l['label'] = 1
                        if average_rating < 0.3:
                            l['label'] = 0

                    for i in users:
                        i['label'] = [j['label'] for j in filter(lambda x: x['userId'] == i['userId'], filtered_list)]

                    filtered_reviews = []
                    for i in reviews:
                        label = [j['label'] for j in filter(lambda x: x['userId'] == i['userId'], filtered_list)]
                        if label != [] and label != [2]:
                            i['label'] = label[0]
                            filtered_reviews.append(i)

                    mode = 'a' if os.path.exists('users3.csv') else 'w'
                    with open('users3.csv', mode, newline='', encoding='utf-8') as csvfile:
                        fieldnames = users[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        if mode == 'w':
                            writer.writeheader()

                        for row in users:
                            if row['label'] != []:
                                row['label'] = row['label'][0]
                                if row['label'] != 2:
                                    writer.writerow(row)

                    with open('reviews3.csv', mode, newline='', encoding='utf-8') as csvfile:
                        fieldnames = filtered_reviews[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        if mode == 'w':
                            writer.writeheader()

                        for row in filtered_reviews:
                            writer.writerow(row)

                print(count)
                break
                if count == 101:
                    break
                users.clear()
                reviews.clear()
                user_ratings.clear()

