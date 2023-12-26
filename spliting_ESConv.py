import json
import sqlite3

sqlite_data = []

dataset = json.load(open('ESConv.json'))

# convert the dataset to sqlite, where ignore the dialog part

# for i in dataset:
#     print(i)
#     modify_doc = {
#         # 'seeker_survey_iei': i['survey_score']['seeker']['initial_emotion_intensity'],
#         # 'seeker_survey_empathy': i['survey_score']['seeker']['empathy'],
#         # 'seeker_survey_relevance': i['survey_score']['seeker']['relevance'],
#         # 'seeker_survey_fei': i['survey_score']['seeker']['final_emotion_intensity'],
#         # 'supporter_survey_relevance': i['survey_score']['supporter']['relevance'],
#         'experience_type': i['experience_type'],
#         'emotion_type': i['emotion_type'],
#         "problem_type": i['problem_type'],
#         "situation": i['situation'],
#         "seeker_question1": i['seeker_question1'],
#         "seeker_question2": i['seeker_question2'],
#         "supporter_question1": i['supporter_question1'],
#         "supporter_question2": i['supporter_question2'],
#         "id": i['id']
#     }
#     try: modify_doc['seeker_survey_iei'] = i['survey_score']['seeker']['initial_emotion_intensity']
#     except: modify_doc['seeker_survey_iei'] = None
#     try: modify_doc['seeker_survey_empathy'] = i['survey_score']['seeker']['empathy']
#     except: modify_doc['seeker_survey_empathy'] = None
#     try: modify_doc['seeker_survey_relevance'] = i['survey_score']['seeker']['relevance']
#     except: modify_doc['seeker_survey_relevance'] = None
#     try: modify_doc['seeker_survey_fei'] = i['survey_score']['seeker']['final_emotion_intensity']
#     except: modify_doc['seeker_survey_fei'] = None
#     try: modify_doc['supporter_survey_relevance'] = i['survey_score']['supporter']['relevance']
#     except: modify_doc['supporter_survey_relevance'] = None
#     sqlite_data.append(modify_doc)
#
# conn = sqlite3.connect('ESConv.db')
# c = conn.cursor()
# c.execute('''CREATE TABLE ESConv
#              (seeker_survey_iei real, seeker_survey_empathy real, seeker_survey_relevance real, seeker_survey_fei real, supporter_survey_relevance real, experience_type text, emotion_type text, problem_type text, situation text, seeker_question1 text, seeker_question2 text, supporter_question1 text, supporter_question2 text, id text)''')
# conn.commit()
#
# for i in sqlite_data:
#     c.execute('''INSERT INTO ESConv VALUES (
#         :seeker_survey_iei, :seeker_survey_empathy, :seeker_survey_relevance, :seeker_survey_fei, :supporter_survey_relevance, :experience_type, :emotion_type, :problem_type, :situation, :seeker_question1, :seeker_question2, :supporter_question1, :supporter_question2, :id
#     )''', i)
# conn.commit()
# conn.close()


conn = sqlite3.connect('ESConv.db')
# select all the ones who have a difference of four between initial and final emotion intensity
c = conn.cursor()
c.execute('''SELECT * FROM ESConv WHERE seeker_survey_iei - seeker_survey_fei = 4''')
conn.commit()

rt = c.fetchall()
# get all the ids
ids = [i[-1] for i in rt]
# based on id get the corresponding dialog in dataset
new_dataset = []

for i in range(len(ids)):
    data_new = {}
    if dataset[i]['id'] in ids:
        # check if there is 'feedback' label in the dialog section
        cont = False
        for j in dataset[i]['dialog']:
            # print(i['annotation'].keys())
            if 'feedback' in j['annotation'].keys():
                cont = True
        if cont:
            data_new['id'] = dataset[i]['id']
            data_new['dialog'] = dataset[i]['dialog']
            new_dataset.append(data_new)

with open('evaluation2.json', 'w') as outfile:
    json.dump(new_dataset, outfile, indent=4)
