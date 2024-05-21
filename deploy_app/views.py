from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pickle
import google.generativeai as genai
import os
import numpy as np
import pandas as pd

region_model = pickle.load(open(os.path.join("","model.pkl"), 'rb'))


GEMINI_KEY = os.environ.get("GEMINI_KEY")
genai.configure(api_key=GEMINI_KEY)

dict_cor = {'job_category': ['Machine Learning and AI',
  'Data Engineering',
  'Data Analysis',
  'Data Science and Research',
  'Data Architecture and Modeling',
  'Leadership and Management',
  'Data Management and Strategy',
  'BI and Visualization',
  'Cloud and Database',
  'Data Quality and Operations'],
 'experience_level': ['Entry-level', 'Mid-level', 'Senior', 'Executive'],
 'employment_type': ['Full-time', 'Contract', 'Freelance', 'Part-time'],
 'work_setting': ['Hybrid', 'In-person', 'Remote'],'region': ['Europe',
  'Africa',
  'United States',
  'Asia',
  'Oceania',
  'North America',
  'Central America',
  'South America']}

encode_dic = {'Contract': 0,
 'Freelance': 1,
 'Full-time': 2,
 'Part-time': 3,
 'Hybrid': 0,
 'In-person': 1,
 'Remote': 2,
 'BI and Visualization': 0,
 'Cloud and Database': 1,
 'Data Analysis': 2,
 'Data Architecture and Modeling': 3,
 'Data Engineering': 4,
 'Data Management and Strategy': 5,
 'Data Quality and Operations': 6,
 'Data Science and Research': 7,
 'Leadership and Management': 8,
 'Machine Learning and AI': 9,
 'Entry-level': 0,
 'Executive': 1,
 'Mid-level': 2,
 'Senior': 3,
 'Africa': 0,
 'Asia': 1,
 'Europe': 2,
 'North America': 3,
 'Oceania': 4,
 'South America': 5,
 'United States': 6}

reg = ['Africa','Asia','Europe', 'North America', 'Oceania', 'South America', 'United States']

model = genai.GenerativeModel('gemini-1.5-flash-latest',system_instruction=f"you are my json interpeter, extract {' '.join(list(dict_cor.keys()))} from the text and return a json format according to these categories: {dict_cor}")

def text_to_json(text:str):
  init_pos = text.find("{")
  end_pos = text.find("}")
  text_base = text[init_pos:end_pos+1].replace("\n","")
  return str(text_base)

@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message')
        response = generate_response(user_message)
        return JsonResponse({'response': response})
    return JsonResponse({'error': 'Invalid request'}, status=400)

def generate_response(message):
    response = model.generate_content(message)
    # Implement your chatbot logic here
    info = json.loads(text_to_json(response.text))
    info.pop("region")
    descr = {}
    for k,v in info.items():
        descr[k] = [encode_dic[v[0]]]
    
    descr["salary_in_usd_2023"] = 10000
    descr = pd.DataFrame(descr)
    pred = region_model.predict(descr)
    return f"Hola, te recomiendo buscar empleo en {reg[pred[0]]}"

def chat(request):
    return render(request, 'deploy_app/home.html')
