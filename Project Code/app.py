from flask import Flask, send_file, request,render_template
import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
import PyPDF2
import csv
from flask import make_response
import zipfile
import math
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zipfile import ZipFile
app = Flask(__name__)
app.static_folder = 'static'
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB

def preprocessing(single_resume):
    single_resume = single_resume.lower()
    single_resume = single_resume.replace(string.punctuation, '')
    single_resume = single_resume.replace("\n", '')
    single_resume = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', single_resume)
    single_resume = re.sub(r'[^a-z A-z ]', '', single_resume)

    return single_resume

@app.route('/', methods=['GET', 'POST'])
def HomePage():
    return render_template('index.html', baseurl = request.host_url)


@app.route('/supervised', methods=['GET', 'POST'])
def supervisedModel():
    dtclf = pickle.load(open('supervised_model.pb', 'rb'))
    vectorizer = pickle.load(open('vectorizer_for_supervised.pb', 'rb'))
    resumefile = request.files['file']
    resumefilename = secure_filename(resumefile.filename)
    resumefile.save(os.path.join(app.config['UPLOAD_FOLDER'], resumefilename))
    resume_list, filenames = unzip(app.config['UPLOAD_FOLDER'], resumefilename)
    result=[]
    for sample_resume in resume_list:
        sample_resume = vectorizer.transform([sample_resume])
        result.append(round(dtclf.predict(sample_resume)[0] * 100, 2))
    dic = {'Resume Name': filenames, 'Resume Score': result}
    df = pd.DataFrame(dic)
    df.sort_values(by=['Resume Score'], inplace=True, ascending=False)
    plt.title('Graph for good resume ')
    plt.xlabel('Resume Name')
    plt.ylabel('Goodness Percentage')
    plt.xticks(np.arange(0, len(filenames), 1.0))
    graph=plt.bar(filenames,result,width=0.5,align='center')
    figure = plt.gcf()
    plt.xticks(rotation = 90)
    i = 0
    for p in graph:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x+width/2,
                y+height*1.02,
                str(result[i])+'%',
                ha='center',
                weight='bold',rotation=90)
        i+=1
    figure.set_size_inches(10, 6)
    plt.savefig("BarChartForResumeRanking.png", dpi=500)

    df.to_csv('ResumeRanking.csv',index=False)
    plt.clf()

    labels = []
    values = []
    for index, row in df.iterrows():
        labels.append(row['Resume Name'])
        values.append(row['Resume Score'])
    return render_template('bar.html', labels = labels, values = values, csv = 'ResumeRanking.csv', graph = "BarChartForResumeRanking.png")



@app.route('/match', methods=['GET', 'POST'])
def percentageMatchingWithJobDescription():
    resumefile = request.files['resume']
    resumefilename = secure_filename(resumefile.filename)
    resumefile.save(os.path.join(app.config['UPLOAD_FOLDER'], resumefilename))

    resume_list, filenames = unzip(app.config['UPLOAD_FOLDER'], resumefilename)
    jdfile = request.files['jobdescription']
    jdfilename = secure_filename(jdfile.filename)
    jdfile.save(os.path.join(app.config['UPLOAD_FOLDER'], jdfilename))
    pdfFileObj = open("uploads/" + jdfilename, "rb")
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    job_description = ''
    for i in range(pdfReader.numPages):
        pageObj = pdfReader.getPage(i)
        job_description += pageObj.extractText()
    percentage_matching = []
    for resume in resume_list:
        text = [resume, job_description]
        cv = CountVectorizer(preprocessor=preprocessing, stop_words='english')
        count_matrix = cv.fit_transform(text)
        print("\nSimlarity Scores:")
        print(cosine_similarity(count_matrix))
        matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
        matchPercentage = round(matchPercentage, 2)
        percentage_matching.append(matchPercentage)
    dic = {'Resume Name': filenames, 'match percentage with Job Description': percentage_matching}
    df = pd.DataFrame(dic)
    df.sort_values(by=['match percentage with Job Description'], inplace=True, ascending=False)
    

# plt.figure(figsize=(20, 3)) 
    plt.title('Graph for Percentage matching with Job Description')
    plt.xlabel('Resume Name')
    plt.ylabel('Percentage matching')
    plt.xticks(np.arange(0, len(filenames), 1.0))
    graph=plt.bar(filenames,percentage_matching,width=0.5,align='center')
    figure = plt.gcf()
    plt.xticks(rotation = 90)
    i = 0
    for p in graph:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x+width/2,
                y+height*1.02,
                str(percentage_matching[i])+'%',
                ha='center',
                weight='bold',rotation=90)
        i+=1
    figure.set_size_inches(10, 6)
    plt.savefig("ResumeMatchingBarChart.png", dpi=500)

    df.to_csv('ResumeMatching.csv',index=False)

    plt.clf()
    labels = []
    values = []
    for index, row in df.iterrows():
        labels.append(row['Resume Name'])
        values.append(row['match percentage with Job Description'])
    return render_template('bar.html', labels = labels, values = values, csv = 'ResumeMatching.csv', graph = "ResumeMatchingBarChart.png")


def unzip(path, filename):
    zipped = zipfile.ZipFile(path + '/' + filename, 'r')
    files = []
    filenames = []
    zipped.extractall(path)
    for file in zipped.namelist():
        filenames.append(file)
        pdfFileObj = open("uploads/" + file, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        extract = ""
        for page in range(pdfReader.numPages):
            pageObj = pdfReader.getPage(page)
            extract = extract + pageObj.extractText()
        pdfFileObj.close()
        files.append(extract)
    zipped.close()
    return files, filenames


@app.route('/unsupervised', methods=['GET', 'POST'])
def unsupervisedClustering():
    file = request.files['zip']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    resume_list, filenames = unzip(app.config['UPLOAD_FOLDER'], filename)
    vectorizer = TfidfVectorizer(stop_words='english', preprocessor=preprocessing)
    features = vectorizer.fit_transform(resume_list)

    dist_points_from_clusters=[]
    k=range(2,8)
    for no_of_clusters in k:
        k_model=KMeans(no_of_clusters)
        k_model.fit(features)
        dist_points_from_clusters.append(k_model.inertia_)
    print(dist_points_from_clusters)
    a=dist_points_from_clusters[0]-dist_points_from_clusters[5]
    b=k[5]-k[0]
    c1=k[0]*dist_points_from_clusters[5]
    c2=k[5]*dist_points_from_clusters[0]
    c=c1-c2
    distance_from_line=[]
    for i in range(6):
        distance_from_line.append(calc_distance(k[i],dist_points_from_clusters[i],a,b,c))
    print(distance_from_line.index(max(distance_from_line))+1)

    k=distance_from_line.index(max(distance_from_line))+1

    model = KMeans(k)
    model.fit(features)
    clusters = model.predict(features)
    clusters_name=[]
    for i in range(len(clusters)):
        clusters_name.append('Type : '+str(clusters[i]))
    dic = {'Resume Name ': filenames, 'Cluster': clusters_name}
    df = pd.DataFrame(dic)
    df.sort_values(by=['Cluster'], inplace=True)
    unique=len(set(clusters))
    count=[0 for i in range(unique)]
    labels=[]
    clusters.sort()
    for i in clusters:
        if 'Type '+str(i) not in labels:
            
            labels.append('Type '+str(i))
        count[i]+=1
    plt.pie(count,labels=labels,autopct=lambda pct: func(pct, count))
    print(count)
    figure = plt.gcf()
    figure.set_size_inches(10, 6)
    plt.title('Pie chart showing kinds of Resume')
    plt.savefig("ResumeGrouping.png", dpi=500)
    df.to_csv('GroupedResume.csv',index=False)
    plt.clf()
    return render_template('pie.html', labels = labels, values = count, csv = 'GroupedResume.csv', graph = "ResumeGrouping.png")

def func(a,b):
    return str(round(a,2))+'%'

def calc_distance(x1,y1,a,b,c):
    d=abs((a*x1+b*y1+c))/(math.sqrt(a*a+b*b))
    return d


@app.route('/reports', methods=['GET', 'POST'])
def download_reports():
    csv = request.args.get('csv')
    graph = request.args.get('graph')
    with ZipFile('my_python_files.zip','w') as zipa:
        zipa.write(csv)
        zipa.write(graph)
    return send_file('my_python_files.zip', attachment_filename='Results.zip', as_attachment=True)

@app.route('/freeup_server_storage', methods=['GET', 'POST'])
def empty_storage():
    folder = 'uploads'
    for filename in os.listdir(folder):
        if filename!="dummy.txt":
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                return """Failed"""
    return """Server Storage Free up successful :)"""

if __name__ == '__main__':
    app.run(debug=True, threaded=False)
