 #REST API:
from flask import Flask
app = Flask(__name__)#referance the file

@app.route("/patient_report/<string:id>/<int:du>")

def hello(id , du):
    
    
    #!/usr/bin/env python
    # coding: utf-8

    # In[1]:


    import datetime
    import pandas as pd
    import numpy as np
    import firebase_admin
    from firebase_admin import credentials
    from firebase_admin import firestore
    from firebase_admin import storage
    import pyrebase

    from datetime import date, timedelta
    import urllib.request, json 
    import time
    #get_ipython().run_line_magic('matplotlib', 'inline')
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import os
    import csv
    from IPython.display import display
    from Model import trainData
    import random
    #from google.cloud import storage

    from matplotlib.patches import Ellipse
    import seaborn as sns

    # signal processing
    from scipy import signal
    from scipy.ndimage import label
    from scipy.stats import zscore
    from scipy.interpolate import interp1d
    from scipy.integrate import trapz

    # misc
    import warnings

    #generate pdf
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import Color, lightblue, black, HexColor


    # In[2]:


    cred = credentials.Certificate("/Users/raghadaziz/Desktop/GP2/SereneReports/SereneReport/serene-firebase-adminsdk.json")
    app = firebase_admin.initialize_app(cred ,  {
        'storageBucket': 'serene-2dfd6.appspot.com',
    }, name='[DEFAULT]')
    db = firestore.client()


    # In[3]:


    duration = du
    userID = id #"UqTdL3T7MteuQHBe1aNfSE9u0Na2"


    # In[4]:


    today = datetime.datetime.now()
    timestamp = today.strftime("%Y-%m-%d %H:%M:%S")
    bucket = storage.bucket(app=app)


    # ## Get data from storage and get list of dates 

    # In[5]:


    dates =[]
    for x in range(0 ,duration):
        today=date.today() 
        yesterday = today - datetime.timedelta(days=1)
        start_date = (yesterday-timedelta(days=duration-x)).isoformat()
        dates.append(start_date)


    # In[6]:


    df= pd.DataFrame()
    # loop through the storage and get the data
    sleep =[]
    for x in range(0 ,len(dates)):
        #Sleep
        blob = bucket.blob(userID+"/fitbitData/"+dates[x]+"/"+dates[x]+"-sleep.json")
        # download the file 
        u = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
        try:
            with urllib.request.urlopen(u) as url:
                data = json.loads(url.read().decode())
                sleepMinutes = data['summary']["totalMinutesAsleep"]
        except:
            pass

        #Activity (Steps)
        blob = bucket.blob(userID+"/fitbitData/"+dates[x]+"/"+dates[x]+"-activity.json")
        # download the file 
        u = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
        try:
            with urllib.request.urlopen(u) as url:
                data = json.loads(url.read().decode())
                steps = data['summary']["steps"]
        except:
            pass

        #heartrate
        blob = bucket.blob(userID+"/fitbitData/"+dates[x]+"/"+dates[x]+"-heartrate.json")
        u = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
        try:
            with urllib.request.urlopen(u) as url:
                data = json.loads(url.read().decode())
                df_heartrate = pd.DataFrame(data['activities-heart-intraday']['dataset'])

            df_heartrate.time.apply(str)
            df_heartrate['time'] = pd.to_datetime(df_heartrate['time'])
            df_heartrate['hour'] = df_heartrate['time'].apply(lambda time: time.strftime('%H'))
            df_heartrate.drop(['time'],axis=1, inplace = True)
            heart_rate = df_heartrate.groupby(["hour"], as_index=False).mean()
            heart_rate['sleepMin'] = sleepMinutes
            heart_rate['TotalSteps'] = steps
            heart_rate['date'] = dates[x]
            heart_rate = heart_rate.astype({"hour": int})  
        except:
            pass

        # append dataframe
        df = df.append(heart_rate, ignore_index = True)


    # In[7]:


    df


    # ### Get user location

    # In[8]:


    # get location from database
    loc_df = pd.DataFrame()
    locID = []
    locations = db.collection(u'PatientLocations').where(u'patientID', u'==', userID ).stream()

    for location in locations:
        loc = location.to_dict()
        locID.append(location.id)
        loc_df = loc_df.append(pd.DataFrame(loc,index=[0]),ignore_index=True)

    loc_df['id'] = locID


    # In[9]:


    loc_df.drop(['anxietyLevel', 'lat','lng', 'patientID'  ], axis=1, inplace = True)


    # In[10]:


    loc_df.time.apply(str)
    loc_df['time'] = pd.to_datetime(loc_df['time'])
    loc_df['date'] = pd.to_datetime(loc_df['time'], format='%Y:%M:%D').dt.date
    loc_df['hour'] = loc_df['time'].apply(lambda time: time.strftime('%H'))
    loc_df.drop(['time'], axis=1, inplace = True)
    loc_df.hour = loc_df.hour.astype(int) 
    loc_df.date = loc_df.date.astype(str)
    df.date = df.date.astype(str)


    # In[11]:


    dfinal = pd.merge(left=df, 
                      right = loc_df,
                      how = 'left',
                      left_on=['hour','date'],
                      right_on=['hour','date']).ffill()


    # ### Test data into model

    # In[12]:


    #test model 
    train_df = dfinal.rename(columns={'value': 'Heartrate'})


    # In[13]:


    Labeled_df = pd.DataFrame()
    Labeled_df = trainData(train_df)


    # In[14]:


    Labeled_df.drop(['lon'],axis=1, inplace = True)


    # In[15]:


    # Replace missing values because it doesn't exist
    Labeled_df['name'].fillna("Not given", inplace=True)
    Labeled_df['id'].fillna("Not given", inplace=True)


    # In[16]:


    # Update firebase with the user anxiety level 
    for row in Labeled_df.itertuples():
        if row.id != 'Not given':
            if row.Label == 'Low' or row.Label == 'LowA':
                anxietyLevel = 1
            elif row.Label == 'Meduim':
                anxietyLevel = 2
            else:
                anxietyLevel = 3 
            doc_ref = db.collection(u'PatientLocations').document(row.id)
            doc_ref.update({
                                u'anxietyLevel':anxietyLevel
                         })


    # ### Show the places with highest anxiety level

    # In[17]:


    # Show the highest level 
    df_high = pd.DataFrame()
    df_high = Labeled_df[Labeled_df.Label == 'High']


    # In[18]:


    df_high.head(5)


    # # Improvements

    # # Recommendation

    # In[19]:


    docDf = pd.DataFrame()
    doc_ref = db.collection(u'Patient').document(userID)
    doc = doc_ref.get().to_dict()
    docDf = docDf.append(pd.DataFrame(doc,index=[0]),ignore_index=True)


    # In[20]:


    age1 = docDf['age'].values
    name1 = docDf['name'].values
    emp1 = docDf['employmentStatus'].values
    mar1 = docDf['maritalStatus'].values
    income1 = docDf['monthlyIncome'].values
    chronicD1 = docDf['chronicDiseases'].values
    smoke1 = docDf['smokeCigarettes'].values
    gad1 = docDf['GAD-7ScaleScore'].values

    age = age1[0] 
    name = name1[0]
    emp = emp1[0]
    mar = mar1[0]
    income = income1[0]
    chronicD = chronicD1[0]
    smoke = smoke1[0]
    gad = gad1[0]


    compareAge = int(age)


    # In[21]:


    sleepMin = Labeled_df['sleepMin'].mean()
    totalSteps = Labeled_df['TotalSteps'].mean()

    sleepRecomendation = False
    stepsRecomendation = False
    recomendedSteps = 'No recomendation'

    if sleepMin < 360:
        sleepRecomendation = True
    if compareAge < 20 and compareAge > 11:
        if totalSteps < 6000:
            stepsRecomendation = True
            recomendedSteps = '6000'
    if compareAge < 66 and compareAge > 19:  
         if totalSteps < 3000:
            stepsRecomendation = True
            recomendedSteps = '3000'

    sleepMin = sleepMin / 60


    float("{:.2f}".format(sleepMin))
    float("{:.2f}".format(totalSteps))


    # In[22]:


    # store recomendation in database
    ID = random.randint(1500000,10000000)
    doc_rec = db.collection(u'LastGeneratePatientReport').document(str(ID))
    doc_rec.set({
        u'steps': totalSteps,
        u'patientID':userID,
        u'sleepMin': sleepMin,
        u'sleepRecomendation': sleepRecomendation,
        u'stepsRecomendation': stepsRecomendation,
        u'recommended_steps': recomendedSteps
    })


    # ## Storage intilization

    # In[113]:


    firebaseConfig = {
        "apiKey": "AIzaSyBoxoXwFm9TuFysjQYag0GB1NEPyBINlTU",
        "authDomain": "serene-2dfd6.firebaseapp.com",
        "databaseURL": "https://serene-2dfd6.firebaseio.com",
        "projectId": "serene-2dfd6",
        "storageBucket": "serene-2dfd6.appspot.com",
        "messagingSenderId": "461213981433",
        "appId": "1:461213981433:web:62428e3664182b3e58e028",
        "measurementId": "G-J66VP2Y3CR"
      }

    firebase = pyrebase.initialize_app(firebaseConfig)
    storage = firebase.storage()


    # # AL

    # In[114]:


    # Change Label values to num, to represent them in a barchart
    nums=[]
    for row in Labeled_df.itertuples():
        if row.Label == 'Low' or row.Label == 'LowA':
            nums.append(1)
        elif row.Label == 'Meduim':
            nums.append(2)
        else:
            nums.append(3)
    Labeled_df['numLabel'] = nums


    # In[115]:


    # Get anxiety level by day and store it in a new data frame
    plot_df = pd.DataFrame()
    avgAnxiety = []
    totalAnxiety = 0
    rowCount = 1
    for x in range(0 ,len(dates)):
        for row in Labeled_df.itertuples():
            if (row.date == dates[x]):
                rowCount += 1
                totalAnxiety += row.numLabel
        avgAnxiety.append(totalAnxiety/rowCount)


    plot_df['date'] = dates
    plot_df['Anxiety'] = avgAnxiety


    # ## To generate graphs for Android application

    # In[116]:


    #divide dataframe into 15 rows (2 weeks)

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    df5 = pd.DataFrame()
    df6 = pd.DataFrame()
    df7 = pd.DataFrame()
    df8 = pd.DataFrame()
    df9 = pd.DataFrame()
    df10 = pd.DataFrame()
    df11 = pd.DataFrame()
    df12 = pd.DataFrame()
    dfarray = []
    count = 0
    if(len(plot_df) > 15):
        df1 = plot_df[:15]
        df2 = plot_df[15:]
        dfarray.append(df1)
        dfarray.append(df2)
        if(len(df2)>15):
            count = (df2.last_valid_index() - (len(df2) - 15))
            df3 = df2[count:]
            dfarray.append(df3)
            if(len(df3)>15):
                count = (df3.last_valid_index() - (len(df3) - 15))
                df4 = df3[count:]
                dfarray.append(df4)
                if(len(df4)>15):
                    count = (df4.last_valid_index() - (len(df4) - 15))
                    df5 = df4[count:]
                    dfarray.append(df5)
                    if(len(df5)>15):
                        count = (df5.last_valid_index() - (len(df5) - 15))
                        df6 = df5[count:]
                        dfarray.append(df6)
                        if(len(df6)>15):
                            count = (df6.last_valid_index() - (len(df6) - 15))
                            df7 = df6[count:]
                            dfarray.append(df7)
                            if(len(df7)>15):
                                count = (df7.last_valid_index() - (len(df7) - 15))
                                df8 = df7[count:]
                                dfarray.append(df8)
                                if(len(df8)>15):
                                    count = (df8.last_valid_index() - (len(df8) - 15))
                                    df9 = df8[count:]
                                    dfarray.append(df9)
                                    if(len(df9)>15):
                                        count = (df9.last_valid_index() - (len(df9) - 15))
                                        df10 = df9[count:]
                                        dfarray.append(df10)
                                        if(len(df10)>15):
                                            count = (df10.last_valid_index() - (len(df10) - 15))
                                            df11 = df10[count:]
                                            dfarray.append(df11)
                                            if(len(df11)>15):
                                                count = (df11.last_valid_index() - (len(df11) - 15))
                                                df12 = df11[count:]
                                                dfarray.append(df12)



    # In[117]:


    # Plot AL
    if(len(plot_df)<15):
        fig, ax = plt.subplots()

        # Draw the stem and circle
        ax.stem(plot_df.date, plot_df.Anxiety, basefmt=' ')
        plt.tick_params(axis='x', rotation=70)

        # Start the graph at 0
        ax.set_ylim(0, 3)
        ax.set_title('Anxiety level (Throughout week)')
        plt.xlabel('Date')
        plt.ylabel('Low        Meduim        High', fontsize= 12)
        ax.yaxis.set_label_coords(-0.1, 0.47)

        (markers, stemlines, baseline) = plt.stem(plot_df.date, plot_df.Anxiety)
        plt.setp(stemlines, linestyle="-", color="#4ba0d1", linewidth=2)
        plt.setp(markers,  marker='o', markersize=5, markeredgecolor="#4ba0d1", markeredgewidth=1)
        plt.setp(baseline, linestyle="-", color="#4ba0d1", linewidth=0)

        conv = str(x)
        fig.savefig('AL.png', dpi = 100)
        imagePath = 'AL.png'
        storage.child(userID+"/lastGeneratedPatientReport/AL.png").put('AL.png')
        os.remove('AL.png')


    else:   
        for x in range(0,len(dfarray)):
            fig, ax = plt.subplots()

            # Draw the stem and circle
            ax.stem(dfarray[x].date, dfarray[x].Anxiety, basefmt=' ')
            plt.tick_params(axis='x', rotation=70)

            # Start the graph at 0
            ax.set_ylim(0, 3)
            ax.set_title('Anxiety level (Throughout week)')
            plt.xlabel('Date')
            plt.ylabel('Low        Meduim        High', fontsize= 12)
            ax.yaxis.set_label_coords(-0.1, 0.47)

            (markers, stemlines, baseline) = plt.stem(dfarray[x].date, dfarray[x].Anxiety)
            plt.setp(stemlines, linestyle="-", color="#4ba0d1", linewidth=2)
            plt.setp(markers,  marker='o', markersize=5, markeredgecolor="#4ba0d1", markeredgewidth=1)
            plt.setp(baseline, linestyle="-", color="#4ba0d1", linewidth=0)


            conv = str(x)
            fig.savefig('ALP'+str(x)+'.png', dpi = 100)
            imagePath = 'ALP'+str(x)+'.png'
            storage.child(userID+"/lastGeneratedPatientReport/ALP"+str(x)+'.png').put('ALP'+str(x)+'.png')
            os.remove('ALP'+str(x)+'.png')



    # ## To generate graphs for PDF report

    # In[108]:


    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    dfarray = []
    count = 0
    if(len(plot_df) > 90):
        df1 = plot_df[:90]
        df2 = plot_df[90:]
        dfarray.append(df1)
        dfarray.append(df2)


    # In[111]:


    # Plot AL
    if(len(plot_df)<=90):

        fig, ax = plt.subplots()

        # Draw the stem and circle
        ax.stem(plot_df.date, plot_df.Anxiety, basefmt=' ')
        plt.tick_params(axis='x', rotation=70)

        # Start the graph at 0
        ax.set_ylim(0, 3)
        ax.set_title('Anxiety level (Throughout week)')
        plt.xlabel('Date')
        plt.ylabel('Low        Meduim        High', fontsize= 12)
        ax.yaxis.set_label_coords(-0.1, 0.47)

        (markers, stemlines, baseline) = plt.stem(plot_df.date, plot_df.Anxiety)
        plt.setp(stemlines, linestyle="-", color="#4ba0d1", linewidth=2)
        plt.setp(markers,  marker='o', markersize=5, markeredgecolor="#4ba0d1", markeredgewidth=1)
        plt.setp(baseline, linestyle="-", color="#4ba0d1", linewidth=0)



        conv = str(x)
        fig.savefig('ALpdf.png', dpi = 100)

    else:    
        for x in range(0,len(dfarray)):
            fig, ax = plt.subplots()

            # Draw the stem and circle
            ax.stem(dfarray[x].date, dfarray[x].Anxiety, basefmt=' ')
            plt.tick_params(axis='x', rotation=70)

            # Start the graph at 0
            ax.set_ylim(0, 3)
            ax.set_title('Anxiety level (Throughout week)')
            plt.xlabel('Date')
            plt.ylabel('Low        Meduim        High', fontsize= 12)
            ax.yaxis.set_label_coords(-0.1, 0.47)

            (markers, stemlines, baseline) = plt.stem(dfarray[x].date, dfarray[x].Anxiety)
            plt.setp(stemlines, linestyle="-", color="#4ba0d1", linewidth=2)
            plt.setp(markers,  marker='o', markersize=5, markeredgecolor="#4ba0d1", markeredgewidth=1)
            plt.setp(baseline, linestyle="-", color="#4ba0d1", linewidth=0)



            fig.savefig('AL'+str(x)+'pdf.png', dpi = 100)





    # # Location Analysis

    # In[41]:


    loc = pd.DataFrame()
    loc = Labeled_df[Labeled_df.name != 'Not given']


    # In[42]:


    loc.drop(['Heartrate', 'sleepMin','TotalSteps', 'id'  ], axis=1, inplace = True)


    # In[43]:


    names = []
    Name =""
    for row in loc.itertuples():
        Name  = row.name         
        names.append(Name)


    # In[44]:


    new_name =pd.DataFrame()
    new_name ['name']= names


    # In[45]:


    new_name = new_name.drop_duplicates()


    # In[46]:


    new_name


    # In[47]:


    fnames = []
    fName =""
    for row in new_name.itertuples():
        fName  = row.name
        fnames.append(fName)


    # In[61]:


    analysis = pd.DataFrame()
    count = 0
    i = 0
    label = ""
    locationName = ""
    counts = []
    labels = []
    locationNames = []
    for x in range(0,len(fnames)):
        count = 0
        locName = fnames[i]
        for row in loc.itertuples():
            if(locName == row.name):
                if(row.Label=='High'):
                    count+=1
                    label = row.Label
                    locationName = row.name

        i+=1           
        counts.append(count)
        labels.append(label)
        locationNames.append(locationName)

    analysis ['Location'] = locationNames
    analysis ['Frequency'] = counts
    analysis ['Anxiety Level'] = labels


    # In[62]:


    analysis


    # In[63]:


    newA = analysis.drop(analysis[analysis['Frequency'] == 0].index, inplace= True)


    # In[64]:


    analysis


    # In[65]:


    import six


    # In[66]:


    def render_mpl_table(data, col_width=5.0, row_height=0.625, font_size=14,
                         header_color='#23495f', row_colors=['#e1eff7', 'w'], edge_color='#23495f',
                         bbox=[0, 0, 1, 1], header_columns=0,
                        ax=None, **kwargs):


        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')

        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center'  ,**kwargs)

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in  six.iteritems(mpl_table._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
                cell.alignment = 'center'

        fig.savefig('Location.png', dpi = 100)
        return ax


    # In[67]:


    if(len(analysis) > 0):
        render_mpl_table(analysis, header_columns=0, col_width=4)


    # # Genertate patient report and save it in storage

    # In[71]:


    pdf = canvas.Canvas('Patient.pdf')
    pdf.setTitle('Patient report')

    #sleepRecomendation
    #recomendedSteps

    pdf.drawImage("serene .png", 150, 730, width=300,height=130, mask= 'auto')


    pdf.setFillColor(HexColor('#e1eff7'))
    pdf.roundRect(57,400, 485,200,4,fill=1, stroke= 0)

    pdf.setFont("Helvetica-Bold", 16)
    pdf.setFillColor(HexColor('#23495f'))

    pdf.drawString(115,570, "Report Duration From: " + dates[0] +" To: "+ dates[len(dates)-1])

    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(250,540, "Improvments: ")

    pdf.drawString(200,500, "Highest day of anxiety level: ")



    pdf.setFillColor(HexColor('#e1eff7'))
    pdf.roundRect(57,160, 485,200,4,fill=1, stroke= 0)

    pdf.setFont("Helvetica-Bold", 16)
    pdf.setFillColor(HexColor('#23495f'))

    pdf.drawString(130,330, "Recommendations: ")
    pdf.drawString(150,300, "Sleep Recomendation: ")
    pdf.drawString(150,260, "Steps Recomendation: ")

    pdf.setFont("Helvetica", 16)
    pdf.setFillColor(black)

    if(sleepRecomendation == True):
        pdf.drawString(180,280, "we reccomend you to sleep from 7-9 hours")
    else:
        pdf.drawString(180,280, "keep up the good work")
    if(stepsRecomendation == True):
        pdf.drawString(180,240, "we reccomend you to walk at least " + recomendedSteps)
    else:
         pdf.drawString(180,240, "keep up the good work")

    pdf.showPage()


    pdf.drawImage("serene .png", 150, 730, width=300,height=130, mask= 'auto')


    pdf.setFont("Helvetica-Bold", 20)
    pdf.setFillColor(HexColor('#808080'))

    pdf.drawString(100,650, "Anxiety Level")

    if(len(plot_df)<=90):
        pdf.drawImage("ALpdf.png", 57, 400, width=485,height=200)
        pdf.drawString(100,350, "Location Analysis")
        if(len(analysis) > 0):
            pdf.drawImage("Location.png", 57, 100, width=485,height=200)
        else:
            pdf.setFont("Helvetica", 15)
            pdf.setFillColor(HexColor('#23495f'))

            t = pdf.beginText(130,250)
            text = [
            name +" condition was stable through this period,", 
            "no locations with high anxiety level were detected." ]
            for line in text:
                t.textLine(line)

            pdf.drawText(t)
            pdf.showPage()

    else:
        j = 400
        for x in range(0,len(dfarray)):
            pdf.drawImage('AL'+str(x)+'pdf.png', 57, j, width=485,height=200)
            j = j-300
        pdf.showPage()

        pdf.drawImage("serene .png", 150, 730, width=300,height=130, mask= 'auto')

        pdf.setFont("Helvetica-Bold", 20)
        pdf.setFillColor(HexColor('#808080'))
        pdf.drawString(100,650, "Location Analysis")
        if(len(analysis) > 0):
            pdf.drawImage("Location.png", 57, 400, width=485,height=200)
        else:
            pdf.setFont("Helvetica", 15)
            pdf.setFillColor(HexColor('#23495f'))

            t = pdf.beginText(130,550)
            text = [
            name +" condition was stable through this period,", 
            "no locations with high anxiety level were detected." ]
            for line in text:
                t.textLine(line)

            pdf.drawText(t)


    pdf.save()


    # In[ ]:


    #new method
    doct = storage.child(userID+"/lastGeneratedPatientReport/patientReport").put('Patient.pdf')


    # In[73]:


    os.remove('Patient.pdf')
    if(len(plot_df)<=90):
         os.remove('ALpdf.png')
    else:
        for x in range(0,len(dfarray)):
            os.remove('AL'+str(x)+'pdf.png')


    # In[ ]:
    
    return "HI"

if __name__ == "__main__":
    app.run(debug=True)




