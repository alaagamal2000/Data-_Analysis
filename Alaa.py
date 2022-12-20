import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

appointments=pd.read_csv("C:/Users/elaa2/OneDrive/Desktop/files/noshowappointments-kagglev2-may-2016.csv")
#first: let's see some rows from our dataset
print("dataset first 5 rows",appointments.head())
#showing dataset columns' names
print("dataset columns names:",appointments.columns)
#we may need to change some columns names format
#showing dataset data types
print("dataset data types",appointments.dtypes)
#we may need to change ScheduleDay datatype to date
#showimg dataset samples and columns
print("This dataset has {} rows".format(appointments.shape[0]))
print("This dataset has {} columns".format(appointments.shape[1]))
#showing some dataset' statistical calculations
print("some dataset' statistical calculations",appointments.describe())
#there in no age in negative so we will remove this row in data cleaning process
#showing dataset (datatypes,null values,memory usage,--)
print("showing dataset info",appointments.info())
#Are there any repetitive rows in our data set?No
print("duplicated rows:",appointments.duplicated().sum())
#Are there any null values?No
print("Null values",appointments.isnull().sum())
#Data Cleaning Process part
#first we should remove a row with age -1
print(appointments.drop(index=99832,inplace=True))
print("statistics after drop",appointments["Age"].describe())
#drop unnecessary columns or columns we will probably not use
appointments.drop(["PatientId","AppointmentDay"],axis=1,inplace=True)
#changing columns names format
#let's make all columns names with underscore no dash
appointments.rename(columns={"No-show":"No_show"},inplace=True)
#correcting spelling
appointments.rename(columns={"Hipertension":"Hypertension"},inplace=True)
#correcting spelling
appointments.rename(columns={"Handcap":"Handicap"},inplace=True)
print(appointments.columns)
#changing ScheduledDay data type column
print("ScheduleDay datatype before adjusting",appointments["ScheduledDay"].dtypes)
appointments["ScheduledDay"]=pd.to_datetime(appointments["ScheduledDay"])
print("ScheduleDay datatype after adjusting",appointments["ScheduledDay"].dtypes)
#Exploratory data analysis part
appointments.hist(figsize=(12,12));
plt.ylabel("Number of patients")
plt.show()
#notes about the histogram
#most ages are less than 25 years
#small portion of patients have scholarship
#most of patients don't suffer from diabetes,hypertension
#most of patients are not addicted to alcohol
#more than 95 percent of patients are not handicapped
#about half of patients did not recieve sms message
appointments["ScheduledDay"].hist()
plt.title("Scheduled days")
plt.show()
#scheduled days start from 2016-3

youth=appointments["Age"]<50
old=appointments["Age"]>50
print("youth",youth.mean())
print("old",old.mean())
#notice that mean of younger people who suffer from diseases are more than that of old people

Attend=appointments["No_show"]=="No"
No_Attend=appointments["No_show"]=="Yes"
print("mean of people who came to the appointment",Attend.mean())
print("mean of people who didn't come to the appointment",No_Attend.mean())
#we notice that more people came than who didn't

#Which gender attend the most? no clear relation
##gender attendance
def attend(appointments,col_name,show,noshow):
    appointments[col_name][Attend].value_counts(normalize=True).plot(kind="pie",label="Attend")
    #appointments[col_name][No_Attend].plot(kind="pie",color="green",label="No Attendance")
    plt.legend()
    plt.title("Which gender attend the most")
    plt.xlabel("Gender")
    plt.ylabel("Number of patients")
attend(appointments,"Gender",Attend,No_Attend)
plt.show()
#gender absence
def attend(appointments,col_name,show,noshow):
    appointments[col_name][No_Attend].value_counts(normalize=True).plot(kind="pie",label="No_Attend")
    plt.legend()
    plt.title("Which gender not attended the most")
    plt.xlabel("Gender")
    plt.ylabel("Number of patients")
attend(appointments,"Gender",Attend,No_Attend)
plt.show()



#Do Alcoholic people attend the most or non-alcoholic? Non alcoholic people attend the most,
#but there are small portion of non-alcoholic  don't attend more than alcoholic people so it is not significant enough
def attend(appointments,col_name,show,noshow):
    appointments[col_name][Attend].plot(color="pink",label="Attend",linestyle="solid")
    appointments[col_name][No_Attend].plot(color="grey",label="No Attendance",linestyle="solid")
    plt.legend()
    plt.title("Do Alcoholic people attend the most or non-alcoholic?")
    plt.xlabel("Alcoholism")
    plt.ylabel("Number of patients")
attend(appointments,"Alcoholism",Attend,No_Attend)
plt.show()

#Do people who attend the most are alcoholic and what their age ranges?
appointments[Attend].groupby("Alcoholism")["Age"].mean().plot(kind="bar",color="blue",label="show")
appointments[No_Attend].groupby("Alcoholism")["Age"].mean().plot(kind="bar",color="yellow",label="noshow")
plt.title("relation between alcoholism,Age,Attendance")
plt.xlabel("alcoholism")
plt.ylabel("mean of ages")
plt.legend()
plt.show()
print("statistics about Do people who attend the most are alcoholic and what their age ranges? ")
print(appointments[Attend].groupby("Alcoholism").mean()["Age"])
print(appointments[No_Attend].groupby("Alcoholism").mean()["Age"])
#notice that a large portion of non alcoholic people didn't attend and their ages about under 35
#only small portion of non alcoholic people who attended their ages are above 35
#also only small portion of alcoholic people who attend are more than 45
print("mean of ages of alcoholic and non alcoholic people who attend",appointments[Attend].groupby("Alcoholism")["Age"].mean())
print("mean of ages of alcoholic and non alcoholic people who didn't attend",appointments[No_Attend].groupby("Alcoholism")["Age"].mean())
#people who are non alcoholic and attended are younger than those who are alcoholic and attended
#people who are non alcoholic and didn't attend are younger than those who are alcoholic and didn't attend

#Do people who have Scholarship prefer to attend more than the others?
appointments["Scholarship"][Attend].hist(color="blue",label="Attend")
appointments["Scholarship"][No_Attend].hist(color="orange",label="Don't attend")
plt.legend()
plt.title("Do people who have Scholarship prefer to attend more than the others")
plt.xlabel("Scholarship")
plt.ylabel("Number of patients")
plt.show()
#number of people who don't have scholarship and attended are more than that who attended and have schoralship
#number of people who did not attend and don't have scholarship are more than that who didn't attend and have scholarship
#it seems that scholarship don't affect so much on the attendance

#what people' ranges of ages who suffer from diabetes and hypertension and attend the clinic or not attend?
appointments[Attend].groupby(["Diabetes","Hypertension"])["Age"].mean().plot(kind="bar",color="blue",label="Attend")
appointments[No_Attend].groupby(["Diabetes","Hypertension"])["Age"].mean().plot(kind="bar",color="yellow",label="No attend")
plt.title("what people' ranges of ages who suffer from diabetes and hypertension "
          "and attend the clinic or not attend")
plt.xlabel("Diseases")
plt.ylabel("mean of ages")
plt.legend()
plt.show()
##Relevant statistics to what people' ranges of ages who suffer from diabetes and hypertension and attend the clinic or not attend
print(appointments[Attend].groupby(["Hypertension","Diabetes"]).mean()["Age"])
print(appointments[No_Attend].groupby(["Hypertension","Diabetes"]).mean()["Age"])
#notice that people who don't attend and don't suffer from both diseases are more than that who attend and also don't suffer from any disease
#and their ages are in 30th
##people who only suffer from hypertension their ages are in 60
#people who only suffer from diabetes their ages are in 50
#there is a positive correlation between ages and diseases young people don't suffer from diseases but older people suffer more
#there is no significant relation between the diseases and attendance of people

#Do people who receive sms attend more than that those who don't receive?
appointments["SMS_received"][Attend].hist(label="Attend",color="blue")
appointments["SMS_received"][No_Attend].hist(label="No Attend",color="green")
plt.legend()
plt.title("Do people who receive sms attend more than that those who don't receive")
plt.xlabel("Sms_received")
plt.ylabel("Number of patients")
plt.show()
#notice that number of people who attend and don't recieve sms message are more than that who attend and receive messages


#does neighbourhood affect sms messages?
plt.figure(figsize=(20,20))
appointments[Attend].groupby("Neighbourhood")["SMS_received"].mean().plot(kind="bar",color="blue",label="Attend")
appointments[No_Attend].groupby("Neighbourhood")["SMS_received"].mean().plot(kind="bar",color="yellow",label="No attend")
plt.title("does neighbourhood affect sms messages")
plt.xlabel("neighbourhood")
plt.ylabel("mean of sms")
plt.legend()
plt.show()
#notice that there are only 5 neighbourhoods who receive sms messages and attend may be the clinic should revise sms information for the rest of neighbourhoods
#General conclusion
#there is a positive correlation between ages and diseases young people don't suffer from diseases but older people suffer more
#notice that number of people who attend and don't recieve sms message are more than that who attend and receive messages
#notice that mean of younger people who suffer from diseases are more than that of old people.
#limitation: No clear correlation between alcohol and attendance of patients ,
# also no clear correlation between attendance and gender, there is no significant relation between the diseases and attendance of people


















