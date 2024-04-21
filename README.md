introduction

Sepsis is a life-threatening medical condition caused by an immune response to a serious infection.
Every year about 1.7 million people in the US develop sepsis and of those about 270,000 die.
Early prediction of sepsis and starting medical treatment accordingly are critical and can save many lives.

data

The data we will use was collected from intensive care patients from two US hospitals.
For each patient we have a file (PSV (values ​​separated-Pipe) containing demographic and medical data about the patient (dictionary)
Variables are attached at the end of the sheet (where each row represents data collected during one hour. The rows are sorted by hours,
When the first row can be considered as the first hour of the patient's arrival at the intensive care unit and the last hour as the hour in which
The patient left intensive care for some reason.

task

Your goal is to predict whether a patient in intensive care suffers from sepsis approximately 6 hours before being identified as suffering from sepsis, based on
Clinical data about his medical condition over time.
Each table, representing a patient, contains a column called SepsisLabel. The value in this column is 1 if the row in which the value appears is up to about 6
hours before the detection of sepsis and 0 otherwise.
If the patient did not suffer from sepsis at all, the column will contain only zeros. A patient will be tagged
As suffering from sepsis (by you) if there is a line in it 1=SepsisLabel.
Also, you must process the tables so that the input to the prediction model does not contain rows that contain data after the first row in
1=SepsisLabel and will not contain the SepsisLabel column.

Notes for the exercise

The py.predict file should call a trained model that knows how to receive any data set in the structure of the attached data files - they are
for the normal part and the competitive part of the exercise.
• Do not train models in the py.predict script. You can save a pre-trained model in your repository and read it from
the script.
• You must not enrich the data set with external sources of information. You must provide all the training code for your model in
repository.
