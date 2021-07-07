
![Alt text](images/AIWonderGirlsLogo_small.png?raw=true)
# AIWondergirls-ICU-OPS
**_Help frontline clinicians triage patients in ICUs by rapidly assessing a patient's overall health for informed clinical decisions that will improve patient outcomes and relieve COVID19 ICU overload._**

We used the [WIDS 2021 dataset](https://www.kaggle.com/c/widsdatathon2021/data), which focuses on patient health, with an emphasis on the chronic condition of diabetes. An ICU patient's chronic conditions, such as heart disease, injuries, or diabetes, may not be readily available due to the patient's condition or if the patient is from another medical provider or system. Knowing a patient's chronic diseases can expedite clinical decisions about their care and ultimately improve their health outcomes.  The speed-up of patient outcomes relieves Intensive Care Units (ICUs) struggling with overload from critical COVID-19 pandemic cases.  Our AI solution, **ICU-OPS**, _is rapidly channeling medical emergencies in the right hands, leading to the best possible patient outcomes._ 

Data mining in healthcare systems allows organizations to quickly access the latest medical research knowledge to deliver better care to patients.
Using NLP,  we researched papers with the _Diabetes Mellitus condition and its risks in Covid19_ available on the web to build a Question and Answer system for Clinicians to understand this connection better. 
With NLP topic modeling and LDA analysis, the Questioning Answering system is developed with a custom-trained BERT model.  

 NLP dataset source: 

    COVID-19 and diabetes mellitus: from pathophysiology to clinical management
    https://www.nature.com/articles/s41574-020-00435-4.pdf

    Research ArticleClinical Findings in Diabetes Mellitus Patients with COVID-19 
    https://www.hindawi.com/journals/jdr/2021/7830136/

    Prevention and management of COVID-19 among patients with diabetes: an appraisal of the literature'
    https://pubmed.ncbi.nlm.nih.gov/32405783/

We developed the backend on MLRun and the Iguazio platform using nuclio serverless functions. These nuclio functions do data preprocessing, training the model, serving the model with REST APIs. The functions can be run locally or on a cluster with auto_mount(). These are especially helpful for large-scale distributed processing as well as computing. 
The NLP Question Answering system using custom trained BERT model on the NLP dataset, and the LGBM, XGB models are serverless nuclio functions.
The User interface(UI) is built in python using the Streamlit library. The UI can access the REST APIs of the serving functions created in the MLRun. To run the user interface, download the UI directory to the local system, in a virtual env, install the libraries in req.txt and start the app.
        
        pip install -r req.txt 
        streamlit run icuops.py
