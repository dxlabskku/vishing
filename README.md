# Vishing

Voice phishing as known as phone scam has become detrimental to today’s society. The damage caused by vishing is serious every year, and as the crime technology increases, measures to prevent damage are needed. Previously, several studies have endeavored to detect voice phishing by constructing a whitelist based on phone numbers. In addition, until recently, detection have been conducted focusing on content by converting the conversation into text, based on the process of many studies on text spam detection. However, in order to detect voice phishing based on content, the process of converting voice data into text data is necessary. There may be case of incorrect conversion and recently the content of vishing may be difficult to distinguish because it consists of content that is likely to be in daily life. Therefore, in this paper, we tried to detect voice phishing using voice, not text content. Our contribution as follows.

- This paper shows the effectiveness of voice phishing detection using voice data without text content. 
- It also confirms the length of effective voice phishing detection.
- Finally, we saw the possibility for real-time voice phishing detection.


# Data
We collected vishing dataset, organized as vishing call data and normal call data from [Korean Financial Supervisor Service](https://www.fss.or.kr/fss/main/sub1voice.do?menuNo=200012) and [AI Hub](https://aihub.or.kr/). We segmented the data by 0.1 second the process was coded as `time_split.py` in `preprocess` folder.

# Model
We used light models for detection because we tried to detect vishing in real-time using voice data. Machine learning models which take relatively short learning and evaluation time, and basic Deep learning models. `basic.py`, `DenseNet.py` and `LSTM.py` in `Model` folder are the code for experiments. Here are some examples to train and evaluate the models.

**Basic ML**
```
python ml.py --model_name 'SVM' --feature_type 'mfcc' --feature_time '0.5' --wav_path './data/split_wav' --result_root './result' --checkpoints_root './checkpoint' --gpu_id 0
```

**Simple DL**
```
python dl.py --model_name 'DenseNet' --feature_type 'mfcc' --feature_time '0.5' --wav_path './data/split_wav' --result_root './result' --checkpoints_root './checkpoint' --gpu_id 0
```

We set the hyperparmeter of each model as table.

![image](https://user-images.githubusercontent.com/117256746/220051121-0bb9ddeb-f7c1-4601-b647-2d370f4e4382.png)


# Result
- **Test accuracy for all feature and models** 

	The test results are reported the average of five experiments. Most results report above the accuracy 99%
 
	![image](https://user-images.githubusercontent.com/117256746/220046859-029d5d67-cc4e-4428-a070-377882d1dab7.png)

- **Test time per case**

	Next table shows the test time for total time segment by each feature extraction and each applied models. As you can see, every case took less than 0.03 seconds meaning it can be used for real-time detection.
	
	![image](https://user-images.githubusercontent.com/117256746/220048686-124448e0-ca8d-4cea-aa51-91d4294aedae.png)

