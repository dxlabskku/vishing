# Vishing

This paper focuses on detection vishing, also known as voice phishing, a type of scam via phone calls. The damage caused by vishing is deteriorating year by year, thus calling the need of a real-time automatic vishing detection. Various previous studies on vishing detection have used whitelist or text contents converted from speech as the input features. However, such previous works cannot detect vishing rapidly. Therefore, this paper attempts to detect vishing based on direct call voice data without constructing a whitelist or converting the audio data to text data. For effective detection, several signal feature extraction methods for voice data are adopted and various machine and deep learning models are employed. The experiment results show that acoustic features are effective for vishing detection, reporting almost 99% accuracy. In addition, we confirm the effective length of data for detection and explore the possibility of real-time detection using voice data. Our contribution as follows.

- We show that it is possible to directly use voice data without converting it into text for vishing detection. 
- We share the appropriate length of the voice data for effective vishing detection. 
- Finally, we report that real-time vishing detection is possible using voice data with experiments.


# Data
We collected vishing dataset, organized as vishing call data and normal call data from [Korean Financial Supervisor Service](https://www.fss.or.kr/fss/main/sub1voice.do?menuNo=200012) and [AI Hub](https://aihub.or.kr/). We segmented the data by 0.1 second and the codes can be found in `time_split.py` in `preprocess` folder.

**- Examples of dataset**

![image](https://user-images.githubusercontent.com/117256746/220054333-20731d77-630b-4eb0-984c-75c66930ca55.png)


# Model

![image](https://user-images.githubusercontent.com/117256746/220055290-cf5f3099-3785-4232-943c-be2d0b9c0372.png)

Image shows the overview of our vishing detection process. We used light models for detection for real-time vishing detection: Machine learning models, which take relatively short learning and evaluation time and basic Deep learning models. `basic.py`, `DenseNet.py` and `LSTM.py` in `Model` folder are the codes for experiments. Here are some examples for training and evaluating the models. 


**Basic ML**
```
python ml.py --model_name 'SVM' --feature_type 'mfcc' --feature_time '0.5' --wav_path './data/split_wav' --result_root './result' --checkpoints_root './checkpoint' --gpu_id 0
```

**Simple DL**
```
python dl.py --model_name 'DenseNet' --feature_type 'mfcc' --feature_time '0.5' --wav_path './data/split_wav' --result_root './result' --checkpoints_root './checkpoint' --gpu_id 0
```

The hyperparmeters of each model are as follow:

![image](https://user-images.githubusercontent.com/117256746/220051121-0bb9ddeb-f7c1-4601-b647-2d370f4e4382.png)


# Result
- **Test accuracy for all feature and models** 

	The test results are average of five experiment runs. Most of the results report above 99% accuracy.
 
	![image](https://user-images.githubusercontent.com/117256746/220046859-029d5d67-cc4e-4428-a070-377882d1dab7.png)

- **Test time per case**

	The table below shows the test time each time segment for each feature extraction methods for all models. As you can see, every case took less than 0.03 seconds, showing that they can be used for real-time detection.
	
	
	![image](https://user-images.githubusercontent.com/117256746/220589754-b01997f6-740f-4b8b-8e0a-9f83c5ec2628.png)

