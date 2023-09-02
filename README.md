# Vishing
This paper introduces a real-time vishing (voice phishing) detection method specifically designed for interactive calls, emphasizing the critical need for early detection to mitigate financial losses. Focused solely on acoustic voice features, not conversation content, this study capitalizes on distinctive phonetic traits exhibited by vishing perpetrators. Experimentation yields promising results: 1) Acoustic voice features effectively detect vishing in conversational contexts, 2) early detection is feasible via significant data time length analysis, and 3) the model demonstrates quick inference times suitable for real-time vishing prevention. The experimental models in our study exhibit impressive accuracy rates, and some even achieve perfection. This approach presents a powerful means of real-time vishing prevention, effectively mitigating the potential for financial devastation.


# Data
We collected vishing dataset, organized as vishing call data and normal call data from [Korean Financial Supervisor Service](https://www.fss.or.kr/fss/main/sub1voice.do?menuNo=200012) and [AI Hub](https://aihub.or.kr/). We segmented the data by 0.1 second and the codes can be found in `time_split.py` in `preprocess` folder. On average, the conversation starts within 0.5 seconds, so it is meaningless to use data shorter than 0.5 seconds, and we did not remove the front silence, before 0.5 seconds, due to see how short data can be detected in the actual call. 

**- Examples of dataset**

![image](https://user-images.githubusercontent.com/117256746/220054333-20731d77-630b-4eb0-984c-75c66930ca55.png)


# Model
![overview](https://github.com/dxlabskku/vishing/assets/117256746/9bcbd101-961e-4418-9683-2c44c26dcbc0)

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

