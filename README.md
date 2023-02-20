# Vishing

Voice phishing as known as phone scam has become detrimental to today’s society. The damage caused by vishing is serious every year, and as the crime technology increases, measures to prevent damage are needed. Previously, several studies have endeavored to detect voice phishing by constructing a whitelist based on phone numbers. In addition, until recently, detection have been conducted focusing on content by converting the conversation into text, based on the process of many studies on text spam detection. However, in order to detect voice phishing based on content, the process of converting voice data into text data is necessary. There may be case of incorrect conversion and recently the content of vishing may be difficult to distinguish because it consists of content that is likely to be in daily life. Therefore, in this paper, we tried to detect voice phishing using voice, not text content. Our contribution as follows.

- This paper shows the effectiveness of voice phishing detection using voice data without text content. 
- It also confirms the length of effective voice phishing detection.
- Finally, we saw the possibility for real-time voice phishing detection.


### Data
We collected vishing dataset, organized as vishing call data and normal call data from [Korean Financial Supervisor Service](https://www.fss.or.kr/fss/main/sub1voice.do?menuNo=200012) and [AI Hub](https://aihub.or.kr/). We segmented the data by 0.1 second the process was coded as `time_split.py` in `preprocess` folder.

### Result
		0.5	0.6	0.7	0.8	0.9	1	1.1	1.2	1.3	1.4	1.5	1.6	1.7	1.8	1.9	2
SVM	mel	1.0000 	0.9928 	1.0000 	1.0000 	1.0000 	0.9928 	0.9892 	0.9964 	0.9928 	0.9928 	0.9964 	0.9964 	0.9964 	0.9964 	0.9964 	0.9964 
	stft	1.0000 	0.9964 	1.0000 	1.0000 	1.0000 	0.9964 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 
	mfcc	0.9964 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 
LR	mel	1.0000 	0.9784 	0.9928 	1.0000 	1.0000 	0.9892 	0.9964 	0.9964 	0.9928 	0.9928 	0.9964 	0.9928 	0.9928 	1.0000 	0.9964 	0.9964 
	stft	0.9964 	0.9928 	0.9964 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 
	mfcc	0.9964 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 
DT	mel	0.9993 	0.9813 	0.9842 	0.9878 	0.9683 	0.9705 	0.9691 	0.9791 	0.9734 	0.9770 	0.9813 	0.9777 	0.9842 	0.9820 	0.9835 	0.9820 
	stft	0.9950 	0.9978 	1.0000 	0.9993 	1.0000 	0.9993 	0.9971 	1.0000 	0.9971 	0.9986 	1.0000 	0.9993 	0.9971 	0.9993 	0.9986 	0.9993 
	mfcc	0.9986 	0.9971 	0.9971 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	0.9993 	0.9993 
RF	mel	0.9993 	0.9878 	0.9755 	0.9849 	0.9942 	0.9899 	0.9892 	0.9899 	0.9878 	0.9871 	0.9892 	0.9871 	0.9878 	0.9878 	0.9885 	0.9885 
	stft	0.9964 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 	1.0000 
	mfcc	1.0000 	1.0000 	1.0000 	0.9993 	0.9993 	0.9986 	0.9978 	0.9986 	0.9978 	0.9978 	0.9978 	0.9971 	0.9971 	0.9964 	0.9964 	0.9964 
DenseNet	mel	 -	 -	0.9950 	0.9964 	0.9957 	0.9957 	0.9935 	0.9928 	0.9928 	0.9928 	0.9928 	0.9935 	0.9928 	0.9928 	0.9928 	0.9928 
	stft	 -	 -	0.9986 	0.9978 	0.9957 	0.9964 	0.9957 	0.9964 	0.9964 	0.9971 	0.9964 	0.9971 	0.9964 	0.9957 	0.9964 	0.9950 
	mfcc	 -	 -	1.0000 	0.9971 	0.9971 	0.9971 	0.9978 	0.9978 	0.9978 	0.9993 	0.9993 	1.0000 	0.9993 	0.9993 	1.0000 	1.0000 
LSTM	mel	0.9993 	0.9935 	0.9899 	0.9863 	0.9777 	0.9784 	0.9799 	0.9784 	0.9705 	0.9633 	0.9669 	0.9784 	0.9734 	0.9741 	0.9640 	0.9727 
	stft	1.0000 	0.9971 	0.9971 	0.9950 	0.9964 	0.9957 	0.9928 	0.9957 	0.9921 	0.9935 	0.9914 	0.9921 	0.9942 	0.9957 	0.9906 	0.9906 
	mfcc	0.9978 	0.9993 	0.9957 	0.9971 	0.9978 	0.9957 	0.9950 	0.9971 	0.9978 	0.9971 	0.9971 	0.9964 	0.9964 	0.9957 	0.9964 	0.9928 
![image](https://user-images.githubusercontent.com/117256746/220044670-b0baef42-a065-41f4-9934-51f5744c4e5c.png)
