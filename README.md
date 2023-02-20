# Vishing

Voice phishing as known as phone scam has become detrimental to today’s society. The damage caused by vishing is serious every year, and as the crime technology increases, measures to prevent damage are needed. Previously, several studies have endeavored to detect voice phishing by constructing a whitelist based on phone numbers. In addition, until recently, detection have been conducted focusing on content by converting the conversation into text, based on the process of many studies on text spam detection. However, in order to detect voice phishing based on content, the process of converting voice data into text data is necessary. There may be case of incorrect conversion and recently the content of vishing may be difficult to distinguish because it consists of content that is likely to be in daily life. Therefore, in this paper, we tried to detect voice phishing using voice, not text content. Our contribution as follows.

- This paper shows the effectiveness of voice phishing detection using voice data without text content. 
- It also confirms the length of effective voice phishing detection.
- Finally, we saw the possibility for real-time voice phishing detection.


### Data
We collected vishing dataset, organized as vishing call data and normal call data from [*Korean Financial Supervisor Service](https://www.fss.or.kr/fss/main/sub1voice.do?menuNo=200012) and [*AI Hub](https://aihub.or.kr/). We segmented the data by 0.1 second the process was coded as `time_split.py` in `preprocess` folder.
