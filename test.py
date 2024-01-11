from transformers import pipeline

distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    top_k=None
)
emo = distilled_student_sentiment_classifier("我喜歡你")[0][0]['label']
emoget = emo
print(emoget)


