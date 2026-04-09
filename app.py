
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from google.colab import drive

st.set_page_config(page_title="News Classifier", page_icon="📰", layout="centered")

st.title("📰 News Topic Classifier — BERT")
st.markdown("Classify news headlines into **World, Sports, Business, Sci/Tech** categories!")

# Model load
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('./bert_news_final')
    model = BertForSequenceClassification.from_pretrained('./bert_news_final')
    model.eval()
    return tokenizer, model

label_names = ['🌍 World', '⚽ Sports', '💼 Business', '💻 Sci/Tech']
colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4']

with st.spinner("⏳ Loading BERT model..."):
    tokenizer, model = load_model()
st.success("✅ Model loaded!")

# Input
headline = st.text_area("Enter a news headline:", 
    placeholder="e.g. Apple announces new iPhone with AI features")

if st.button("🔍 Classify"):
    if headline.strip():
        inputs = tokenizer(headline, return_tensors='pt', 
                          truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        pred = torch.argmax(probs).item()
        
        st.markdown("---")
        st.markdown(f"### 🏷️ Predicted Category: **{label_names[pred]}**")
        st.markdown("### 📊 Confidence Scores:")
        
        for i, (label, prob) in enumerate(zip(label_names, probs)):
            st.progress(float(prob), text=f"{label}: {prob:.1%}")
    else:
        st.warning("Please enter a headline!")
