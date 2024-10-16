import re
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools
from sklearn.metrics import confusion_matrix
import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import altair as alt
import warnings
import pdfkit

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pandas')
warnings.filterwarnings("ignore", category=FutureWarning, module='pandas')

# Disable the PyplotGlobalUseWarning (commented out as it might be causing the issue)
# st.set_option('deprecation.showPyplotGlobalUse', False)

def set_background_image(image_url):
    # Apply custom CSS to set the background image
    page_bg_img = '''
    <style>
    .stApp {
        background-position: top;
        background-image: url(%s);
        background-size: cover;
    }

    @media (max-width: 768px) {
        /* Adjust background size for mobile devices */
        .stApp {
            background-position: top;
            background-size: contain;
            background-repeat: no-repeat;
        }
    }
    </style>
    ''' % image_url
    st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    # Set the background image URL
    background_image_url ="https://images.news18.com/ibnlive/uploads/2023/12/whatsapp-logo-2023-12-0bdd9c0eb2aaab57b916a45ac8c8f352.jpg?impolicy=website&width=640&height=480"
    # Set the background image
    set_background_image(background_image_url)

    custom_css = """
       <style>
       body {
           background-color: #4699d4;
           color: #ffffff;
           font-family: Arial, sans-serif;
       }
       select {
           background-color: #000000 !important; /* Black background for select box */
           color: #ffffff !important; /* White text within select box */
       }
       label {
           color: #ffffff !important; /* White color for select box label */
           font-size: 20px; /* Adjust font size for labels */
           font-weight: bold; /* Make labels bold */
       }
       h1, h2, h3 {
           font-weight: bold; /* Make headings bold */
           font-size: 24px; /* Adjust heading font size */
       }
       .whatsapp-stats {
           font-size: 20px; /* Adjust font size for WhatsApp stats */
           font-weight: bold; /* Make WhatsApp stats bold */
           color: #333333; /* Dark color for WhatsApp stats */
       }
       .stBlock {
           border-right: 2px solid #ffffff; /* Draw a vertical line */
           padding-right: 10px;
           margin-right: 10px;
       }
       .stBlock .stFileUploader {
           border: 1px solid #ffffff; /* Add a border around the file uploader */
           padding: 10px;
           margin: 10px;
       }
       </style>
       """

    st.markdown(custom_css, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


def analyze_whatsapp_chat(uploaded_file):
    # Read the WhatsApp group chat text file
    chat = uploaded_file.getvalue().decode("utf-8")

    # Regular expression patterns
    date_pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2})\s*(am|pm) - (.*?): (.*)$'
    image_pattern = r"<Media omitted>"
    voice_note_pattern = r"\<.*\>"
    video_pattern = r"\<.*\>"
    link_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    emoji_pattern = r"[^\w\s,]"
    deleted_pattern = r"This message was deleted"
    question_pattern = r"\?"

    # Extract dates and messages
    messages_with_dates = re.findall(date_pattern, chat, re.MULTILINE)

    # Extract images
    images = re.findall(image_pattern, chat)
    num_images = len(images)

    # Extract voice notes
    voice_notes = re.findall(voice_note_pattern, chat)
    num_voice_notes = len(voice_notes)

    # Extract videos
    videos = re.findall(video_pattern, chat)
    num_videos = len(videos)

    # Extract links
    links = re.findall(link_pattern, chat)
    num_links = len(links)

    # Extract emojis
    emojis = re.findall(emoji_pattern, chat)
    num_emojis = len(emojis)

    # Extract deleted messages
    deleted_messages = re.findall(deleted_pattern, chat)
    num_deleted_messages = len(deleted_messages)

    # Extract questions
    questions = re.findall(question_pattern, chat)
    num_questions = len(questions)

    # Create DataFrame
    df = pd.DataFrame({'date': [pd.to_datetime(f'{message[0]}, {message[1]} {message[2]}', format='%d/%m/%Y, %I:%M %p', errors='coerce') for message in messages_with_dates],
                       'user': [message[3] for message in messages_with_dates],
                       'message': [message[4] for message in messages_with_dates]})

    # Drop rows with invalid dates (errors='coerce' in pd.to_datetime creates NaT for invalid dates)
    df = df.dropna(subset=['date'])

    # Sentiment analysis
    df['sentiment'] = df['message'].apply(lambda msg: TextBlob(msg).sentiment.polarity)
    sentiment_counts = df['sentiment'].apply(
        lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')).value_counts()

    # Most active day and hour
    most_active_day = df['date'].dt.date.value_counts().idxmax()
    most_active_hour = df['date'].dt.hour.value_counts().idxmax()

    # Average message length
    avg_message_length = df['message'].apply(len).mean()

    # Number of days active
    num_days_active = df['date'].dt.date.nunique()

    return df, num_images, num_voice_notes, num_videos, Counter(df['user']), sentiment_counts, len(df), num_links, num_emojis, num_deleted_messages, most_active_day, most_active_hour, num_questions, avg_message_length, num_days_active

def plot_bar_chart(data, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=data.index, y=data.values, palette='viridis', ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    st.pyplot(fig)

def plot_wordcloud(df):
    text = " ".join(msg for msg in df.message)
    wordcloud = WordCloud(background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

def plot_time_series(df):
    df.set_index('date', inplace=True)
    df['hour'] = df.index.hour
    hourly_counts = df.groupby('hour').size()
    fig, ax = plt.subplots(figsize=(10, 6))
    hourly_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Number of Messages')
    ax.set_title('Activity Over Time')
    st.pyplot(fig)

def download_report(df, num_images, num_voice_notes, num_videos, active_users, sentiment_counts, total_messages, num_links, num_emojis, num_deleted_messages, most_active_day, most_active_hour, num_questions, avg_message_length, num_days_active):
    # Specify the correct path to wkhtmltopdf executable
    config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")  # Update with actual path

    html = f"""
    <h1>WhatsApp Chat Analysis Report</h1>
    <h2>Statistics</h2>
    <p>Total Number of Messages: {total_messages}</p>
    <p>Number of Images: {num_images}</p>
    <p>Number of Voice Notes: {num_voice_notes}</p>
    <p>Number of Videos: {num_videos}</p>
    <p>Number of Links: {num_links}</p>
    <p>Number of Emojis: {num_emojis}</p>
    <p>Number of Deleted Messages: {num_deleted_messages}</p>
    <p>Number of Questions: {num_questions}</p>
    <p>Average Message Length: {avg_message_length:.2f} characters</p>
    <p>Number of Days Active: {num_days_active}</p>
    <p>Most Active Day: {most_active_day}</p>
    <p>Most Active Hour: {most_active_hour}</p>
    <h2>Active Users</h2>
    <ul>
    {"".join([f"<li>{user}: {count}</li>" for user, count in active_users.items()])}
    </ul>
    <h3>Sentiment Distribution</h3>
    <ul>
    {"".join([f"<li>{sentiment}: {count}</li>" for sentiment, count in sentiment_counts.items()])}
    </ul>
    """
    try:
        pdfkit.from_string(html, 'report.pdf', configuration=config)
        with open('report.pdf', 'rb') as f:
            st.download_button('Download Report', f, file_name='report.pdf')

    except IOError as e:
        st.error(f"IOError: {e}. Check if wkhtmltopdf is installed and configured correctly.")
    except Exception as e:
        st.error(f"An error occurred while generating the report: {e}")

st.title("WhatsApp Chat Analyzer")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Chat File")
    uploaded_file = st.file_uploader("Upload a WhatsApp chat text file", type="txt")

if uploaded_file is not None:
    whatsapp_df, num_images, num_voice_notes, num_videos, active_users, sentiment_counts, total_messages, num_links, num_emojis, num_deleted_messages, most_active_day, most_active_hour, num_questions, avg_message_length, num_days_active = analyze_whatsapp_chat(uploaded_file)

    st.write("### Analyzed WhatsApp Chat:")
    st.write(whatsapp_df.head())

    st.write("### Statistics:", unsafe_allow_html=True)
    with st.container():
        st.markdown(f"""
        <div class="whatsapp-stats">Total Number of Messages: {total_messages}</div>
        <div class="whatsapp-stats">Number of Images: {num_images}</div>
        <div class="whatsapp-stats">Number of Voice Notes: {num_voice_notes}</div>
        <div class="whatsapp-stats">Number of Videos: {num_videos}</div>
        <div class="whatsapp-stats">Number of Links: {num_links}</div>
        <div class="whatsapp-stats">Number of Emojis: {num_emojis}</div>
        <div class="whatsapp-stats">Number of Deleted Messages: {num_deleted_messages}</div>
        <div class="whatsapp-stats">Number of Questions: {num_questions}</div>
        <div class="whatsapp-stats">Average Message Length: {avg_message_length:.2f} characters</div>
        <div class="whatsapp-stats">Number of Days Active: {num_days_active}</div>
        <div class="whatsapp-stats">Most Active Day: {most_active_day}</div>
        <div class="whatsapp-stats">Most Active Hour: {most_active_hour}</div>
        <div class="whatsapp-stats">Active Users:</div>
        """, unsafe_allow_html=True)

        for user, count in active_users.items():
            st.markdown(f"<div class='whatsapp-stats'>- {user}: {count} messages</div>", unsafe_allow_html=True)

    # Plotting graphs
    st.write("### Visualization:")
    st.write("#### Number of Messages per User")
    plot_bar_chart(pd.Series(active_users), "User", "Number of Messages", "Number of Messages per User")

    st.write("#### Sentiment Distribution")
    plot_bar_chart(sentiment_counts, "Sentiment", "Frequency", "Sentiment Distribution")

    st.write("#### Word Cloud")
    plot_wordcloud(whatsapp_df)

    st.write("#### Time Series Analysis")
    plot_time_series(whatsapp_df)

    # Download report
    download_report(whatsapp_df, num_images, num_voice_notes, num_videos, active_users, sentiment_counts, total_messages, num_links, num_emojis, num_deleted_messages, most_active_day, most_active_hour, num_questions, avg_message_length, num_days_active)
