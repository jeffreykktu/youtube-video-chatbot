# Youtube Video Chatbot 

A chatbot that allows users to ask questions based on the content of a YouTube video.
Provide a YouTube URL, the video is transcribed, and users can pose questions to the chatbot to receive answers derived from the video content.
![image](https://github.com/jeffreykktu/youtube-video-chatbot/assets/42402011/38af4ba3-1f4b-44ff-a9f4-a03fe68eec0f)


### Installation
1. Clone the Repository:
```
git clone [repository-url]
cd [repository-directory]
```
2. Install Dependencies:
```
pip install -r requirements.txt
```
3. Setup Environment Variables:
Create a .env file in the root directory and set up your OpenAI API key:
```
OPENAI_API_KEY=your_openai_key
```

### Usage
1. Run the Streamlight App:
```
streamlit run main.py
```
2. Input the YouTube URL in the sidebar.
3. Click "Process URL" to transcribe and process the video.
4. Once processed, ask questions about the video in the main input field.

Notes: 
- The YouTube Video Transcript will be saved to your directory.
- The FAISS index will be saved in a local file path in pickle format for future use.
