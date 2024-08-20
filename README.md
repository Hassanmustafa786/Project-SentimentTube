# Project-SentimentTube

Week-6: Project

Idea: Train a Language Model to see if the overall intent / sentiment of a video is positive or negative. You can pass it URL of any video, and it will predict (from the URL) if the video is positive, neurtal, and negative

TODOs:
Data Collection
- Transcribe enough Youtube videos and manually label them as positive or negative. You can write code to automatically download and transcribe the youtube video. (Please do your research on it, should be fairly simple to find resources and code)
- Label Each video
Model Training
- Choice of language model and why (GPT, T5, BERT, BART, Flan-T5 etc.)
- Should you train it from scratch or use a pre-trained language model and fine-tune it.
- Monitor the training and evaluation metrics during training / fine-tuning
Model Deployment
- Create a complete pipeline of your project to deploy it to HuggingFace Spaces using Gradio.(ref article: https://medium.com/@obandoandrew8/deploying-a-ml-model-with-gradio-and-hugging-face-python-machine-learning-83f076c58a0c)
- Your inference pipeline should be like: I should paste transcript of any youtube video and it should classify the intent. 

Article
- Write an article on this project explaining your code
Github
- Upload your final pipeline on Github. Make sure to make your code well written and follow general Python coding practices.

Guidelines
- You can not use LLMs and API based models
- Your code should be reproducible
- Your code should be able to run on Colab free version.
