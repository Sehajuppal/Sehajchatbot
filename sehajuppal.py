from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.comparisons import LevenshteinDistance
from chatterbot.response_selection import get_random_response
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import subprocess
import os

# Initialize NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize chatbot
chatbot = ChatBot('PythonJava Expert',
                  logic_adapters=[
                      {
                          'import_path': 'chatterbot.logic.BestMatch',
                          'default_response': 'I am sorry, but I do not understand.',
                          'maximum_similarity_threshold': 0.90
                      },
                      {
                          'import_path': 'chatterbot.logic.SentimentAnalysisAdapter'
                      },
                      {
                          'import_path': 'chatterbot.logic.RuleAdapter'
                      }
                  ],
                  statement_comparison_function=LevenshteinDistance,
                  response_selection_method=get_random_response
                  )

# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot
trainer.train('chatterbot.corpus.english.greetings',
              'chatterbot.corpus.english.conversations',
              'path/to/programming_corpus'  # Replace with the path to your programming-related training corpus
              )

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

# Chat with the bot
while True:
    user_input = input("You: ")

    # Preprocess user input
    user_input = user_input.lower()
    user_input = [lemmatizer.lemmatize(word) for word in user_input.split() if word not in stopwords]
    user_input = ' '.join(user_input)

    # Execute code if user inputs 'run'
    if user_input == 'run':
        code = input("Enter your code: ")
        filename = 'temp.py'
        with open(filename, 'w') as file:
            file.write(code)
        try:
            output = subprocess.check_output(['python', filename], universal_newlines=True)
            print("Output: ", output)
        except subprocess.CalledProcessError as e:
            print("Error: ", e.output)
        finally:
            os.remove(filename)
    else:
        response = chatbot.get_response(user_input)
        print("Bot:", response)

        # Store conversation history
        with open('conversation_history.txt', 'a') as file:
            file.write(f"You: {user_input}\n")
            file.write(f"Bot: {response}\n")

        # User feedback loop
        feedback = input("Was this response helpful? (yes/no): ")
        if feedback.lower() == 'yes':
            chatbot.learn_response(response, user_input)