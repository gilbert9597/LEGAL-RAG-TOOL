from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import os
import openai

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"
# Initialize ChatGPT model
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Global variable for the conversation chain
conversation = None
us_in =None

# Function to create or update the conversation chain
def create_or_update_conversation_chain(memory_flag: str, memory_limit: int):
    global conversation
    if memory_flag == "yes":
        # Use or update memory
        if conversation is None or not isinstance(conversation.memory, ConversationBufferWindowMemory):
            memory_object = ConversationBufferWindowMemory(k=memory_limit)
            conversation = ConversationChain(
                llm=chat_model,
                memory=memory_object,
                prompt=PromptTemplate(
                    input_variables=["history", "input"],
                    template="The following is a conversation:\n{history}\nUser: {input}\nAI:"
                ),
            )
        else:
            # Update memory limit dynamically
            conversation.memory.k = memory_limit


# Chat loop
def bot(memory_limit, memory_flag, user_input, clear_memory="no"):
    global conversation

    print("user_input:", user_input)
    print("memory_flag:", memory_flag)
    print("memory_limit:", memory_limit)
    print("clear_memory",clear_memory)

    # Create or update the conversation chain
    create_or_update_conversation_chain(memory_flag, memory_limit)

    # Clear memory if required
    # if clear_memory.lower() == "yes" and memory_flag == "yes" and conversation.memory is not None:
    if clear_memory == "yes":

        conversation.memory.clear()
        print("Memory has been cleared!")
        return "Memory has been cleared!"

    # Generate a respons
    if memory_flag =="yes":
        response = conversation.run(user_input)
        print(f"AI: {response}")
        return response
    else:
        try:
            # Make a request to the OpenAI API using GPT-3.5 Turbo
            prompt = f"You are the ai assistant. answer the following Uaser query: Query:{user_input}"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Using the GPT-3.5 turbo model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the assistant's response
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"

        return response

def memory_clean(clear_memory):
    if clear_memory == "yes":
        conversation.memory.clear()
        print("Memory has been cleared!")
        return "Memory has been cleared!"

# Run the chatbot
if __name__ == "__main__":
    memory_limit = 4
    memory_flag = "yes"
    user_input = "Who are you?"
    clear_memory = "no"

    # # Example interactions
    # print(bot(memory_limit, memory_flag, user_input))
    # print(bot(memory_limit, memory_flag, "What is your purpose?"))
    # print(bot(memory_limit, memory_flag, "Tell me about yourself."))
    # print(bot(memory_limit, memory_flag, "Goodbye.", clear_memory="yes"))
