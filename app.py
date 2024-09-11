from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

class DocumentChatbot:
    def __init__(self, model_name='distilbert-base-uncased-distilled-squad'):
        # Load pre-trained model and tokenizer
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load a model for embedding the document
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Changed model for better similarity
        self.document = []
        self.embedded_document = []

    def load_document_from_file(self, file_path):
        """Load and embed the document from a file and split into sentences."""
        with open(file_path, 'r') as file:
            # Split the document into individual sentences
            self.document = file.read().split('. ')  # Changed to sentence-level splitting

        # Embed each sentence individually
        self.embedded_document = self.embedding_model.encode(self.document, convert_to_tensor=True)

    def answer_question(self, question):
        """Answer a question based on the loaded document."""
        if not self.document:
            return "No document loaded."

        # Generate embeddings for the question
        question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)

        # Find the most relevant sentence by calculating cosine similarity
        cosine_scores = util.pytorch_cos_sim(question_embedding, self.embedded_document)
        best_sentence_idx = cosine_scores.argmax()  # Get index of the best matching sentence
        best_score = cosine_scores[0][best_sentence_idx].item()

        # Adjust threshold as needed (lowered for flexibility)
        if best_score < 0.2:  # Reduced threshold for broader matches
            return "I couldn't find a relevant answer in the document."

        # Get the best matching sentence
        best_sentence = self.document[best_sentence_idx]

        # Tokenize the input with the best matching sentence
        inputs = self.tokenizer.encode_plus(question, best_sentence, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        # Get model outputs
        outputs = self.model(**inputs)
        answer_start = outputs.start_logits.argmax()
        answer_end = outputs.end_logits.argmax() + 1

        # Convert token IDs back to text
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        return answer or "Sorry, I couldn't find the exact answer."

# Example usage
if __name__ == "__main__":
    chatbot = DocumentChatbot()

    # Load the document from a file
    chatbot.load_document_from_file('chatbot_document.txt')

    # Interact with the chatbot
    while True:
        question = input("Ask a question: ")
        if question.lower() in ['exit', 'quit']:
            break
        answer = chatbot.answer_question(question)
        print("Answer:", answer)
