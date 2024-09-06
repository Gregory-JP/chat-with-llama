from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_groq import ChatGroq

class QnA:
    def __init__(self, file_path, model_name, api_key):
        # Carregar o arquivo de texto
        document_text = self.load_txt_file(file_path)

        # Dividir o texto em fragmentos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(document_text)]

        # Criar embeddings para os fragmentos
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = FAISS.from_documents(documents, embeddings)

        # Configurar o modelo de linguagem
        self.llm = ChatGroq(
            temperature=0,
            model_name='llama-3.1-70B-versatile',
            api_key=api_key,
            model_kwargs={
                'top_p': 0.999,
            }
        )
    
        # Criar uma cadeia de recuperação conversacional
        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.vectorstore.as_retriever(),
            return_source_documents=True
        )

        # Inicializar o historico do chat
        self.chat_history = []

    def load_txt_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
        
    def get_answer(self, user_input):
        system_prompt = """
                        Você é um assistente experiente e inteligente, capaz de responder qualquer pergunta sobre o Brasil.
                        Responda apenas com informações verdadeiras e relevantes, contidas no texto fornecido.
                        Não responda com informações que não estão no texto, e perguntas que não são relacionadas ao Brasil.
                        Se o usuário perguntar algo que não está no texto, você pode dizer que não sabe.
                        """
        
        full_input = f'{system_prompt}\n\nUser: {user_input}'

        # Processar a consulta
        result = self.chain({'question': full_input, 'chat_history': self.chat_history})

        # Atualizar o historico de chat
        self.chat_history.append((user_input, result['answer']))

        # Retornar a resposta
        return result['answer']