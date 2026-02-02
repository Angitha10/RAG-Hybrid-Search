from pydantic import BaseModel, Field
import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
load_dotenv()



class Payload(BaseModel):
    text: str


class StructuredAnswer(BaseModel):
    answer: str = Field(description="The answer to the user query.")

class AnswwerCuratorAgent:
    def __init__(self, retrieved_content: Payload):

        self.llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
        
        self.retrieved_content = retrieved_content
        # Define the reasoning chain prompt
        self.answer_agent_prompt = PromptTemplate(
            input_variables=["user_query", "content"],

        template="""
        You are an assistant that helps users answer their questions from content from user manuals. Your goal is to be able to verbally  help the users with their queries.
        ### Your Task ###
        You will receive:
        1. User Query {user_query}: A specific question about election guidelines
        2. Pdf Content {content}: Data relevent to user query

        ### Instructions ###
        **Content Guidelines**
        Answer ONLY from the provided manual content - do not add external information

        ### Response Structure ###
        * Start with a clear response to the user's question

        * Do not add any bold texts.
        * Do not add any headings.
        
"""
        )
    
    async def answer(self, user_query):
        chain = self.answer_agent_prompt | self.llm | StrOutputParser()

        print("self.retrieved_content", self.retrieved_content)

        result = await chain.ainvoke({
            "user_query": user_query,
            "content": self.retrieved_content
        })

        return result

