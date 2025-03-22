from pydantic import BaseModel, Field
from typing import Dict, Optional, Union
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage

class EmotionalAnalysis(BaseModel):
    emotions: Dict[str, float] = Field(
        ..., default_factory=dict, description="A dictionary of emotions and their probability. The sum of probabilities should be 1"
    )
    dominant_emotion: str = Field(
        ..., description="The most likely emotion. It is the emotion in emotions dict with the highest probability"
    )
    
class DecisionOutput(BaseModel):
    decision: Optional[str] = Field(
        ..., description="""What the emotions think the customer the should do next. Some examples: ask_question, hesitate, take_conversation_off_track, get_frustrated, 
        objection, interested_in_buying, answer_agent_question, continue_conversation""")

class BotBoolean(BaseModel):
    bot_boolean: bool

class ConversationEndCheck(BaseModel):
    call_ended: bool

class ConversationState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_profile: Dict[str, Union[str, int, None]]
    emotional_state: Optional[EmotionalAnalysis] = None
    call_ended: ConversationEndCheck = ConversationEndCheck(call_ended=False)
    bot_boolean: BotBoolean = BotBoolean(bot_boolean=False)
    customer_sys_message: Optional[str] = ""
    decision: DecisionOutput = DecisionOutput(decision="continue_conversation")

class EmotionalDecisionState(ConversationState):
    max_turns: int = 10
    discussion_history: Annotated[list[AnyMessage], add_messages] = [{"role": "human", "content":""}] #[MessagesState(messages=[{"role":"human", "content":""}])] #"messages": [{"role": "human", "content":""}]
    emotion: str = "neutral"