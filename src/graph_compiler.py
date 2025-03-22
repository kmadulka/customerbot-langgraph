from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from .classes_defined import ConversationEndCheck, BotBoolean, EmotionalDecisionState
from .nodes import bot_or_human, generate_initial_emotional_opinion, analyze_emotions, generate_emotional_discussion, generate_customer_response,check_conversation_end, agent_response, initiate_all_emotional_opinions

def test():
    return 2

def build_graph():
    # Define graph
    customer_bot_graph = StateGraph(EmotionalDecisionState)

    # Add nodes
    # customer_bot_graph.add_node("check_history", check_history)
    customer_bot_graph.add_node("select_bot_or_human", bot_or_human)
    customer_bot_graph.add_node("generate_initial_emotional_opinion", generate_initial_emotional_opinion)
    customer_bot_graph.add_node("analyze_emotions", analyze_emotions)
    customer_bot_graph.add_node("generate_emotional_discussion", generate_emotional_discussion)
    customer_bot_graph.add_node("generate_customer_response", generate_customer_response)
    customer_bot_graph.add_node("check_conversation_end", check_conversation_end)
    customer_bot_graph.add_node("agent_response", agent_response)

    # Define edges bot_or_human
    customer_bot_graph.add_edge(START, "select_bot_or_human")
    customer_bot_graph.add_conditional_edges(
        "select_bot_or_human",
        lambda state: "analyze_emotions" if state.bot_boolean== BotBoolean(bot_boolean=False) else "agent_response",
        ["agent_response", "analyze_emotions"]
    )

    customer_bot_graph.add_conditional_edges(
        "analyze_emotions", 
        initiate_all_emotional_opinions, 
        ["generate_initial_emotional_opinion"]  # Possible destinations
    )

    customer_bot_graph.add_edge("generate_initial_emotional_opinion", "generate_emotional_discussion")
    customer_bot_graph.add_edge("generate_emotional_discussion", "generate_customer_response")
    customer_bot_graph.add_edge("generate_customer_response", "check_conversation_end")
    customer_bot_graph.add_conditional_edges(
        "check_conversation_end",
        lambda state: END if state.call_ended== ConversationEndCheck(call_ended=True) else "agent_response",
        ["agent_response", END]
    )
    customer_bot_graph.add_edge("agent_response", "analyze_emotions")  # Loop back

    # Compile

    return customer_bot_graph
    # memory = MemorySaver()

