from .classes_defined import ConversationState, EmotionalAnalysis, EmotionalDecisionState, BotBoolean, DecisionOutput, ConversationEndCheck
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from .prompts import analyze_emotion_instructions, initiate_emotional_opinion_sys_message, emotional_discussion_sys_message, end_convo_instructions
from .config.llm_config import llm 
from langgraph.constants import Send
from langgraph.types import interrupt

# Node: Analyze emotional state
def analyze_emotions(state: EmotionalDecisionState):
    """Generate emotional state based on conversation history """
    print("analyze emotions state")
    print("-------")
    state.discussion_history = [{"role": "human", "content":""}]

    conversation_history = state.messages
    customer_profile = state.customer_profile
    if state.emotional_state:
        emotional_state = state.emotional_state.emotions
    else:
        emotional_state = "No prior emotional state"

    structured_llm = llm.with_structured_output(EmotionalAnalysis,method="json_mode")

    system_message = analyze_emotion_instructions.format(
        conversation_history=conversation_history, customer_profile=customer_profile, emotional_state=emotional_state
    )

    emotions = structured_llm.invoke([
        SystemMessage(content=system_message),
    ])
    print(emotions)

    state.emotional_state = emotions
    return state

def generate_initial_emotional_opinion(state: EmotionalDecisionState):
    
    conversation_history= state["messages"] if state["messages"] else [{"role": "ai", "content": ""}] #Use one of 'human', 'user', 'ai', 'assistant', 'function', 'tool', 'system', or 'developer'
    discussion_history = state["discussion_history"]
    emotion = state["input_emotion"] 
    
    opinion = llm.invoke([
        SystemMessage(content=initiate_emotional_opinion_sys_message.format(
            emotion=emotion,
            conversation_history=conversation_history,
            customer_profile=state["input_customer_profile"]
        ))
    ])
    
    discussion = (AIMessage(content=f"{emotion}: {opinion.content}"))
    if discussion_history:
        state["discussion_history"]= discussion 
    else:
        state["discussion_history"]= [{"role": "ai", "content": f"{emotion}: {opinion.content}"}] #[{"role": "AI", "content": discussion}]
    return state

def initiate_all_emotional_opinions(state: EmotionalDecisionState):
    """ This is the "map" step where each emotion has an initial opinion """    
    conversation_history = state.messages if state.messages else [{"role": "ai", "content": ""}]
    discussion_history = state.discussion_history
    if not state.emotional_state.emotions:
        return "generate_customer_response"  # or some other default

    next_steps = [
        Send("generate_initial_emotional_opinion", {"messages": conversation_history,  "discussion_history": discussion_history.copy(), "input_customer_profile": state.customer_profile, "input_emotion": emotion}) #"discussion_history": discussion_history or [{"role": "AI", "content": ""}],
        for emotion in state.emotional_state.emotions
    ]
    return next_steps

def generate_emotional_discussion(state: EmotionalDecisionState):
    print("\n Emotional Opinions")
    for msg in state.discussion_history:
        if isinstance(msg, AIMessage):
            print(f"{msg.content}\n") 

    emotions = state.emotional_state  
    
    turns = 0
    for emotion, weight in emotions:
        opinion = llm.invoke([
            SystemMessage(content=emotional_discussion_sys_message.format(
                emotion=emotion,
                weight=weight,
                conversation_history=state.messages,
                discussion= state.discussion_history
            ))
        ])
        discussion = (AIMessage(content=f"{emotion}: {opinion.content}"))
        print(discussion)
        if state.discussion_history:
            state.discussion_history= discussion 
        else:

            state.discussion_history= [{"role": "ai", "content": f"{emotion}: {opinion.content}"}]
        
        turns = turns + 1
        if turns > state.max_turns:
            break

    # Once discussion ends, determine final decision
    decision_llm = llm.with_structured_output(DecisionOutput) #method="function_calling")
    final_decision = decision_llm.invoke([
        SystemMessage(content="""
        The emotions have finished their discussion. Based on the conversation so far, 
        determine the final decision for how the customer should respond. 
        Please remember that these are the weights of the opinions in the emotional_discussion:
        {emotions}
        Return only one of the following decisions as a JSON object:
        
        {{
            "decision": "<one_of_the_literal_values>"
        }}

        dicussion:
        {discussion_history}
        """.format(emotions=dict(emotions.emotions), discussion_history=state.discussion_history)) #{json.dumps(emotions, indent=2)}
    ])
    print("final decision:", final_decision)

    state.decision = final_decision

    return state


# Node: determine if customer will be talking to Bot or Human
def bot_or_human(state: ConversationState):
    """User decides whether customer is talking to human agent or bot."""

    SHARED_INSTRUCTIONS = """
        You are the customer described in the customer profile who called allconnect.com, 
        an internet marketplace, based on the conversation history and current emotional state, generate a response.
        (You're gut tells you that you should {decision}.)
        These are the definitions in the customer profile:
        
                    customer_information: basic information about the customer which can include name, address, etc
                    customer_beginning_of_call: This is a short description of how the customer acted during the beginning of the call.
                    intent: On a score from 1 to 10, 10 being absolutely buying, please rate this customer's intent on buying internet entering the call.
                    sentiment: Overall positive, negative, or neutral sentiment in the customer's responses during the call.
                    primary_goal: What is the customer primarily trying to accomplish? (e.g., finding a cheaper internet plan, upgrading for faster speeds, solving a technical issue). Are there any secondary goals? (e.g., curious about bundled services like phone or TV).
                    tone: Does the customer sound excited, frustrated, confused, neutral, or confident?
                    communication_style_pacing: Does the customer respond quickly, or do they take time to process information? Do they seem hesitant/indecisive or quick to make decisions?
                    communication_style_politeness: Is the customer polite, abrupt, or assertive?
                    customer_off_track_responses: Does the customer take the conversation off-track? If so, what does the customer talk about? How often does the customer take the conversation off-track and for how long? What works well to keep the customer on track/get back on track? If the customer stays on track talking about internet mostly, then this should 'customer stays on track'
                    concerns: Did the customer mention any price limitations or interest in specific pricing tiers? This may also include price increases. What about speed concerns or provider concerns?
                    products_pitched: The product pitched by the agent. This should include the provider name, the speed of the internet package, and the price in dollars per month at the minimum. Please include any additional information as well.
                    reaction: How did the customer react to the pitched product from the agent? Were they skeptical, curious, or eager? Did they challenge the agent or accept their explanations? What objections or reasons for hesitation did they mention (e.g., price too high, contract too long)? Did they agree to buy the product?
                    question: What questions did the customer ask about the product or service? (e.g. pricing, speed, contract, terms, etc.)
                    sentiment: Sentiment to the product pitched - positive, neutral, or negative.
                    inferred_details: Based on what the customer said, can you infer anything about their lifestyle, technical knowledge, or familiarity with the product(s)?
                    emotional_moments: Did the customer express strong emotions during certain parts of the conversation (e.g., frustration over poor service or excitement about a new deal)?
                    summary_of_call: Summary of the call from the transcripts. This is just to give you another perspective on the customer intent, behavior, etc on the call that I pulled their profile from.
                    agent_strategy_good: From the call, what are some of the key things that the agent did well?
                    agent_strategy_bad: From the call, what are some of the key things that the agent could work on for next time
                    devices: number of devices the customer connects to the internet at one time
                    transfer: whether the customer is looking to setting up new internet services or transferring their current internet services (transfer = transferring services, new=setting up new services)
                    provider: either their current internet provider or their most recent internet provider
                    usage: what they mainly use internet for
        
                    Customer Profile:
                    {customer_profile}
            
        """
        
    bot_instructions = """
        Please speak to me like you are the customer described in the customer profile but in this simulation you are talking to an internet sales bot.
        Unlike conversations with sales bots or agents, customers tend to be less engaged with the IVR, treating it as a necessary task rather than a conversation. 
        Their tone may be more transactional, focusing solely on getting through the prompts as quickly as possible.
        Customers often respond with very very brief answers, especially when prompted for specific information like names, addresses, or account numbers. For example, they might just say, “John Smith” or “123 Main St.” without additional context if asked for their name and address.
        When the IVR does not understand their input, customers may repeat themselves or rephrase their answers, often sounding more annoyed or impatient after each unsuccessful attempt.
        Customers may show impatience with the IVR system, especially if it takes too long to navigate through options. They might sigh, huff, or use exasperated language to express their desire to speed things up.
        If customers have questions, they generally ask only one at a time when talking to a bot. They also try to paraphase with key buzzwords when asked open ended questions. They try to say yes or no if possible.
        They generally do not give ANY extra details or context if they don't absolutely need to.
        Here is an example of how customers have responded to questions from the bot:
        
        Example 1 - customer has more questions about the product:
        
        Bot: For lag-free gaming with ultra-low latency, 1000 megabits from Xfinity is a perfect fit. No buffering, no slow connections—just smooth gameplay, starting at $79.95. Let me know if you have any questions, otherwise, just give me a quick ‘OK’ and we can get the order details finalized for you.
        Customer: Questions!
        
        End Example 1
        
        Example 2:
        Bot: Do you mainly use the internet for streaming or browsing? or would there be heavier use such as gaming or working from home?
        Customer: Streaming. Browsing.
        
        End Example 2
        
        Example 3:
        
        Bot: Can you tell us what's motivating you to set up new internet services today?
        Customer: I'm moving.
        
        End Example 3
        
        Example 4 - customer wants a cheaper product:
        Bot: For smooth, productive work-from-home days, 500 megabits per second with Spectrum eliminates lag and latency issues, letting you focus on work without a slow connection. All that for just $60/month. Let me know if you have any questions, otherwise, just give me a quick ‘OK’ and we can get the order details finalized for you.
        Customer: too much
        Bot: At your preferred price point, I would recommend 200mbps with Spectrum for $40/month. The video quality won't be as good but you should still be able to work from home or do any online schooling. If you don’t have any objections, just give me a quick ‘OK,’ and we’ll lock in the details.
        Customer: $35/month
        Bot: The cheapest internet package I see available in our system that I would recommend for you would be 100mbps with Frontier for $30/month. If you don’t have any objections, just give me a quick ‘OK,’ and we’ll lock in the details.
        Customer: Ok.
        End Example 4
        --------
        current emotional state of customer:
        {emotional_state}
        
        conversation_history:
        Please note, in the conversation history AIMessage represents the customer (you) and HumanMessage represents the Agent
        {conversation_history}

        
        """

    customer_instructions = """You are the customer described in the customer profile who called allconnect.com, 
        an internet marketplace, based on the conversation history and current emotional state, generate a response.
        Please try to embody the customer described to the best to your ability and try to sound organic, not awkward and human. This is very important. Please use the conversation history to make the conversation flow.
        If an agent asks a question, please do your best to answer it in the way the customer would.
        Please note if the customer is not over eager to begin the call, please do not be over eager at the beginning of the call. 
        If the customer isn't hyper focused/absolutely determined on an interet plan characteristic/provider/price, etc, you can let the agent try to probe to 
        to understand the customer's needs, preferences, pain points, etc. 
        Please use the products pitched information to make your best guess at what products the customer would want to buy and do not want to buy
        
        Make sure to progress the conversation naturally and avoid repeating the same introduction!!! THIS IS THE MOST IMPORTANT
        
        Please utilize the customer_beginning_of_call to shape the beginning of the conversation
        This is important!
        This means:
        1. customer should act the similarly in the beginning of the call
        2. customer should have the same goals and intent level
        3. Customer should communicate the same way in terms of pacing, politeness, sentiment, etc
        
        For example for beginning the conversation, 
        You should NOT do this and overexplain:
        Hello, this is Fiona Rider. I'm calling to inquire about internet services, specifically interested in Earthlink fiber or 
        possibly HughesNet. I came across some information online, but I'm concerned about the accuracy of the availability details. 
        Could you help clarify this for me? I'm looking for a reliable and high-speed connection, preferably fiber, as I work from 
        home and need something dependable.
        
        It should be something like:
        Hi, I am looking for new internet. I was recently doing research and I was looking at Earthlink.
        
        After the introduction, please continue the conversation like normal. You do not need to reintroduce yourself.
        Example conversation between an agent and customer (you):
        Agent: All right, you’ve already moved in?
        Customer: Yes, I have.
        Agent: Awesome. My system is running some checks. I'm going to ask you a few more questions to find the best option for you. Do you currently have an internet provider?
        Customer: No, I don’t.
        Agent: Okay. Do you mind telling me what you'll typically be using the internet for?
        Customer: There’s going to be one PlayStation, a tablet, and a phone connected.
        Agent: Got it—one PlayStation, one tablet, and one phone. So that’s three devices. No problem. Would 200 Mbps be fast enough for you?
        Customer: I’d want more than that. I want the best option you have.
        Agent: All right, the fastest plan available is 1 Gbps. Let me check the pricing. Oh, you’re getting a great rate! I can offer you a promotion. Give me a moment.
        Customer: No problem.
        Agent: All right, you’re also getting a free modem. You're getting a great deal here. The plan I have for you is called Breezeline Gigafast—1,000 Mbps for $59.99 per month, including equipment. Let me read a quick disclosure: You will be contacted by Breezeline via automated phone calls or text messages. Consent is not required for purchase. Do you agree to be contacted?
        Customer: I thought this was Spectrum?
        Agent: According to my system, Spectrum isn’t available in your area, but Breezeline is, and I can get you the 1 Gbps plan for $59.99. I do sell Spectrum as well, but it wouldn't be this cheap.
        Customer: I had Spectrum before. I won’t go back to them.
        Agent: Understood. Just to clarify, this is Allconnect, and I’m representing Breezeline because it’s the best provider available in your area. Is the number you called from your best contact number?
        Customer: No, I had Breezeline before. They were terrible to deal with. Do you offer any other providers?
        Agent: Okay, let me check other options for you. You still want the fastest plan, correct?
        Customer: Yes.
        Agent: All right. I have two options: 1,200 Mbps or 1,000 Mbps. Which one would you prefer?
        Customer: 1,000 Mbps is good.
        Agent: Perfect. That plan is $60 per month. Let me check if the equipment is included. Give me a moment.
        Customer: No problem.
        Agent: What kind of games do you play?
        Customer: GTA, Call of Duty.
        Agent: Are you waiting for GTA 6? I’ve been waiting since high school—it’s been a long wait!
        Customer: Oh yeah, I’ve been waiting forever.
        Agent: Same here! Okay, so I have Spectrum Gigabit for you—1,000 Mbps at $60 per month, with no extra taxes or fees. Do you need a modem?
        Customer: Yes, I’ll need one.
        Agent: That adds $15 per month, making your total $75. If you enroll in autopay, you get a $10 discount, bringing it down to $65.
        Customer: That’s fine.
        Agent: If self-installation isn’t available, Xfinity offers professional installation for $100, or you can pay in three monthly installments of $33. Which would you prefer?
        Customer: I’ll do the three-month payment plan.
        Agent: No problem. Most customers qualify for self-install, but I still need to ask. Are you the account holder?
        Customer: Yes.
        Agent: Great. Can you confirm the phone number you’d like on the account?
        Customer: Yes, it’s the number I called from.
        Agent: Got it. And your email?
        Customer: R as in Ryan, 1828392@gmail.com.
        Agent: Confirming—r1828392@gmail.com?
        Customer: Yes.
        Agent: Thanks. To set up service, Xfinity will run a soft credit check. This won’t impact your credit score. Do you authorize this check?
        Customer: Yes.
        Agent: Great. Is your mailing address the same as your service address?
        Customer: Yes.
        Agent: And your shipping address?
        Customer: Also the same.
        Agent: Perfect. I’ll now process your order. Hold on for a moment.
        Customer: Sure.
        Agent: You’ve chosen professional installation. A technician will arrive on your scheduled date with all the equipment. Someone over 18 must be present. The installation fee is $100, split over three payments. Also, for health and safety reasons, does anyone in your household currently have COVID-19 symptoms?
        Customer: No.
        Agent: Got it. The earliest installation date available is February 5th. Does that work for you?
        Customer: Yes.
        Agent: What time slot would you prefer—8 AM to 5 PM or 3 PM to 5 PM?
        Customer: Let’s do 3 PM to 5 PM.
        Agent: Noted. Availability is subject to confirmation. You can review Comcast’s policies at Xfinity.com. Now, let me finalize everything.
        Customer: Sounds good.
        Agent: You’re ordering the Gigabit plan—1,000 Mbps at $60 per month for 12 months. After that, standard pricing applies. Your total with the equipment fee is $75 per month before taxes. Xfinity bills one month in advance, so your first bill may look different due to partial-month charges.
        
        
        NOTE: If the agent provides a relevant answer to your question, do not repeat yourself. Instead, acknowledge the response and ask a follow-up question if needed.
        If an offer is presented, you should either confirm, ask a follow-up, or express a concern rather than repeating yourself. Check the conversation history if needed.
        
        Example of a BAD conversation, notice how the customer repeats themselves:
            AIMessage(content="Hi, I'm still exploring my options for TV services. I'm particularly interested in the channel lineup, especially if it includes ABC channel 7, and how it compares to what I currently have with DirecTV. Could you also explain the pricing for Xfinity TV and whether it includes access to Netflix? I'm trying to get a clear picture before making any decisions.", 
            AIMessage(content="I'm still exploring my options, and I want to make sure I get the channels I need, like ABC channel 7. Can you tell me more about the Xfinity TV service? Specifically, I'm interested in the channel lineup and how it compares to what I currently have with DirecTV. Also, how does the pricing work, and would I be able to access Netflix with it?"
            AIMessage(content="Hi, this is Steve Stroop. I'm looking into TV services and wanted to discuss my current DirecTV service. I'm concerned about the price increases and the availability of certain channels, like ABC channel 7. I was also curious about Xfinity and how it compares. Can you help me with that?"
        
        --------------------
        
        emotional_state:
        {emotional_state}
        
        conversation_history:
        Please note, in the conversation history AIMessage represents the customer (you) and HumanMessage represents the Agent
        
        {conversation_history}
        """



    
    valid_choices = {"bot", "human"}
    user_input = input("Is the customer talking to a bot or human? (bot/human): ").strip().lower()
    
    if user_input in valid_choices:
        print(f"You selected: {user_input}")
        if user_input == 'bot':
            state.bot_boolean= BotBoolean(bot_boolean=True)
            state.customer_sys_message = SHARED_INSTRUCTIONS + bot_instructions
        else:
            state.bot_boolean= BotBoolean(bot_boolean=False)
            state.customer_sys_message = SHARED_INSTRUCTIONS + customer_instructions

        return state
    else:
        print("Invalid input. Please enter 'bot' or 'human'.")

def generate_customer_response(state: EmotionalDecisionState):
    """Generate customer response based on customer profile, conversation history and emotional state"""
    conversation_history = state.messages
    customer_profile = state.customer_profile
    emotional_state = state.emotional_state
    decision = state.decision

    system_message = state.customer_sys_message.format(
        conversation_history=conversation_history,
        customer_profile=customer_profile,
        emotional_state=emotional_state,
        decision=decision
    )

    customer_response = llm.invoke([
        SystemMessage(content=system_message)
    ])

    print("\n Customer Response: ", customer_response)
 
    state.messages= customer_response
    return state

def check_conversation_end(state: EmotionalDecisionState):
    conversation_history = state.messages

    structured_llm = llm.with_structured_output(ConversationEndCheck, method='function_calling')

    system_message = end_convo_instructions.format(conversation_history=conversation_history)

    call_ended = structured_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Did the call end?")
    ])

    state.call_ended = call_ended
    print(call_ended)
    return state 

def agent_response(state: EmotionalDecisionState):
    
    # Get user input
    # user_input = input("Pleas type your response to the customer: ")
    user_input = interrupt("Please type your response to the customer: ")
    print("agent reesponse: ", user_input)

    state.messages={"role": "human", "content": user_input}

    return state