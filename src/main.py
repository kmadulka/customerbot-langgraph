from .graph_compiler import build_graph
from langgraph.checkpoint.memory import MemorySaver
from .utils import generate_random_customer_profile, clean_nans
import pandas as pd
import os
from time import sleep
from langgraph.graph.state import Command


def main():
    customer_bot_graph = build_graph()
    memory = MemorySaver()
    customer_bot_compiled = customer_bot_graph.compile(checkpointer=memory)

    current_directory = os.getcwd()
    # Navigate up two levels (assuming your structure is 'repo-root/src/data')
    repo_root_directory = os.path.abspath(os.path.join(current_directory, '..'))
    # Construct the path to the data folder
    file_path = os.path.join(repo_root_directory, 'data', 'customer_profile_samp_modified.csv')
    customer_profile_df = pd.read_csv(file_path)

    customer_profile = clean_nans(generate_random_customer_profile(customer_profile_df))
    thread = {"configurable": {"thread_id": "1"}}
    customer_bot_compiled.invoke({"customer_profile": customer_profile, "messages": [{"role": "human", "content":""}]},
        thread,
        stream_mode="updates"
    )

    while customer_bot_compiled.get_state(thread).next: #need while loop if there is feedback
        graph_state = customer_bot_compiled.get_state(thread)
        interrupt_value = graph_state.tasks[0].interrupts[0].value

        # Occasionally, the previous print statement is not visible in the console.
        sleep(0.5)

        human_feedback_text = input(interrupt_value) #input stored here
        print(f"\nAgent Response: {human_feedback_text}\n\n")

        # config["callbacks"] = [MLflowTracerOverrideWarnings()]
        customer_bot_compiled.invoke(Command(resume=human_feedback_text), config=thread)

if __name__ == "__main__":
    main()



