from phi.agent import Agent
from phi.model.openai import OpenAI
from phi.tools.duckduckgo import DuckDuckGo

# Initialize LLM and search tool
llm = OpenAI(model="gpt-4", api_key="your-openai-api-key")
search = DuckDuckGo()

import warnings
warnings.filterwarnings("ignore")

# Agent Definitions
idea_agent = Agent(
    name="IdeaValidator",
    model=llm,
    tools=[search],
    instructions=[
        "You are a startup idea validation expert.",
        "Evaluate ideas for market need, uniqueness, and risks.",
        "Give a score out of 10 with reasoning.",
        "Keep it short and structured."
    ],
    show_tool_calls=True,
    markdown=True
)

market_agent = Agent(
    name="MarketResearcher",
    model=llm,
    tools=[search],
    instructions=[
        "You are a market research specialist.",
        "Estimate TAM/SAM/SOM, identify top competitors and trends.",
        "Summarize clearly in bullet points."
    ],
    show_tool_calls=True,
    markdown=True
)

biz_agent = Agent(
    name="BizModeler",
    model=llm,
    tools=[search],
    instructions=[
        "You are a business model expert.",
        "Propose revenue streams, pricing, and key metrics.",
        "Keep suggestions concise."
    ],
    show_tool_calls=True,
    markdown=True
)

pitch_agent = Agent(
    name="PitchGenerator",
    model=llm,
    tools=[],
    instructions=[
        "You are a startup pitch expert.",
        "Create a short, crisp 3-4 sentence elevator pitch based on the startup idea, market, and business model.",
        "Be persuasive and clear."
    ],
    markdown=True
)

# Main Program
print("Multi-Agent Startup Advisor")
print("=" * 50)
idea = input("Enter your startup idea: ").strip()
if not idea:
    print("Please provide an idea.")
else:
    print("\nValidating Idea...")
    idea_response = idea_agent.run(f"Validate this idea: {idea}")
    print(idea_response.content)

    print("\nConducting Market Research...")
    market_response = market_agent.run(f"Conduct market research for: {idea}")
    print(market_response.content)

    print("\nDesigning Business Model...")
    biz_response = biz_agent.run(f"Design a business model for: {idea}")
    print(biz_response.content)

    print("\nGenerating Elevator Pitch...")
    pitch_prompt = (
        f"Here is the idea:\n{idea}\n\n"
        f"Validation:\n{idea_response.content}\n\n"
        f"Market:\n{market_response.content}\n\n"
        f"Business:\n{biz_response.content}\n\n"
        f"Generate an elevator pitch."
    )
    pitch_response = pitch_agent.run(pitch_prompt)
    print(pitch_response.content)




